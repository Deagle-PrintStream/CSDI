import pickle
import logging

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']


def extract_hour(x):
    h, _ = map(int, x.split(":"))
    return h


def parse_data(x):
    """extract the last value for each attribute"""
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_id(id_, missing_ratio=0.1):
    data = pd.read_csv("./data/physio/set-a/{}.txt".format(id_))
    # set hour
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

    # create data for 48 hours x 35 attributes
    observed_values = []
    for h in range(48):
        observed_values.append(parse_data(data[data["Time"] == h]))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def get_idlist():
    """get patient id from file names in 6 digital form and sort"""
    patient_id = []
    for filename in os.listdir("./data/physio/set-a"):
        match = re.search(r"\d{6}", filename)  # vscode warning fixed
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id


class Physio_Dataset(Dataset):
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = (
            "./data/physio_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        )

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            logging.info(
                "dataset loaded with physio_missing "
                + str(missing_ratio)
                + " and seed"
                + str(seed)
            )
            idlist = get_idlist()
            for id_ in idlist:
                try:
                    observed_values, observed_masks, gt_masks = parse_id(
                        id_, missing_ratio
                    )
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.gt_masks.append(gt_masks)
                except Exception as e:
                    print(id_, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)

            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
            tmp_values = self.observed_values.reshape(-1, 35)
            tmp_masks = self.observed_masks.reshape(-1, 35)
            mean = np.zeros(35)
            std = np.zeros(35)
            for k in range(35):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()
            self.observed_values = (
                (self.observed_values - mean) / std * self.observed_masks
            )

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(
    seed=1, nfold: int = 0, batch_size=16, missing_ratio=0.1
):  # minor type bug fixed

    # only to obtain total length of dataset
    dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    test_ratio = 0.2
    partition_size = int(len(dataset) * test_ratio)
    start = nfold * partition_size
    end = (nfold + 1) * partition_size
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    """dataset division: 0.2 for test, 0.7 for train, 0.1 for validation"""
    train_ratio = 0.7
    logging.info(
        f"dataset size:{len(dataset)},training ratio:{train_ratio},\
        validation ratio:{1-test_ratio-train_ratio},test ratio:{test_ratio},test fold No. {nfold}."
    )
    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * train_ratio)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_dataset = Physio_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    ) #lot of time cost, should be diminished since `dataset` has inited
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # minor type bug fixed
    valid_dataset = Physio_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )#lot of time cost 
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = Physio_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )#lot of time cost 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader
