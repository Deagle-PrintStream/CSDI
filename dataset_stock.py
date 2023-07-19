import pickle
import logging
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# 5 attributes
#attributes = ["open", "high", "low", "close", "volume"]
attributes = ["open", "high", "low", "close"]

class Stock_Dataset(Dataset):
    def __init__(
        self,
        stock_name: str="SH",
        use_index_list=None,
        missing_ratio: float = 0.0,
        seed: int = 0,
        window_size: int = 100,
        slide_step: int = 100,
    ) -> None:
        # super().__init__()

        csv_path = f"./data/stock/{stock_name}.csv"
        pk_path = "./data/stock_{}_missing{}_seed{}.pk".format(stock_name,missing_ratio,seed)
        self.observed_values = np.array([])
        self.observed_masks = np.array([])
        self.gt_masks = np.array([])
        self.window_size = window_size
        self.slide_step = slide_step

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        # if dataset with given missing ratio is precessed already
        if os.path.isfile(pk_path) == True:
            with open(pk_path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
            if use_index_list is None:
                self.use_index_list = np.arange(len(self.observed_values))
            else:
                self.use_index_list = use_index_list
            return

        logging.info(
            f"dataset loaded with stock {stock_name} and missing_ratio {str(missing_ratio)} and seed {str(seed)} "
        )
        if os.path.isfile(csv_path) == False:
            raise FileNotFoundError("stock price dataset not found")

        observed_values = []
        observed_masks = []
        gt_masks = []
        # read raw stock price data
        df = pd.read_csv(csv_path, index_col="date")
        df.reset_index(drop=True, inplace=True)
        df = df.fillna(method="ffill")
        df=df[attributes]
        size = len(df)
        # Z-score normalization for entire dataset
        df=self.normalize(df)
        # static mask
        ob_mask = np.ones((window_size, len(attributes)))
        gt_mask_ind = np.arange(int(window_size * (1-missing_ratio)), window_size)
        gt_mask = np.ones((window_size, len(attributes)))
        gt_mask[gt_mask_ind, :] = 0
        # segment into slides with given length of `window_size`
        for start in range(0, size - window_size, slide_step):
            _data = df.loc[start:start + window_size-1].copy()
            #_data=self.normalize(_data)
            observed_values.append(_data)
            observed_masks.append(ob_mask)
            gt_masks.append(gt_mask)
        self.observed_values = np.array(observed_values)
        logging.info(f"dataset shape:{self.observed_values.shape}")
        self.observed_masks = np.array(observed_masks)
        self.gt_masks = np.array(gt_masks)

        self.use_index_list = np.arange(len(self.observed_values))

        with open(pk_path, "wb") as f:
            pickle.dump([self.observed_values, self.observed_masks, self.gt_masks], f)

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.window_size),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
    
    @staticmethod
    def normalize(df:pd.DataFrame)->pd.DataFrame:
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
        return df

def get_dataloader(stock_name:str="SH",
    seed=1, nfold: int = 0, batch_size=16, missing_ratio=0.1
):  # minor type bug fixed

    # only to obtain total length of dataset
    dataset = Stock_Dataset(stock_name=stock_name, missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indlist)
    """dataset division: 0.2 for test, 0.75 for train, 0.05 for validation"""
    train_ratio = 0.75
    test_ratio = 0.2
    valid_ratio = 1 - test_ratio - train_ratio

    test_index = indlist[int(len(indlist) * (1 - test_ratio)) :]
    remain_index = np.delete(indlist, test_index)
    np.random.seed(seed)
    np.random.shuffle(remain_index)
    train_index = remain_index[0 : int(len(indlist) * train_ratio)]
    valid_index = np.delete(indlist, train_index)

    logging.info(
        f"dataset size:{len(dataset)},training ratio:{train_ratio},\
        validation ratio:{valid_ratio},test ratio:{test_ratio},test fold No. {nfold}."
    )
    print("dataset loading start")

    train_dataset = Stock_Dataset(
        stock_name=stock_name,
        use_index_list=train_index,
        missing_ratio=missing_ratio,
        seed=seed,
    )  # lot of time cost, should be diminished since `dataset` has inited
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # minor type bug fixed
    valid_dataset = Stock_Dataset(
        stock_name=stock_name,
        use_index_list=valid_index,
        missing_ratio=missing_ratio,
        seed=seed,
    )  
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = Stock_Dataset(
        stock_name=stock_name,
        use_index_list=test_index,
        missing_ratio=missing_ratio,
        seed=seed,
    )  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logging.info("dataset loading completed")
    print("dataset loading completed")
    return train_loader, valid_loader, test_loader
