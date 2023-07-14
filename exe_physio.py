import argparse
import torch
import datetime
import json
import yaml
import os, sys  # for relative file path

from line_profiler import LineProfiler

from main_model import CSDI_Physio
from dataset_physio import get_dataloader
from CSDI_utils import train, evaluate

# python exe_physio.py --config test.yaml --modelfolder physio_fold0_20230714_162137

def parse_argument() -> argparse.Namespace:
    """read in arguments from command line, kernel arguments:
    configuration file, random seed, test missing ratio, folder number in CV,
    pretrained model to load(or not), count of samples.
    """
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--device", default="cuda:0", help="Device for Attack")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument(
        "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
    )
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)

    args = parser.parse_args()
    print(args)
    return args


def read_config(cfg_path: str, is_unconditional, test_missing_ratio: float):
    """load the configuration of testment,
    including `train`, `model` and `diffusion` part,
    `diffusion` part passed to `CSDI_Physio` only"""
    os.chdir(sys.path[0])
    path = "config/" + cfg_path
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except IOError as e:
        raise IOError("config file not found")

    config["model"]["is_unconditional"] = is_unconditional
    config["model"]["test_missing_ratio"] = test_missing_ratio

    print(json.dumps(config, indent=4))
    return config


def save_config(nfold: int, config) -> str:
    """save config file for later checkment"""
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/physio_fold" + str(nfold) + "_" + current_time + "/"
    print("model folder:", foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
    return foldername


if __name__ == "__main__":
    args = parse_argument()
    config = read_config(args.config, args.unconditional, args.testmissingratio)
    foldername = save_config(args.nfold, config)
    train_loader, valid_loader, test_loader = get_dataloader(
        args.seed,
        args.nfold,
        config["train"]["batch_size"],
        config["model"]["test_missing_ratio"],
    )
    """initialize the model"""
    model = CSDI_Physio(config, args.device).to(args.device)

    if args.modelfolder == "":

        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        pretrain_model_path = "./save/" + args.modelfolder + "/model.pth"
        model.load_state_dict(torch.load(pretrain_model_path))
    """test the model"""
    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

