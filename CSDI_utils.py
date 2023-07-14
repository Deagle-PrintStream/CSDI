import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

__all__=["train","evaluate"]

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval:int=5,
    foldername:str="",
):
    """ train the model with given config files and dataset by `config` and `train_loader`,
        save the model under to `foldername`
        
        PARAMETER
        ------
        `model`: subclass of `nn.Module`, target model to train \\
        `config`: `dict`, only  `lr` and `epochs` are transfered\\
        `train_loader`: `DataLoader` for training\\
        `valid_loader`: `DataLoader` for validation, default is None\\
        `valid_epoch_interval`: `int`, if `valid_loader` is not None, after each amount of this steps, check the performance\\
        `foldername`: folder to save the trained model
        """
    
    #initalize the optimizer
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    #learning rate decay with tricky milestones
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = np.inf 
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        #training part
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step() #this one should come after validation part?
        #validation part
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            batch_no:int=0 #number of batch
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0) #loss calculation for validation part
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

        #learning rate adjustment
        #lr_scheduler.step()

    #save the model
    output_path:str=""
    if foldername != "":
        output_path = foldername + "/model.pth"
    else:
        output_path="./model.pth"
    if foldername != "":
        torch.save(model.state_dict(), output_path)

"""sub indicator functions for evalutation"""

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    """quantile loss """
    #conversion from tensor to float TODO
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    ) #type:ignore


def calc_denominator(target, eval_points):
    """calculate the denominator to normalize quantile_loss """
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    """ Calculate continuous ranked probability score, integrated of the quantile loss from all quantile levels\\
        We actually approximates CRPS with discretized quantile levels with 0.05 ticks """
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    #unknown member of float type item() TODO
    return CRPS.item() / len(quantiles) #type:ignore


def evaluate(model, test_loader, nsample:int=100, scaler:float=1, mean_scaler:float=0, foldername=""):
    """evaluate the performance of model on testment dataset
    
    PARAMETER
    =====
    `model`: target model\\
    `test_loader`: `DataLoader`\\
    `nsample`: sample amount to approximate the probability distribution\\ 
    `scaler` `float`, scaler for CPRS, default is 1\\
    `mean_scaler`: `float`, offset of all targets within CPRS ,default is 0\\
    `foldername`: path to save output samples    
    """

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                "Root mean squared error RMSE"
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                "Mean absolute error MAE"
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            #save the predicted imputated samples
            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            #save the performance indicators
            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
