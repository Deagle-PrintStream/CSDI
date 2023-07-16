import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import logging

__all__=["train","evaluate"]

def train(
    model:torch.nn.Module,
    learning_rate:float,
    epoches:int,
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
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    #learning rate decay with tricky milestones
    p1 = int(0.75 * epoches)
    p2 = int(0.9 * epoches)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = np.inf 
    logging.info(f"training start with epochs:{epoches},learning_rate:{learning_rate}")
    for epoch_no in range(epoches):
        avg_loss = 0 
        loss_temp=torch.zeros((1,1)).to(("cuda")) #temp container to sum up total loss
        model.train(True) #set to training mode

        #training part
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            batch_no=0 #since we don't know the preset batch size, we have to get from iterator
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch) #major time cosumed as expected
                loss.backward()
                loss_temp+=loss 
                #by shifting the loss sum up calculation into GPU and decrease the hit of `item()`, time saved a lot
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        #"avg_epoch_loss": avg_loss / batch_no, #we no more know the average loss since we want to reduce the hit of `item()`
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )            
            lr_scheduler.step() #this one should come after validation part?
        avg_loss+=loss_temp.item()/batch_no
        #validation part
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval() #set to testing mode
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
                logging.info(f"best loss:{avg_loss_valid / batch_no},epoch:{epoch_no}")

        #learning rate adjustment
        #lr_scheduler.step()

    logging.info(f"training completed with best_valid_loss:{best_valid_loss}")
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

    tick:float=0.05
    quantiles = np.arange(tick, 1.0, tick)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom

    return CRPS.item() / len(quantiles) #type:ignore

def calc_RMSE(median,c_target,eval_points,scaler:float=1):
    """Root mean squared error RMSE"""
    return ((median - c_target) * eval_points) ** 2* (scaler ** 2)

def calc_MAE(median,c_target,eval_points,scaler:float=1):
    """Mean absolute error MAE"""
    return ((median - c_target) * eval_points) *scaler


def evaluate(model:torch.nn.Module, test_loader, nsample:int=100, scaler:float=1, mean_scaler:float=0, foldername=""):
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

    logging.info(f"evaluation start with nsample:{nsample}")
    with torch.no_grad():
        model.eval() #switch to test mode
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
                samples, c_target, eval_points, observed_points, observed_time = \
                    model.evaluate(test_batch, nsample) #type:ignore
                
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

                mse_current=calc_RMSE(samples_median.values,c_target,eval_points,scaler)
                mae_current=calc_MAE(samples_median.values,c_target,eval_points,scaler)

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
                logging.info(f"KPIs at batch_no:{batch_no}:MSE:{mse_current.sum().item()},MAE:{mae_current.sum().item()}")

        all_generated_samples = torch.cat(all_generated_samples, dim=0)
        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_observed_point = torch.cat(all_observed_point, dim=0)
        all_observed_time = torch.cat(all_observed_time, dim=0)
        #save the predicted imputated samples
        with open(
            foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
        ) as f:

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
        MSE=np.sqrt(mse_total / evalpoints_total)
        MAE=mae_total / evalpoints_total

        #save the performance indicators
        with open(
            foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
        ) as f:
            pickle.dump([MSE,MAE,CRPS,],f,)
        #print KPIs in entire process
        logging.info(f"RMSE:{ np.sqrt(mse_total / evalpoints_total)}")
        logging.info(f"MAE:{ mae_total / evalpoints_total}")
        logging.info(f"CRPS:{ CRPS}")
        print("RMSE:", np.sqrt(mse_total / evalpoints_total))
        print("MAE:", mae_total / evalpoints_total)
        print("CRPS:", CRPS)
