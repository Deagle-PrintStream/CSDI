import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI
import logging

class CSDI_base(nn.Module):
    """kernel model for time series imputation based on DiffWave"""
    def __init__(self, target_dim, config:dict, device):
        super().__init__()
        """set members including: 
        dataset dimensions : embedded time, embedded feature, total
        configuration of diffusion model:
        hyperparameters: beta sequence, alpha sequence
        """
        if torch.cuda.is_available()==False and torch.backends.mps.is_available()==False: #type:ignore
            device="cpu"
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        
        if self.target_strategy=="random":
            self.get_mask=self.get_randmask
        elif self.target_strategy=="historical":
            self.get_mask=self.get_hist_mask
        elif self.target_strategy=="mix":
            self.get_mask=self.get_mix_mask
        elif self.target_strategy=="forecast":
            self.get_mask=self.get_forecast_mask

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        # parameters for diffusion models
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        self.num_steps = config_diff["num_steps"]

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        self.diffmodel = diff_CSDI(config_diff, inputdim=1 if self.is_unconditional == True else 2)

        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                np.sqrt(config_diff["beta_start"]), np.sqrt(config_diff["beta_end"]) , self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, feature_dim:int,d_model:int=128):
        """time embedding of s = {s_1:L} to learn the temporal dependency, shown in Eq (13)"""
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(2).expand(-1, -1, feature_dim, -1)

    def get_randmask(self, observed_mask):
        """mask oberseved values as missing ones by random strategy"""
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        """mask by Historical strategy"""
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            # draw another sample for histmask (i-1 corresponds to another sample)
            cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask
    
    def get_mix_mask(self,observed_mask, for_pattern_mask=None):
        """mask by Mix strategy"""
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        rand_mask=self.get_randmask(observed_mask) 
        cond_mask = observed_mask.clone()

        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if  mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask
    
    def get_fixed_mask(self,observed_mask:torch.Tensor):
        """mask a fixed pattern as unknown part(in forecast mode)"""
        MISSING_RATIO=0.10

        cond_mask=observed_mask.clone()
        num_time_steps = observed_mask.shape[2]
        cond_mask[:,:,int(num_time_steps*(1-MISSING_RATIO)):]=0
        return cond_mask

    def get_forecast_mask(self,observed_mask:torch.Tensor):
        """Mask observed values as missing ones with a weighted random strategy."""

        num_time_steps = observed_mask.shape[2]
        time_weights = torch.logspace(base=0.1,start=1,end=0,steps=num_time_steps).to(("cuda:0")) # Increasing weights from 0 to 1

        cond_mask=observed_mask.clone()
        for i in range(len(cond_mask)):
            mask=torch.rand_like(observed_mask[i])*time_weights
            sample_ratio = np.random.rand()  # missing ratio    
            k=   int(sample_ratio*observed_mask[i].sum().item())
            mask=mask.reshape(-1)
            mask[mask.topk(k).indices]=-1
            mask=mask>0
            cond_mask[i]=mask.reshape(cond_mask[i].shape).float()
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        """ time embedding and categorical feature embedding for K features"""
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp,feature_dim=K, d_model=self.emb_time_dim) 
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        """get loss function curve in all epoch"""
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        """calculate loss function"""
        #major time consumer
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = torch.sqrt(current_alpha ) * observed_data + torch.sqrt(1.0 - current_alpha)  * noise #sqrt from numpy or torch  TODO

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        #loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1) #another time consumer line TODO
        try:    
            coeff=1/num_eval
            loss=torch.pow(residual , 2).sum() * coeff
        except ValueError:
            loss=torch.pow(residual , 2).sum()
        finally:
            return loss #type:ignore

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """concanate time embedding with feature ones, not actually transfering input into diffWave model"""
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        """time series imputation with given observed values and side information

        PARAMETER
        ------
        `observed_data`: observed data array and not masked\\
        `cond_mask`: conditional mask array\\
        `side_info`:time and feature embedding input\\
        `n_samples`: count of samples to impute
        """
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            noisy_cond_history:list = []    #expand variable scope
            if self.is_unconditional == True:
                noisy_obs = observed_data
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = np.sqrt(self.alpha_hat[t]) * noisy_obs + np.sqrt(self.beta[t]) * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            #reverse process defined as Markov chain
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    #same as set_input_to_diffmodel, replacing noise with current sample
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / np.sqrt(self.alpha_hat[t])
                coeff2 = (1 - self.alpha_hat[t]) / np.sqrt(1 - self.alpha[t]) 
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = np.sqrt(
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) 
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        """forward process defined as Markov chain"""
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        
        #summon mask array by preset strategy
        cond_mask=...
        if is_train == 0:#validation mode            
            cond_mask = gt_mask
        else:
            cond_mask=self.get_mask(observed_mask)        

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        """this should be renamed as `predict`, which doesn't actually calculate KPIs, return imputed samples """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp
    
    def process_data(self, batch)->tuple:
        """virtual method in base class, need to be overriden."""
        raise NotImplementedError("virtual method")


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        """override the base method"""
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)
        logging.info(f"CSDI model with parameters: {self.__dict__}")

    def process_data(self, batch):
        """convert batch data into :\\
            oberved data, mask for hidden, timestamp, gt mask, zero-array for cutting"""
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
