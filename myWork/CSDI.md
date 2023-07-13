# CSDI: model reimplementation and optimization

## Task
> CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation
> 1. read the paper, make a summary
> 2. read the code, understand the project architecture
> 3. simple modifications upon raw code to make it work.
> 4. add more details in essential modules
> 5. test the code upon `healthcase` dataset,
> 6. data visialization upon `healthcase` dataset
> 7. point out defaults of code or paper
> 8. make optimizations and compare the results.

## Summary of paper

1. main task
   1. impute missing values using machine learning techniques within time series data.
2. innovations
   1. proposed a conditional diffusion model for time series imputation
   2. and a self-supervised training method to handle the missing values
3. formula expression
   1. $X=\{x_{1:K,1:L}\}\in \mathbb{R}^{K\times L}$ is the origin time series, where $K$ is  the number of features , $L$ is the length of timespan. $M=\{m_{1:K,1:L}\}\in \{1,0\}^{K\times L}$ is the observation mask. Timestamp is $s=\{s_{1:L}\}$. Each time series is expressed as $\{\bold{X,M,s}\}$
   2. forward process in model is defiend by Markov chain:$$q(\bold{x}_{1:T} | \bold{x}_0)\coloneqq \prod_{t=1}^T q(x_t|x_{t-1}), q(x_t|x_{t-1})\sim \mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t \bold{I})$$, where $\beta_t$ is the small positive constant to constrain the input noise.$\hat{\alpha}\coloneqq 1-\beta_t,\alpha_t\coloneqq \prod_{i=1}^t\hat{\alpha}_i$. 
   3. Reverse process denoises is defined by Markov chain as well:$$p_\theta(\bold{x}_{0:T})\coloneqq p(\bold{x}_T)\prod_{t=1}^T{p_\theta(\bold{x}_{t-1} | \bold{x}_{t})},\bold{x}_T\sim \mathcal{N}(\bold{0,I})$$
   4. denoising diffusion probabilistic models (DDPM) with parameters:$$\bold{\mu}_\theta(\bold{x}_t,t)=\frac{1}{\alpha_t}\bigg({\bold{x}_t-\frac{\beta_t}{\sqrt{1-\alpha_t}}\epsilon_\theta(\bold{x}-t,t)}\bigg)\\
   \sigma_\theta(\bold{x}_t,t)=\left\{\begin{array}{l}\frac{1-\alpha_{t-1}}{1-\alpha_{t}}\beta_t&&t>1\\\beta_1&&t=1
   \end{array}\right.
$$
   5. Our goal is to predict conditional probability of $q(\bold{x}^{co}_0 | \bold{x}^{co}_0)$. Whereby adding an input $\epsilon_\theta$ into prediction noise netwrok to represente $x_0^{co}$, loss function turns into:$$\hat{\theta}=\arg\min\limits_{\theta}\mathcal{L}(\theta)\coloneqq\arg\min\limits_{\theta}\mathbb{E}_{x_0\sim q(x_0),\epsilon\sim\mathcal{N}(0,\bold{I}),t}{\lVert\epsilon-\epsilon_\theta(x_{t}^{ta},t | x^{co}_0)   \rVert^2_2}$$
4. model architecture
![model architecture](./model%20architecture.jpg)
   1. training: **self-supervised learning** like MLM, manually masked a part of oberved values as missing ones and train the model to impute this part. 
   2. four strategy of mask is provided:
      1. *random strategy*
      2. *historical strategy* exploit missing pattern among different samples
      3. *mix strategy* mix of above two strategies, avoid overfitting.
      4. *test pattern strategy* when we know the missing pattern in test dataset.
   3. right-most part procedure is the same as DDPM but with one more input $x^{co}_0$.
   4. 2-dimensional **attention mechanism** is used in each residual layer to capture temporal and feature dependencies.
   5. **side information** is provided for training along with $\epsilon_\theta$: time embedding $\bold{s}$ and categorical feature embedding for $K$ features.
   6. model is based on **DiffWave** and refined for time series imputation: replacing convolution kernel with transformer structure for capture of time series features.
1. results![result1](./probabilistic%20results.jpg)![result2](./deterministic%20results.jpg)
   1. improves the continuous ranked probability score(CRPS) over existing probabilistic methods
   2. decreases the mean absolute error (MAE) compared with sota
   3. can be applied to time series interpolations or forecasting for one step further.

## Structure of project

Functions of each source file:
1. `download.py`: download dataset from official website, saved in `./data/`
2. `exe_physio.py`: shell functions for execution
   1. essential arguments: config file, device, seed, test missing ratio, fold number for testing in 5-fold test, pretrained model to load, count of samples.
   2. load config settings: `config["model","diffusion","train"]` and save to `./save/physio_fold.../` in json form
   3. create the iterators for loading dataset by `get_dataloader()`
   4. initialize the model by `CSDI_Physio()`
   5. train the model by `train()`
   6. test the model by `evaluate()`
3. `dataset_physio.py`: load dataset from `./data/physio/set-a/*`
   1. divide dataset into 3 parts: training, testment, and validation
   2. create iterators for loading data, each with different observed values and masks 
   3. preset a ratio for missing values in attributes
   4. set the rest values as ground truth, aiming to train the model to predict the missing values that are hidden intentionally
   5. normalize each attributes of all patients with Z-score normalization $\tilde{\bold{x}}=\frac{\bold{x}-\mu}{\sigma^2}$, where mean and std omittes all missing values.
4. `main_model.py`:class for CSDI model
   1. `CSDI_physio` overwrite `process_data()`
   2. base class `CSDI_base` have following methods:
5. `diff_models.py`: kernel diffusion model implementation
6. `CSDI_utils`:training and evaluation function
   1. 

Project structure overview:
![structure](./structure.drawio)

## Simple modifications

1. minor bug fixs: 
   1. variable type declearations
   2. no more implicit type conversion
   3. complete different modes in accordance with input argument
2. an alterative dataset loading method for better visualization.

## Details in essential modules

## result reimplementation

## Data visualization

See: [data visualization on health care](./time_series_view.ipynb)

## Defaults of demo

*TODO*

## Optimizations

*TODO*

