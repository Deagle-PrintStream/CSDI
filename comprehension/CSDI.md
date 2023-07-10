# CSDI: model reimplementation and optimization

## Task
> CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation
> 1. read the paper, list out: research background, methodology, major results, innovative points.
> 2. read the code, list out: each modules and their functions, overview struct.
> 3. simple modifications upon raw code to make it work.
> 4. add more details into essential modules
> 5. test the code upon `healthcase` dataset, compare the results with original paper.
> 6. data visialization upon `healthcase` dataset
> 7. point out defaults of code or paper
> 8. make optimizations and compare the results.

## Summary of paper

1. main task
   1. impute missing values using machine learning techniques within time series data.
2. innovations
   1. proposed a conditional diffusion model for time series imputation
   2. and a self-supervised training method to handle the missing values
3. model architecture
4. results
   1. improves the continuous ranked probability score(CRPS) over existing probabilistic methods
   2. decreases the mean absolute error (MAE) compared with sota
   3. can be applied to time series interpolations or  forecasting for one step further

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
   5. normalize each attributes of all patients with $\tilde{\bold{x}}=\frac{\bold{x}-\mu}{\sigma^2}$
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
2. 

## Details in essential modules

## Testment of sample dataset

## Data visualization

## Defaults


## Optimizations



