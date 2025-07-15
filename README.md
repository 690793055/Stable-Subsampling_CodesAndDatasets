# Stable-Subsampling_CodesAndDatasets
The Codes and datasets of the paper "Stable Subsampling under Model Misspecification and Covariate  Shift"


This is a brief description for the experiments of the USSP algorithm project. This project contains all the programs used in Paper "Stable Subsampling under Model Misspecification and Data Shift", including both uniform design tables and real dataset data.
The programs are written in MATLAB and can be found in the "Code" folder.
The uniform design tables used in the project are located in the "uniform design tables" folder. If other uniform design tables are needed, they can be generated using the R package UniDOE.
The real dataset data of Section 6.1 and 6.2 are stored in the "Real dataset" folder. This includes China's urban air quality dataset collected from the website https://aqicn.org/city/ and Top 10 cities weather dataset of the United States, which can be obtained from the website https://www.kaggle.com/datasets/shubhamkulkarni01/us-top-10-cities-electricity-and-weather-data.
The NICO dataset is too large, the reader can find it  from the website  https://github.com/shaoshitong/NICO?tab=readme-ov-file 


Usage
All programs with the suffix .m can be directly executed in MATLAB.  The Supp_NICO.ipynb are provided for the simulation in Section 6.3 , it can be directly executed in juypter. All programs with the suffix .py are the functions in Supp_NICO.ipynb. Please refer to the function comments for specific usage instructions of each program.

ToyExample.m  Function to generate the toy example figure of Figure 1. 
GSL.m  Function for calculating the global stability loss of the dataset x.
normalization.m  Function for normalizing data x by column to any interval [ymin, ymax] range.
parfor_progress.m  Function for parallel computing to shorten USSP computation time.
rand_corr_matrix.m    Function for generating random correlation coefficient matrix of multivariate normal distribution.
rsquare.m      Function for Compututing coefficient of determination of data fit model and RMSE in all simulations.

Sample_Generate_GP_LinearlySeparableTerm.m   Function for Generating normal distribution train samples of experiment 5.2 "Stable predictions in gaussian process regression tasks with changing environments--Linearly Separable Data Shift".
Sample_Generate_GP_LinearlySeparableTerm_Test.m   Function for Generating uniform distribution test samples of experiment 5.2 "Stable predictions in gaussian process regression tasks with changing environments--Linearly Separable Data Shift".
Sample_Generate_GP_NoninearlySeparableTerm.m   Function for Generating normal distribution train samples of experiment 3.2 "Stable predictions in gaussian process regression tasks with changing environments--Nonlinearly Separable Data Shift".
Sample_Generate_GP_NoninearlySeparableTerm_Test.m   Function for Generating uniform distribution test samples of experiment 3.2 "Stable predictions in gaussian process regression tasks with changing environments--Nonlinearly Separable Data Shift".
Sample_Generate_GP_OneRegion.m   Function for Generating normal distribution train samples of experiment 3.1 in supplement materials "Stable prediction for gaussian process regression in consistent and inconsistent environments".
Sample_Generate_GP_OneRegion_Test.m  Function for Generating uniform distribution test samples of experiment 3.2 in supplement materials "Stable prediction for gaussian process regression in consistent and inconsistent environments".
Sample_Generate_OLS_8dimension.m  Function for Generating normal distribution train samples of experiment 5.1 "Stable predictions in linear regression tasks with changing environments".
Sample_Generate_OLS_8dimension_Test.m  Function for Generating uniform distribution test samples of experiment 5.1 "Stable predictions in linear regression tasks with changing environments".
USSP.m   Function for the uniform-subsampled stable prediction (USSP) algorithm in matlab.
IBOSS.m   Function for the IBOSS algorithm in matlab.
LowCon.m  Function for the LowCon algorithm in matlab.

USSP_GP_LinearlySeparableTerm.m  Main function of experiment 5.2 "Stable predictions in gaussian process regression tasks with changing environments--Linearly Separable covariate Shift".
USSP_GP_NonlinearlySeparableTerm.m  Main function of experiment 3.1 in supplement materials "Stable predictions in gaussian process regression tasks with changing environments--Nonlinearly Separable Data Shift".
USSP_GP_OneRegion.m   Main function of experiment 3.2 in supplement materials "Supplementary results of stable prediction for Gaussian process regression in consistent and
inconsistent environments".
USSP_OLS_DifferentRtrain.m   Main function of experiment 5.1 "Stable predictions in linear regression tasks with changing environments".
USSP_Realdata_Air_GP.m  Main function of experiment 6.2 GP model in "The urban air quality dataset of China".
USSP_Realdata_US10_OLS.m  Main function of experiment 6.1 OLS model in "Top 10 cities weather dataset of the United States".
USSP_GP_Covariate_Concept_Shift.m    Main function of experiment 5.3 "Stable predictions in gaussian process regression tasks with covariate shfit and concept shift".


Supp_NICO.ipynb  Main function of Supplementary materials section 4, the ResNet-50 CNN in NICO while B take different values.
Supp_NICO_new.ipynb  Main function of latest simluation of Section 5.3, the Standard, DRO, IRM ResNet-50 CNN in NICO.
USSP.py   Function for the uniform-subsampled stable prediction (USSP) algorithm in python.
IBOSS.py   Function for the IBOSS algorithm in python.
LowCon.py  Function for the LowCon algorithm in python.
NICO_ResNet50_TrainTest.py   Function for the NesNet-50  training and test in python.
NICO_transforms.py    Function to generate the ehanced trian dataset of NICO-autumn.
NICO_ResNet50_DRO_new.py   Function for the DRO NesNet-50 model.
NICO_ResNet50_IRM_new.py   Function for the IRM NesNet-50 model.
NICO_ResNet50_ERM_new.py   Function for the Standard NesNet-50 model.
NICO_ResNet50_extract.py   Function for the feature extraction NesNet-50 model.


July 10, 2025
