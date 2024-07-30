#
# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.
 
## To create environment
conda env create -n mle-dev --file env.yml

## To activate environment
conda activate mle-dev

## To excute the script
python3 nonstandard.py

## MLflow UI 
## Experiment Page
![image](https://github.com/user-attachments/assets/108ada4c-e4c4-4dd0-b513-64269c1d5548)
## Parameters and artifacts stored for each file are shown below
# ingest_data.py
The processed data has been stored in the artifacts folder
![image](https://github.com/user-attachments/assets/8defbe60-0a10-4003-8d7e-e666511e3d63)'

# train.py
The trained models are stored 
![image](https://github.com/user-attachments/assets/8f717646-f396-40a6-bbb7-247bf99bfbff)

# score.py
The score for each model is stored
![image](https://github.com/user-attachments/assets/562c019d-325c-4543-974a-462eb01d28ab)

