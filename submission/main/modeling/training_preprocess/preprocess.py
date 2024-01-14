'''
Features for preprocessing

IndividualDateRank --> numeric (exponential decreasing) -> fit log transform --> impute missing values with 0
Score --> numeric --> fit standard scaler --> impute missing values with 0
Penalty --> numeric --> fit standard scaler --> impute missing values with 0
E_Score --> numeric --> fit standard scaler --> impute missing values with 0 
D_Score --> numeric --> fit standard scaler --> impute missing values with 0
Rank --> numeric --> fit log transform --> impute missing values with max
Apparatus --> category --> one-hot encoder --> impute with "not available"
Round --> category --> one-hot encoder --> "not available"
Country --> category --> one-hot encoder --> "not available"
Competitor --> category --> one-hot encoder --> "not available"
'''

import math
import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline

def Log(x):
    return math.log10(x)

def invLog(x):
    return 10**x

LogTransformer = FunctionTransformer(np.log)

def preprocess_help(data):
    '''
    helper function for data preprocessing
    '''
    max_log_num = data["IndividualDateRank"].max()

    log_numerical_transformer_0 = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value=0)), 
        ('scaler', LogTransformer)
    ])
    log_numerical_transformer_max = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value=max_log_num)), 
        ('scaler', LogTransformer)
    ])
    norm_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value=0)), 
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value="not available")), 
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    log_0_cols = ["IndividualDateRank"]
    log_max_cols = ["Rank"]
    norm_numeric_cols = ["Score", "Penalty", "E_Score", "D_Score"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('log0', log_numerical_transformer_0, log_0_cols), 
            ('logmax', log_numerical_transformer_max, log_max_cols), 
            ('norm_numeric', norm_numerical_transformer, norm_numeric_cols), 
            ('category', categorical_transformer, categorical_columns)
        ], remainder='passthrough'
    )

    return preprocessor

def preprocess_train(data: pd.DataFrame):
    '''
    calls the preprocess help function to get the preprocessed features
    NOTE - when getting actual values, make sure to unscale the response data 
    according to the correct parameters - maybe it would be a better idea to perform
    those transformations separately?

    The mean and standard deviation for this data can be obtained from the original processed data - 
    Simply undue the scaling transformation to obtain actual values

    Feeding in the D_Score, E_Score, Penalty, Rank, and for scaling 
    will just make it so that the 

    Returns a tuple of 8 datasets - the datasets needed for training 4 separate models
    - x1 --- excludes D_Score, E_Score, Penalty, and Rank - used to predict D_Score
    - x2 --- excludes E_Score, Penalty, and Rank - used to predict E_Score
    - x3 --- excludes Penalty and Rank - used to predict Penalty
    - x4 --- excludes Rank - used to predict Rank
    - y1 --- D_Score labels
    - y2 --- E_Score labels
    - y3 --- Penalty labels
    - y4 --- Rank labels
    '''
    full_dat = data.drop(["Format Competition", "DateTime", "Location", "LastName", "FirstName", "Date", "DateRank"], axis=1)
    preprocessor = preprocess_help(full_dat)
    pipe1 = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])
    pipe1.set_output(transform="pandas")
    proc_dat = pipe1.fit_transform(full_dat)

    # debug statement --- print(proc_dat.columns)
    # training data for fitting model to difficulty score
    X1 = proc_dat.drop(["norm_numeric__D_Score", "norm_numeric__E_Score", "norm_numeric__Penalty", "logmax__Rank"], axis=1)
    # training data for fitting model to estimate execution score
    X2 = proc_dat.drop(["norm_numeric__E_Score", "norm_numeric__Penalty", "logmax__Rank"], axis=1)
    # training data for fitting model to predict penalty
    X3 = proc_dat.drop(["norm_numeric__Penalty", "logmax__Rank"], axis=1)
    # training data for fitting model to predict rank
    X4 = proc_dat.drop(["logmax__Rank"], axis=1)

    Y1 = proc_dat["norm_numeric__D_Score"]
    Y2 = proc_dat["norm_numeric__E_Score"]
    Y3 = proc_dat["norm_numeric__Penalty"] 
    Y4 = proc_dat["logmax__Rank"]

    return X1, X2, X3, X4, Y1, Y2, Y3, Y4

def preprocess_predict(data: pd.DataFrame):
    '''
    Args:
    - data --- takes as input a pandas dataframe 
    and outputs the preprocessed dataframes of interest for 
    model validation / prediction steps

    Returns: 
    - x, y --- preprocessed data frames for model validation

    Functionality notes: to get the originally scaled values for 
    the D_Score, E_Score, Penalty, and Rank, use the following transformations:
    - D_Score --- get mu, std -> apply (x*std + mu)
    - E_Score --- get mu, std -> apply (x*std + mu)
    - Penalty --- get mu, std -> apply (x*std + mu)
    - Rank    ---             -> apply (10**x)
    '''
    full_dat = data.drop(["Format Competition", "DateTime", "Location", "LastName", "FirstName", "Date", "DateRank"], axis=1)
    preprocessor = preprocess_help(full_dat)
    pipe1 = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])
    pipe1.set_output(transform="pandas")
    proc_dat = pipe1.fit_transform(full_dat)

    x = proc_dat.drop(["norm_numeric__D_Score", "norm_numeric__E_Score", "norm_numeric__Penalty", "logmax__Rank"], axis=1)
    y = proc_dat["logmax__Rank"]

    return x, y

if __name__ == "__main__":
    data_file = "all_data"
    file_name = "../../processed_data/{}.csv".format(data_file)
    comp_data = pd.read_csv(file_name)
    X1, X2, X3, X4, Y1, Y2, Y3, Y4 = preprocess_train(comp_data)