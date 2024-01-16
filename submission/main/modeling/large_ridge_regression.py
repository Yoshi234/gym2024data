import pandas as pd 
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_selection import SelectFromModel
from training_preprocess.preprocess import preprocess_train


def save_model(location, model, mod_name, input_features):
    filename = "{}/{}.pkl".format(location, mod_name)
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    with open("{}/{}_features.pkl".format(location, mod_name), "wb") as f2:
        pickle.dump(input_features, f2)

def train_linear_mod(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    linear = LinearRegression().fit(x_train, y_train)
    model_r2 = linear.score(x_test, y_test)
    return model_r2, linear

def train_ridge_mod(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(x_train, y_train)
    # importance = np.abs(ridge.coef_)

    # # what does the `SelectFromModel` function do?
    # sfm = SelectFromModel(ridge, threshold='median')
    # sfm.fit(x_train, y_train)
    # # x_train_selected = sfm.transform(x_train)
    # # x_test_selected = sfm.transform(x_test)

    # selected_feature_names = x.columns[sfm.get_support()].values
    model_r2 = ridge.score(x_test, y_test)

    # return the model score and the model itself
    return model_r2, ridge

def mini_score_linear_model():
    data_file = "../processed_data/all_data.csv"
    dat = pd.read_csv(data_file)

    x1, _, _, _, y1, y2, y3, y4, y5 = preprocess_train(dat)
    score, r_mod = train_linear_mod(x1, y4)
    print(score)

def mini_score_ridge_model():
    data_file = "../processed_data/all_data.csv"
    dat = pd.read_csv(data_file)

    # d_score prediction -- 0.63
    # e_score prediction -- 0.48
    # penalty prediction -- 0.01
    # rank    prediction -- 0.65 (Lucy = 0.74)
    x1, x2, _, _, y1, y2, y3, y4, y5 = preprocess_train(dat)
    # print(x1.columns)
    score, r_mod = train_ridge_mod(x1, y4)
    print(score)
    save_model("pretrained", r_mod, "r6_mod", input_features=x1.columns)

def mini_rank_ridge_model():
    data_file = "../processed_data/all_data.csv"
    dat = pd.read_csv(data_file)

    x1, _, _, _, _, _, _, y4, _ = preprocess_train(dat)
    score, r_mod = train_ridge_mod(x1, y4)
    print(score)

def large_ridge_models():
    data_file = "../processed_data/all_data.csv"
    dat = pd.read_csv(data_file)
    x1, x2, x3, x4, y1, y2, y3, y4, y5 = preprocess_train(dat)

    score_r1, r1_mod = train_ridge_mod(x1, y1)
    print("Model 1: predict D_Score => {}".format(score_r1))
    # print("Features: {}".format(x1.columns))
    score_r2, r2_mod = train_ridge_mod(x2, y2)
    print("Model 2: predict E_Score => {}".format(score_r2))
    # print("Features: {}".format(x2.columns))
    score_r3, r3_mod = train_ridge_mod(x3, y3)
    print("Model 3: predict penalty => {}".format(score_r3))
    # print("Features: {}".format(x3.columns))
    score_r4, r4_mod = train_ridge_mod(x4, y4)
    print("Model 4: predict rank    => {}".format(score_r4))
    # print("Features: {}".format(x4.columns))
    score_r5, r5_mod = train_ridge_mod(x4, y5)
    print("Model 5: predict score   => {}".format(score_r5))

    folder_path = "pretrained/"
    print("\nSaving models to '{}'".format(folder_path))
    save_model(folder_path, r1_mod, "r1_mod", input_features=x1.columns)
    save_model(folder_path, r2_mod, "r2_mod", input_features=x2.columns)
    save_model(folder_path, r3_mod, "r3_mod", input_features=x3.columns)
    save_model(folder_path, r4_mod, "r4_mod", input_features=x4.columns)
    save_model(folder_path, r5_mod, "r5_mod", input_features=x4.columns)

if __name__ == "__main__":
    large_ridge_models()
    mini_score_ridge_model()
