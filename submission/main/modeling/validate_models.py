import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from training_preprocess.preprocess import preprocess_predict

def load_model(folder, mod_name):
    filename = "{}/{}.pkl".format(folder, mod_name)
    print("reading from {}".format(filename))
    with open(filename, "rb") as f:
        model = pickle.load(f)
    with open("{}/{}_features.pkl".format(folder, mod_name), "rb") as f2:
        features = pickle.load(f2)
    return model, features

def validate_ridge_models(data):
    folder = "pretrained"
    # get preprocessed data - x is initial data, y is final result
    x1, y4 = preprocess_predict(data)
    # load all models
    r1_mod, r1_features = load_model(folder, "r1_mod")
    r2_mod, r2_features = load_model(folder, "r2_mod")
    r3_mod, r3_features = load_model(folder, "r3_mod")
    r4_mod, r4_features = load_model(folder, "r4_mod")
    # pass models through pipeline
    y1 = pd.DataFrame(r1_mod.predict(x1), columns=["norm_numeric__D_Score"])
    
    x2 = pd.concat([x1, y1], axis=1)
    x2 = x2[r2_features]
    y2 = pd.DataFrame(r2_mod.predict(x2), columns=["norm_numeric__E_Score"])
    
    x3 = pd.concat([x2, y2], axis=1)
    x3 = x3[r3_features]
    y3 = pd.DataFrame(r3_mod.predict(x3), columns=["norm_numeric__Penalty"])
    
    x4 = pd.concat([x3, y3], axis=1)
    x4 = x4[r4_features]
    final_score = r4_mod.score(x4, y4)
    print(final_score)

def validate_model():
    data_file = "../processed_data/all_data.csv"
    dat = pd.read_csv(data_file)
    validate_ridge_models(dat)

if __name__ == "__main__":
    validate_model()