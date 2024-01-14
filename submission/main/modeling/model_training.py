import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from time import time
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv ("../processed_data/all_data.csv")

    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(df[['Gender', 'Country','DateTime',
        'Round', 'Location', 'Apparatus', 'Format Competition', 'Competitor']])
    feature_names = encoder.get_feature_names_out()
    features = pd.DataFrame(one_hot.toarray(), columns=feature_names)
    print(features.head())
    df_new = df.join(features) 
    print(df_new.head())

    data= df_new.drop(['Rank','LastName', 'FirstName', 'Gender', 'Country', 'Date', 'DateTime','Competition',
        'Round', 'Location', 'Apparatus', 'Format Competition', 'Competitor', 'D_Score','E_Score','Penalty'], axis=1).dropna()


    print(data)
    X=data.drop(["Score"], axis=1)
    y=data["Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
    )

    # model.fit(X_train, ytrain)

    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
    importance = np.abs(ridge.coef_)
    # plt.bar(height=importance, x=feature_names)
    # plt.title("Feature importances via coefficients")
    # plt.show()

    sfs_forward = SequentialFeatureSelector(
        ridge, n_features_to_select=12, direction="forward"
    ).fit(X, y)

    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )

if __name__ == "__main__":
    main()