import pandas as pd  
import numpy as np 

def concat_names(first, last):
    return "{} {}".format(first, last)

def fix_names(df):
    df["Competitor"] = np.nan
    df["LastName"] = df["LastName"].str.lower()
    df["FirstName"] = df["FirstName"].str.lower()
    df["Competitor"] = df.apply(lambda x: concat_names(x["FirstName"], x["LastName"]), axis=1)
    return df

def set_competition_name(date, comp, loc):
    ''' 
    take the date, competition, and location and concatenate together to get an overall 
    competition name value - drop the extra values from the data frame
    '''
    # new_date <tuple> = (month, year)
    new_date = (date.split(" ")[-2], date.split(" ")[-1])
    new_loc = loc.split(",")[-1].strip()

    return "{}-{} {} {}".format(new_date[0], new_date[1], comp, new_loc)

def fix_competitions(df):
    df["Format Competition"] = df.apply(lambda x: set_competition_name(x["Date"], x["Competition"], x["Location"]), axis=1)
    return df

def merge_data():
    df1 = "data_2017_2021_1"
    df2 = "data_2022_2023_1"

    # read in both data frames from the processed data 
    comp1 = pd.read_csv("../processed_data/{}.csv".format(df1))
    comp2 = pd.read_csv("../processed_data/{}.csv".format(df2))

    comp3 = pd.concat([comp1, comp2])
    comp3.to_csv("../processed_data/all_data.csv", index=False)

def main():
    data_file = "data_2017_2021"
    file_name = "../../../cleandata/{}.csv".format(data_file)
    df = pd.read_csv(file_name)
    df = fix_competitions(df)
    df = fix_names(df)
    print(df["Date"].unique())
    df.to_csv("../processed_data/{}_1.csv".format(data_file), index=False)

if __name__ == "__main__":
    merge_data()