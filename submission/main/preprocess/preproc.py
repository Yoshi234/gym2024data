import pandas as pd  
import numpy as np 
from names_cleaning import fixes
from datetime import datetime

def concat_names(first, last):
    return "{} {}".format(first, last)

def fix_names(df):
    df["Competitor"] = np.nan
    df["LastName"] = df["LastName"].str.lower()
    df["FirstName"] = df["FirstName"].str.lower()
    df["Competitor"] = df.apply(lambda x: concat_names(x["FirstName"], x["LastName"]), axis=1)
    return df

class ranked_date: 
    def __init__(self, og_date, date_val:datetime):
        self.og_date = og_date
        self.date_val = date_val
        self.rank = None

    def __str__(self):
        return f"{self.datetime} - rank => {self.rank}"

def rank_dates(comp_data):
    raw_dates = list(comp_data["DateTime"].unique())
    dates_ranked = [ranked_date(date, datetime.strptime(date, "%Y-%m-%d")) for date in raw_dates]
    dates_ranked = sorted(dates_ranked, key=lambda rank_date: rank_date.date_val)
    for i in range(len(dates_ranked)): dates_ranked[i].rank = i

    comp_data["DateRank"] = np.nan
    for i in dates_ranked:
        comp_data.loc[comp_data["DateTime"] == i.og_date, "DateRank"] = i.rank
    return comp_data

class comp_date:
    def __init__(self, date_rank, name):
        self.name = name
        self.date_rank = date_rank
        self.new_rank = None

def indiv_rank_dates(comp_data):
    for competitor in list(comp_data["Competitor"].unique()):
        small_df = comp_data.loc[comp_data["Competitor"] == competitor]
        date_ranks = list(small_df["DateRank"].unique())
        date_ranks = [comp_date(date_rank, competitor) for date_rank in date_ranks]
        dates_ranked = sorted(date_ranks, key=lambda comp_date_rank: comp_date_rank.date_rank)
        for i in range(len(dates_ranked)): 
            dates_ranked[i].new_rank = i
            comp_data.loc[(comp_data["Competitor"] == competitor) & (comp_data["DateRank"] == dates_ranked[i].date_rank), "IndividualDateRank"] = dates_ranked[i].new_rank
    return comp_data

class unique_date:
    def __init__(self, date, split_date):
        self.date = date
        self.split_date = split_date
        self.datetime = None
        self.vals = len(split_date)
    def __str__(self):
        return str(self.datetime)

def fix_dates(comp_data):
    # lowercase and fix a specific date
    comp_data["Date"] = comp_data["Date"].str.lower()
    comp_data.loc[comp_data["Date"] == "29 jul-2 aug 2022", "Date"] = "29 jul 2022 - 2 aug 2022"

    # initialize empty list of unique dates and add all of them to the unique dates list
    unique_dates = []
    for date in list(comp_data["Date"].unique()):
        x = date.split(" ")
        for i in range(len(x)): x[i] = x[i].strip()
        unique_dates.append(unique_date(date=date, split_date=x))
    
    # set the date_vals for replacement
    date_vals = {
        ("jan"): "01",
        ("feb"): "02",
        ("mar"): "03",
        ("apr"): "04",
        ("may"): "05",
        ("june", "jun"): "06",
        ("jul", "july"): "07",
        ("aug"): "08",
        ("sep", "sept"): "09",
        ("oct"): "10",
        ("nov"): "11",
        ("dec"): "12",    
    }

    # initialize new column for the datetime values
    comp_data["DateTime"] = np.nan
    # set the datetime values to use
    for i in range(len(unique_dates)):
        mm = None
        dd = None
        YYYY = None
        YYYY = unique_dates[i].split_date[-1]
        for date_set in date_vals:
                if unique_dates[i].split_date[-2] in date_set:
                    mm = date_vals[date_set]
                    continue
        if not unique_dates[i].vals in {4,7}:
            x = unique_dates[i].split_date[0].split("-")
            if len(x) == 1: dd = x[0]
            else: dd = x[-1]
        else:
            dd = unique_dates[i].split_date[-3]
        unique_dates[i].datetime = "{}-{}-{}".format(YYYY, mm, dd)
    
    # add the datetime values into the dataframe
    for i in unique_dates:
        comp_data.loc[comp_data["Date"] == i.date, "DateTime"] = i.datetime

    return comp_data

def set_competition_name(datetime, comp, loc):
    ''' 
    take the date, competition, and location and concatenate together to get an overall 
    competition name value - drop the extra values from the data frame
    '''
    # new_date <tuple> = (month, year)
    new_date = datetime[:4]
    new_loc = loc.split(",")[-1].strip()

    return "{} - {} {}".format(new_date, comp, new_loc)

def fix_competitions(df):
    df["Format Competition"] = df.apply(lambda x: set_competition_name(x["DateTime"], x["Competition"], x["Location"]), axis=1)
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
    data_file_1 = "data_2017_2021"
    data_file_2 = "data_2022_2023"
    file_1 = "../../../cleandata/{}.csv".format(data_file_1)
    file_2 = "../../../cleandata/{}.csv".format(data_file_2)
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)
    
    # fix the dates (lowercase and set datetimes)
    df1 = fix_dates(df1)
    df2 = fix_dates(df2)

    # after setting different datetimes - make all of the date times the same for tokyo
    df1 = fix_tokyo_dates(df1)

    # fix the competitions
    df1 = fix_competitions(df1)
    df2 = fix_competitions(df2)

    # fix the names
    df1 = fix_names(df1)
    df2 = fix_names(df2)

    # rank the dates
    df1 = rank_dates(df1)
    df2 = rank_dates(df2)

    # save the data
    df1.to_csv("../processed_data/{}_1.csv".format(data_file_1), index=False)
    df2.to_csv("../processed_data/{}_1.csv".format(data_file_2), index=False)
    
    # merge the datasets together
    merge_data()
    # fix duplicate naming issues
    df3 = fix_dupl_names() 
    df3 = indiv_rank_dates(df3)

    df3.to_csv("../processed_data/all_data.csv", index=False)


def fix_tokyo_dates(df):
    # only use on the japan olympics data set
    df["Format Competition"] = "jul-2021 Olympic Games Japan"
    df = fix_names(df)
    return df

def fix_dupl_names():
    data_file = "all_data"
    file_name = "../processed_data/{}.csv".format(data_file)
    df = pd.read_csv(file_name)
    
    for name_set in fixes:
        for unique_name in list(df["Competitor"].unique()):
            if unique_name in name_set: df.loc[df["Competitor"] == unique_name, "Competitor"] = fixes[name_set]
    df.to_csv("../processed_data/{}.csv".format(data_file), index=False)
    return df

if __name__ == "__main__":
    # print("Please set the function desired to run in the file")
    
    main()

    # steps taken to process the data: 
    # (0) run the fix tokyo dates function to set all of the dates the same
    # (1) run the main function for both of the data sets to get "processed" versions of them
    # (2) run the merge function to merge the "processed" data sets into a single data set
    # (3) run the fix_dupl_names() function to fix issues with duplicate names that are just slight variations of one another.