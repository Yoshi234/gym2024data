'''
python source file for ordering the entries of each competitor by date
'''

import pandas as pd
import numpy as np

def order_dates():
    '''
    take as input the all_data dataframe
    - convert the given dates into dates which are actually usable for our purposes, i.e. YYYY-MM-DD format
    - once dates are converted - each of these can be converted to datetime objects
    - datetime objects can be associated with each of the samples with their attached indices
    '''
    data_file = "all_data"
    file_name = "processed_data/{}.csv".format(data_file)
    
    # read data into scope
    df = pd.read_csv(file_name)

    '''
    pseudo code for function: 

    split data frame in three parts - elements where the 
    '''

