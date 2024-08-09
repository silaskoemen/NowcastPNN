import numpy as np
import pandas as pd
from epiweeks import Week, Year
from datetime import datetime

def reporting_data(matrix: np.ndarray, idx: int, past_units: int = 7, max_delay: int = 6, future_obs: int = 0): # future units for future to correct positions
    """ Function for returning reporting data
    
    Easiest with df and index, then just take past units and mask correctly"""
    assert future_obs < past_units, "Number of future observed units should be smaller than the number of past units included (otherwise exceeds the matrix)"
    assert future_obs >= 0, "Number of days of additional observations needs to be non-negative"
    matrix = matrix[(idx-past_units+1):(idx+1), :].copy() # otherwise modifies inplace, changes dataset
    y = matrix.sum(axis = 1)[-(1+future_obs)]
    y_otd = matrix[-1, 0]
    # Could add y_atm as matrix[-1, 0] and return for plotting
    mask = np.zeros((past_units, max_delay), dtype=bool)
    for p in range(max_delay):
        for d in range(max_delay):
            if p + d >= max_delay:
                mask[p+(past_units-max_delay), d] = True
    matrix[mask] = 0.
    return matrix, y

import torch
from torch.utils.data import Dataset

class ReportingDataset(Dataset):
    ## Theoretically, should contain covariates for date too, return tuple of matrix and covariates as well as label at each iteration

    def __init__(self, df, max_val, triangle = False, past_units=6, max_delay=6, future_obs = 0, device = "mps"):
        """
        Initialize the dataset with a start and end date.
        The dataset will generate matrices for each date within this range.
        
        Parameters:
        - start_date: The start date for generating matrices.
        - end_date: The end date for generating matrices.
        - past_days: Number of past days to consider for each matrix.
        - max_delay: Maximum delay to consider for each matrix.
        """
        if isinstance(df, pd.DataFrame):
            self.df = np.array(df, dtype = np.float32)
        else:
            self.df = df
        self.past_units = past_units
        self.max_delay = max_delay
        self.device = device
        self.triangle = triangle
        self.max_val = max_val
        self.future_obs = future_obs

    def get_length(self):
        return self.df.shape[0]

    def __len__(self):
        # Calculate the number of days between 60 days after start_date and 46 days before end_date
        return len(self.df) - (self.past_units-1) - (self.max_delay-1)
    
    def __getitem__(self, idx):
        # Calculate the date for the current iteration, considering the adjusted range
        idx += self.past_units-1
        assert idx < len(self.df), "Index out of range"
        
        # Generate the matrix for the current date
        matrix, label = reporting_data(self.df, idx=idx, past_units=self.past_units, max_delay=self.max_delay, future_obs=self.future_obs)
        
        # Convert the matrix to a PyTorch tensor
        tensor = torch.from_numpy(matrix)
        tensor = tensor.to(device=self.device)

        if not self.triangle: # sum
            tensor = torch.sum(tensor, dim = 1)
        
        # Compute the sum of the delays for the current date (row sum)
        label = torch.tensor(label).to(self.device)
        return tensor/self.max_val, label
        #return tensor, label

def get_dataset(weeks = True, triangle = False, past_units = 6, max_delay = 6, state = "SP", future_obs = 0, return_df = False, return_mat = False):
    """ Have to return the iterable dataset, so first read in csv file, then convert to delay-format
    Then feed to iterable dataset and return that
    
    Args:
    
    Returns:
    """
    assert not (return_df and return_mat), "Only either dataframe or matrix can be returned"
    dengdf = pd.read_csv(f"../data/derived/DENG{state}.csv", index_col=0)
    date_format = "%Y-%m-%d"
    dengdf['DT_NOTIFIC'] = pd.to_datetime(dengdf['DT_NOTIFIC'], format=date_format)
    dengdf['DT_SIN_PRI'] = pd.to_datetime(dengdf['DT_SIN_PRI'], format=date_format)

    ## Filter to only take rows with onset after 2012 (2013-1, others assumed to be faulty data)
    dengdf = dengdf.loc[((dengdf["DT_SIN_PRI"].dt.year > 2012) & (dengdf["DT_SIN_PRI"].dt.year < 2021) & (dengdf["DT_NOTIFIC"].dt.year > 2012) & (dengdf["DT_NOTIFIC"].dt.year < 2021))]

    if weeks:
        dengdf = dengdf.dropna(subset=['DT_SIN_PRI', 'DT_NOTIFIC', "SEM_NOT", "SEM_PRI"])
        ## Know minimum year is 2013 and maximum 2020, so can discard faulty observations
        ## CHANGE if data before 2013 or after 2020 is added
        
        # Week.week returns as int, können also einfach mit b-a (falls über Jahr einfach mit 52)
        dengdf["WK_SIN_PRI"] = dengdf["DT_SIN_PRI"].apply(lambda x: Week.fromdate(x))
        dengdf["WK_NOTIFIC"] = dengdf["DT_NOTIFIC"].apply(lambda x: Week.fromdate(x))
        ## Convert wk sin pri and notific to cdcformat, can join with other df from fct
        dengdf["WK_SIN_PRI_INT"] = dengdf["WK_SIN_PRI"].apply(lambda x: x.week)
        dengdf["WK_NOTIFIC_INT"] = dengdf["WK_NOTIFIC"].apply(lambda x: x.week)
        dengdf['delay'] = (dengdf['WK_NOTIFIC_INT'] - dengdf['WK_SIN_PRI_INT'])
        dengdf["delay"] = dengdf["delay"].apply(lambda x: x if x >= 0 else x + 52) # if notified across year boundary (negative value), correct
        ## If do not want as triangle, could look at higher max_delay values bc summed either way
        dengdf = dengdf.groupby('WK_SIN_PRI')['delay'].apply(lambda x: x.value_counts().reindex(range(max_delay))).unstack(fill_value=0)
    else:
        dengdf = dengdf.dropna(subset=['DT_SIN_PRI', 'DT_NOTIFIC'])
        dengdf['delay'] = (dengdf['DT_NOTIFIC'] - dengdf['DT_SIN_PRI']).dt.days
        dengdf = dengdf.groupby('DT_SIN_PRI')['delay'].apply(lambda x: x.value_counts().reindex(range(max_delay))).unstack(fill_value=0)
        dengdf["DT_SIN_PRI"] = pd.to_datetime(dengdf.index)
        dengdf.index = range(len(dengdf))
        start_date = dengdf['DT_SIN_PRI'].min()
        end_date = dengdf['DT_SIN_PRI'].max()

        # Create a new DataFrame with a continuous range of dates
        date_range_df = pd.DataFrame({'DT_SIN_PRI': pd.date_range(start=start_date, end=end_date)})
        print(len(date_range_df), start_date, end_date)

        # Merge the original DataFrame with the date range DataFrame, filling missing values with 0
        dengdf = pd.merge(date_range_df, dengdf, how='left', on='DT_SIN_PRI')

        # Replace NaN values in numeric columns with 0
        for col in dengdf.columns.drop('DT_SIN_PRI'):
            dengdf[col] = dengdf[col].fillna(0)
        
        dengdf.drop("DT_SIN_PRI", axis = 1, inplace = True)

    # Rename columns to reflect delays, internal checks
    dengdf.columns = [f'delay_{col}' for col in dengdf.columns]

    if weeks:
        max_val = dengdf[dengdf.columns[1:]].sum(axis = 1).max()
    else:
        max_val = dengdf.max(axis = 1).max(axis=0)

    #dengdf.reset_index(inplace=True)
    dengdf.fillna(0, inplace=True)

    if return_df:
        return dengdf
    elif return_mat:
        return np.array(dengdf.values, dtype = np.float32)
    dengdf = np.array(dengdf.values, dtype = np.float32) # wahrscheinlich falsch wenn DT/WK nicht drin ist
    
    ## Define dataset
    return ReportingDataset(dengdf, max_val=max_val, triangle=triangle, past_units=past_units, max_delay=max_delay, future_obs=future_obs)
    


""" Could use to find units of maximum value, return with dataset and then parse to NN as self.const
counter = len(str(max_number))
    
# Calculate the nearest unit of length based on the counter
nearest_unit = 10 ** (counter - 1)
"""