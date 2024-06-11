import numpy as np
import pandas as pd
from epiweeks import Week, Year
from datetime import datetime

def reporting_data(matrix, idx, past_units = 7, max_delay = 6, future_units = None): # future units for future to correct positions
    """ Function for returning reporting data
    
    Easiest with df and index, then just take past units and mask correctly"""
    matrix = matrix[(idx-past_units+1):(idx+1), :]
    y = matrix.sum(axis = 1)[-1] # use different ones here for future values
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

    def __init__(self, df, max_val, triangle = False, past_units=6, max_delay=6, device = "mps"):
        """
        Initialize the dataset with a start and end date.
        The dataset will generate matrices for each date within this range.
        
        Parameters:
        - start_date: The start date for generating matrices.
        - end_date: The end date for generating matrices.
        - past_days: Number of past days to consider for each matrix.
        - max_delay: Maximum delay to consider for each matrix.
        """
        self.df = df
        self.past_units = past_units
        self.max_delay = max_delay
        self.device = device
        self.triangle = triangle
        self.max_val = max_val

    def get_length(self):
        return self.df.shape[0]

    def __len__(self):
        # Calculate the number of days between 60 days after start_date and 46 days before end_date
        return len(self.df) - (self.past_units-1) - self.max_delay
    
    def __getitem__(self, idx):
        # Calculate the date for the current iteration, considering the adjusted range
        idx += self.past_units-1
        assert idx < len(self.df), "Index out of range"
        
        # Generate the matrix for the current date
        matrix, label = reporting_data(self.df, idx=idx, past_units=self.past_units, max_delay=self.max_delay)
        
        # Convert the matrix to a PyTorch tensor
        tensor = torch.from_numpy(matrix)
        tensor = tensor.to(device=self.device)

        if not self.triangle:
            # Sum
            tensor = torch.sum(tensor, dim = 1)
        
        # Compute the sum of the delays for the current date (row sum)
        label = torch.tensor(label).to(self.device)
        return tensor/self.max_val, label/1000 # könnte label lassen, nur 1000xrate in Poisson

def get_dataset(weeks = True, triangle = False, past_units = 6, max_delay = 6, state = "SP"):
    """ Have to return the iterable dataset, so first read in csv file, then convert to delay-format
    Then feed to iterable dataset and return that
    
    Args:
    
    Returns:
    """

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

    # Rename columns to reflect delays, internal checks
    dengdf.columns = [f'delay_{col}' for col in dengdf.columns]

    if weeks:
        max_val = dengdf[dengdf.columns[1:]].sum(axis = 1).max()
    else:
        max_val = dengdf[dengdf.columns[1:]].max(axis = 1).max(axis=0)
    dengdf.reset_index(inplace=True)
    dengdf.fillna(0, inplace=True)

    dengdf = np.array(dengdf.iloc[:, 1:].values, dtype = np.float32)
    
    ## Define dataset
    return ReportingDataset(dengdf, max_val=max_val, triangle=triangle, past_units=past_units, max_delay=max_delay)
    


