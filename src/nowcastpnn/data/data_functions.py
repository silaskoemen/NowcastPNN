import numpy as np
import pandas as pd
from epiweeks import Week, Year
from datetime import datetime

from sklearn.model_selection import train_test_split as TTS
from torch.utils.data.sampler import SubsetRandomSampler as SRS
from torch.utils.data import Dataset, DataLoader

from nowcastpnn.utils.plotting import days_to_date
from nowcastpnn.utils import SubsetSampler as SS


def reporting_data(matrix: np.ndarray, idx: int, past_units: int = 40, max_delay: int = 40, future_obs: int = 0, vector_y = False, dow = False): # future units for future to correct positions
    """ Function for returning reporting data

    Easiest with df and index, then just take past units and mask correctly"""
    assert future_obs < past_units, "Number of future observed units should be smaller than the number of past units included (otherwise exceeds the matrix)"
    assert future_obs >= 0, "Number of days of additional observations needs to be non-negative"
    matrix = matrix[(idx-past_units+1):(idx+1), :].copy() # otherwise modifies inplace, changes dataset
    if vector_y:
        y = matrix[-(1+future_obs), :].copy()
    else:
        y = matrix.sum(axis = 1)[-(1+future_obs)].copy()
    #y_otd = matrix[-1, 0]
    mask = np.zeros((past_units, max_delay), dtype=bool)
    for p in range(max_delay):
        for d in range(max_delay):
            if p + d >= max_delay:
                if abs(p + past_units - max_delay) < past_units:
                    mask[p+(past_units-max_delay), d] = True
    matrix[mask] = 0.
    if dow:
        return matrix, days_to_date("2013-01-01", idx, past_units).weekday(), y
    return matrix, y

import torch
from torch.utils.data import Dataset

class ReportingDataset(Dataset):
    ## Theoretically, should contain covariates for date too, return tuple of matrix and covariates as well as label at each iteration

    def __init__(self, df, max_val, triangle = True, past_units=40, max_delay=40, future_obs = 0, device = "mps", vector_y = False, dow = False, return_number_obs = False):
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
        self.vector_y = vector_y
        self.dow = dow
        self.start_date = "2013-01-01"
        self.return_number_obs = return_number_obs

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
        if self.dow:
            matrix, dow_val, label = reporting_data(self.df, idx=idx, past_units=self.past_units, max_delay=self.max_delay, future_obs=self.future_obs, vector_y = self.vector_y, dow=self.dow)
            dow_val = torch.tensor(dow_val).to(self.device)
        else:
            matrix, label = reporting_data(self.df, idx=idx, past_units=self.past_units, max_delay=self.max_delay, future_obs=self.future_obs, vector_y = self.vector_y, dow=self.dow)

        # Convert the matrix to a PyTorch tensor
        tensor = torch.from_numpy(matrix)
        tensor = tensor.to(device=self.device)

        if not self.triangle: # sum
            tensor = torch.sum(tensor, dim = 1)

        # Compute the sum of the delays for the current date (row sum)
        label = torch.tensor([label]).to(self.device)
        if self.return_number_obs:
            num_obs = tensor.sum(axis = 1)[-(1+self.future_obs)].clone() # probably wrong
            label = (label, num_obs)
        if self.dow:
            return (tensor/self.max_val, dow_val), label
        return tensor/self.max_val, label
        #return tensor, label


def get_dataset(weeks = False, triangle = True, past_units = 40, max_delay = 40, future_obs = 0, return_df = False, return_mat = False, return_number_obs = False, vector_y = False, dow = False, path = "../data/derived/DENGSP.csv", reference_date_col = 'DT_SIN_PRI', report_date_col = 'DT_NOTIFIC'):
    """ Have to return the iterable dataset, so first read in csv file, then convert to delay-format
    Then feed to iterable dataset and return that

    Args:

    Returns:
    """
    assert not (return_df and return_mat), "Only either dataframe or matrix can be returned"
    ## Add reference_col and report_col so use reference_date and report_date if None and given strings otherwise
    dengdf = pd.read_csv(path, index_col=0, low_memory=False)  # DTypeWarning of mixed dtypes otherwise
    date_format = "%Y-%m-%d"
    dengdf[report_date_col] = pd.to_datetime(dengdf[report_date_col], format=date_format)
    dengdf[reference_date_col] = pd.to_datetime(dengdf[reference_date_col], format=date_format)

    ## Filter to only take rows with onset after 2012 (2013-1, others assumed to be faulty data)
    dengdf = dengdf.loc[((dengdf[reference_date_col].dt.year > 2012) & (dengdf[reference_date_col].dt.year < 2021) & (dengdf["DT_NOTIFIC"].dt.year > 2012) & (dengdf["DT_NOTIFIC"].dt.year < 2021))]

    if weeks:
        dengdf = dengdf.dropna(subset=[reference_date_col, report_date_col, "SEM_NOT", "SEM_PRI"])
        ## Know minimum year is 2013 and maximum 2020, so can discard faulty observations
        ## CHANGE if data before 2013 or after 2020 is added

        # Week.week returns as int, können also einfach mit b-a (falls über Jahr einfach mit 52)
        dengdf["WK_SIN_PRI"] = dengdf[reference_date_col].apply(lambda x: Week.fromdate(x))  # type: ignore
        dengdf["WK_NOTIFIC"] = dengdf[report_date_col].apply(lambda x: Week.fromdate(x))  # type: ignore
        ## Convert wk sin pri and notific to cdcformat, can join with other df from fct
        dengdf["WK_SIN_PRI_INT"] = dengdf["WK_SIN_PRI"].apply(lambda x: x.week)
        dengdf["WK_NOTIFIC_INT"] = dengdf["WK_NOTIFIC"].apply(lambda x: x.week)
        dengdf['delay'] = (dengdf['WK_NOTIFIC_INT'] - dengdf['WK_SIN_PRI_INT'])
        dengdf["delay"] = dengdf["delay"].apply(lambda x: x if x >= 0 else x + 52) # if notified across year boundary (negative value), correct
        ## If do not want as triangle, could look at higher max_delay values bc summed either way
        dengdf = dengdf.groupby('WK_SIN_PRI')['delay'].apply(lambda x: x.value_counts().reindex(range(max_delay))).unstack(fill_value=0)
    else:
        dengdf = dengdf.dropna(subset=[reference_date_col, report_date_col])
        dengdf['delay'] = (dengdf[report_date_col] - dengdf[reference_date_col]).dt.days
        dengdf = dengdf.groupby(reference_date_col)['delay'].apply(lambda x: x.value_counts().reindex(range(max_delay))).unstack(fill_value=0)
        dengdf[reference_date_col] = pd.to_datetime(dengdf.index)
        dengdf.index = range(len(dengdf))
        start_date = dengdf[reference_date_col].min()
        end_date = dengdf[reference_date_col].max()

        # Create a new DataFrame with a continuous range of dates
        date_range_df = pd.DataFrame({reference_date_col: pd.date_range(start=start_date, end=end_date)})
        print(len(date_range_df), start_date, end_date)

        # Merge the original DataFrame with the date range DataFrame, filling missing values with 0
        dengdf = pd.merge(date_range_df, dengdf, how='left', on=reference_date_col)

        # Replace NaN values in numeric columns with 0
        for col in dengdf.columns.drop(reference_date_col):
            dengdf[col] = dengdf[col].fillna(0)

        dengdf.drop(reference_date_col, axis = 1, inplace = True)

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
    dengdf = np.array(dengdf.values, dtype = np.float32)

    ## Define dataset
    return ReportingDataset(dengdf, max_val=max_val, triangle=triangle, past_units=past_units, max_delay=max_delay, future_obs=future_obs, vector_y = vector_y, dow = dow, return_number_obs = return_number_obs)


def get_dataset_config(cfg):
    return get_dataset(
        weeks=cfg.data.weeks,
        triangle=cfg.data.triangle,
        past_units=cfg.data.past_units,
        max_delay=cfg.data.max_delay,
        future_obs=cfg.data.future_obs,
        return_df=cfg.data.output.return_df,
        return_mat=cfg.data.output.return_mat,
        return_number_obs=cfg.data.output.return_number_obs,
        vector_y=cfg.data.vector_y,
        dow=cfg.data.use_dow,
        path=cfg.data.filepath,
        reference_date_col=cfg.data.reference_date_col,
        report_date_col=cfg.data.report_date_col,
    )


def get_loaders_from_dataset(dataset, cfg):
    """ Utility function to create DataLoader objects for training, validation, and testing.

    Args:
        dataset: The dataset to split and load.
        batch_size: Number of samples per batch.
        val_split: Proportion of the dataset to use for validation.
        test_split: Proportion of the dataset to use for testing.
        shuffle: Whether to shuffle the dataset before splitting.
        random_seed: Seed for random number generator (for reproducibility).

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    if cfg.data.random_split:
        all_idcs = range(dataset.__len__())
        train_idcs, test_idcs = TTS(all_idcs, test_size=cfg.data.test_split_size, shuffle=cfg.data.shuffle, random_state=cfg.seed)
        train_idcs, val_idcs = TTS(train_idcs, test_size=cfg.data.val_split_size, shuffle=cfg.data.shuffle, random_state=cfg.seed)
        VAL_BATCH_SIZE, TEST_BATCH_SIZE = len(val_idcs), len(test_idcs)
    else:
        """if cfg.data.weeks: # could also do random split, for now last indices as test
            train_idcs, test_idcs = range(300), range(300, dataset.__len__())
            TEST_BATCH_SIZE = dataset.__len__() - 300
        else:"""
        train_idcs, test_idcs = range(int((1-cfg.data.test_split_size)*dataset.__len__())), range(int((1-cfg.data.test_split_size)*dataset.__len__()), dataset.__len__()) # 2844 total obs - 711 test, still 25% even without random split, last outbreak 2353
        train_idcs, val_idcs = TTS(train_idcs, test_size=cfg.data.val_split_size, shuffle=cfg.data.shuffle, random_state=cfg.seed)
        VAL_BATCH_SIZE, TEST_BATCH_SIZE = len(val_idcs), len(test_idcs)

    ## Define generator so sampling during training is deterministic and reproducible
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_sampler, val_sampler, test_sampler = SRS(train_idcs, generator=g), SRS(val_idcs), SS(test_idcs)
    train_loader, val_loader, test_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, sampler=train_sampler), DataLoader(dataset, batch_size=VAL_BATCH_SIZE, sampler=val_sampler, shuffle=False), DataLoader(dataset, batch_size=TEST_BATCH_SIZE, sampler=test_sampler, shuffle=False)
    return train_loader, val_loader, test_loader


""" Could use to find units of maximum value, return with dataset and then parse to NN as self.const
counter = len(str(max_number))

# Calculate the nearest unit of length based on the counter
nearest_unit = 10 ** (counter - 1)
"""
