import numpy as np
import pandas as pd
from epiweeks import Week, Year
from datetime import datetime

def reporting_sums_weekly(date, past_weeks = 7, max_delay = 6, state = "SP"):
    week = pd.to_datetime(date, format=date_format).week
    result_df = dengac13.copy()
    result_df["WK_SIN_PRI"] = result_df['DT_SIN_PRI'].apply(lambda x: x.week) # convert each date to corresponding calendar week
    result_df["WK_NOTIFIC"] = result_df['DT_NOTIFIC'].apply(lambda x: x.week)
    result_df["delay_week"] = result_df['WK_NOTIFIC'] - result_df['WK_SIN_PRI']
    #print(result_df[["WK_SIN_PRI", "WK_NOTIFIC", "delay_week"]])
    result_df = result_df.groupby('WK_SIN_PRI')['delay'].apply(lambda x: x.value_counts().reindex(range(max_delay))).unstack(fill_value=0)

    # Rename columns to reflect delays
    result_df.columns = [f'delay_{col}' for col in result_df.columns]

    # Scale delay values by max
    max_val = result_df[result_df.columns[1:]].max(axis=1).max(axis=0)

    # Reset index to move 'DT_SIN_PRI' back to a column
    result_df.reset_index(inplace=True)

    result_df.fillna(0, inplace=True)
    result_df = result_df[result_df.WK_SIN_PRI <= week].iloc[-past_weeks:, :].reset_index()# correct length

    ## Mask lower triangle bc not reported
    result_df.drop("index", axis = 1, inplace=True)

    result_array = np.array(result_df.iloc[:, 1:].values, dtype = np.float32)
    #print(f"Array shape: {result_array.shape}")
    #print(f"Sums for y with shape {result_array.sum(axis = 1).shape}")
    #print(result_array.sum(axis = 1))
    y = result_array.sum(axis = 1)[-1]
    mask = np.zeros((past_weeks, max_delay), dtype=bool)
    #if past_days > max_delay:
    #    mask[:past_days-max_delay, :] = False
    for p in range(max_delay): # not correct!!
        for d in range(max_delay):
            if p + d >= max_delay:
                mask[p+(past_weeks-max_delay), d] = True
    print(mask, mask.shape)
    result_array[mask] = 0.

    result_array = result_array.sum(axis = 1)   
    #print(f"Resulting array shape: {result_array.shape}")
    result_array /= max_val
    
    return result_array, y

## Have fct to directly return dataset given name and past_weeks and max_delay

def get_dataset(weeks = True, past_units = 6, max_delay = 6, state = "SP"):
    """ Have to return the iterable dataset, so first read in csv file, then convert to delay-format
    Then feed to iterable dataset and return that
    
    Args:
    
    Returns:
    """

    dengac13 = pd.read_csv("../data/derived/DENGAC13.csv", index_col=0)
    return -1