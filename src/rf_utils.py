## Build Random Forest (RF) model from paper https://www.sciencedirect.com/science/article/pii/S0025556421001012
## Uses delay, delay squared, delay cubed, weekday and raw count as predictors (weekday as dummy)
import pandas as pd
import numpy as pd

# Get df from get_dataset, then convert to RF features

def datadf_to_rf_features(df: pd.DataFrame, future_obs = 0):
    ## df has reindexed delays
    # For same-day nowcast, have delay = 0 -> delay sq and cub = 0, raw count is 1st

    # Make new df with count at delay 0 and date to weekday, if future_obs, need to do partial sum available

    # Write fct that uses future_obs for partial sum and delay values

    # Return df
    pass

## Make train and test split with RF

## Train RF

## Predict y_test

## Create prediction intervals

## Store