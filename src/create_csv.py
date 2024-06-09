## Create csv from dbc files and then concatenate all of them for one dataset per region
import numpy as np
import pandas as pd
from os.path import exists

import subprocess


def dbc2csv(raw_filename):
    """ Function to convert .dbc file to .csv file
    Calls an R script to read and convert the file. Note that raw filename should NOT contain the .dbc suffix,
    it's added in the R script automatically for correct naming
    
    Args:
    raw_filename [str]: String name of file
    
    Returns:
    [bool]: status of conversion
    """
    dbc2csv_path = "./dbc2csv.R " + "../data/raw/Dengue/" + " " + "../data/derived/" + " " + raw_filename

    try:
        r_script_path = subprocess.getstatusoutput('which Rscript')[1]
        subprocess.call(r_script_path + " --vanilla " + dbc2csv_path, shell=True)
        return True
    except:
        print("(Rscript) Error converting file: " + raw_filename)

    return False

## Iterate over all states and all years
regions = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
years = range(13, 21)

for r in regions: # takes ~13m40
    for y in years:
        if not exists(f"../data/derived/DENG{r}{y}.csv"): dbc2csv(f"DENG{r}{y}")

## Concatenate, basically append rows and store in one
for r in regions: # takes ~9m
    temp_df = None
    if exists(f"../data/derived/DENG{r}.csv"): # skips creation if already exists, allows for running of whole script without problems
        continue
    for y in years:
        if temp_df is None:
            if exists(f"../data/derived/DENG{r}{y}.csv"): # some have 2013 unavailable
                temp_df = pd.read_csv(f"../data/derived/DENG{r}{y}.csv", index_col=0)
        else:
            # Append rows 
            temp_df = pd.concat((temp_df, pd.read_csv(f"../data/derived/DENG{r}{y}.csv")), ignore_index = True)
    temp_df.to_csv(f"../data/derived/DENG{r}.csv")

