#############################################
############### RIVM Nowcast ################
#############################################
setwd("/Users/silaskoemen/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Imperial/NowcastPNN/")
## Load packages
library(Matrix)
library(tidyverse)
library(scales)
library(RColorBrewer)
library(data.table)
library(dplyr)
library(tidyr)
library(purrr)
library(hash)

blue <- brewer.pal(n = 9, name = "Set1")[2]
oran <- brewer.pal(n = 9, name = "Set1")[5]
grey <- brewer.pal(n = 9, name = "Set1")[9]

# Set color palettes
blue.pal <- brewer.pal(n = 9, name = "Blues")   %>% colorRampPalette
oran.pal <- brewer.pal(n = 9, name = "Oranges") %>% colorRampPalette
grey.pal <- brewer.pal(n = 9, name = "Greys")   %>% colorRampPalette

## Load all functions necessary from subdirectory
source('src/rivm_utils/RIVM_functions.R')

## Load data
full_df = read.csv("data/derived/DENGSP.csv")

PAST_UNITS <- 365
# Remove invalid observations
full_df$DT_SIN_PRI <- as.Date(full_df$DT_SIN_PRI)
full_df$DT_NOTIFIC <- as.Date(full_df$DT_NOTIFIC)
full_df$year_DT_SIN_PRI <- as.numeric(format(full_df$DT_SIN_PRI, "%Y"))
full_df$year_DT_NOTIFIC <- as.numeric(format(full_df$DT_NOTIFIC, "%Y"))
filtered_df <- full_df[full_df$year_DT_SIN_PRI >= 2012 & full_df$year_DT_NOTIFIC >= 2012 & full_df$year_DT_SIN_PRI <= 2021 & full_df$year_DT_NOTIFIC <= 2021, ]

# Rename, filter by non-NA and define final df
filtered_df <- rename(filtered_df, onset.date = DT_SIN_PRI, report.date = DT_NOTIFIC)
filtered_df <- filtered_df[complete.cases(filtered_df[, c('onset.date', 'report.date')]), ]
df <- filtered_df[, c("onset.date", "report.date")]

# Uses max.delay + 1 (R notation of range), so have to remove one 
f.priordelay <- genPriorDelayDist(mean.delay = 4.413471, max.delay = 39, p = 0.99) # 4.413471
# Check that is adds up to one
sum(f.priordelay)

#############################################
############# RANDOM TEST DAYS ##############
#############################################
test_dates = t(read.csv("src/test_dates_recent.csv", header=F)) # simply fill in which test set
rownames(test_dates) <- 1:(length(test_dates))

agg_list = list()
for (i in 1:length(test_dates)) {
  agg_list[[test_dates[i]]] <- array(NA, dim = c(14, 9, 2)) # 7 CIs + med + min/max
}

levels <- c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)

## At times MLE estimation diverged and the count cannot be estimated. The estimate is then replaced by the previous day's value and estimation is restarted from the day before.
progress_counter = 1
for(td in test_dates) {
  print(paste0("Date ",progress_counter,"/",length(test_dates), " (",td,")"))
  ## Initialize relevant data
  rep.data <- dataSetup(
    data         = df,
    start.date   = as.Date(td)-PAST_UNITS-10, # Starting date of outbreak - 2013-01-01+(39-1)
    end.date     = NULL, # Ending date of outbreak (in real-time, leave NULL so end.date = nowcast.date)
    nowcast.date = as.Date(td),    # Nowcast date
    days.back    = PAST_UNITS-1,                    # Number of days back from nowcast.date to include in estimation procedure
    f.priordelay = f.priordelay)          # Prior reporting delay PMF
  
  model.setup <- modelSetup(
    data = rep.data,
    ord = 2,
    kappa = list(u = 1e6, b = 1e6, w = 0.01, s = 1e-6))
  
  nowcast.list <- nowcast(
    data = rep.data,
    model = model.setup,
    levels = levels)
  
  ncst <- nowcast.list$nowcast
  
  ## Go along 14 recent observations
  for(p in 0:13) {
    temp_date = format(as.Date(td) - p, format="%Y-%m-%d")
    #print(paste("Temp date to investigate", temp_date))
    temp_lvls = ncst[ncst$Date == temp_date, ]
    
    # Add levels to agg_list to use later, is array, index p
    bounds <- matrix(nrow = length(levels), ncol = 2)
    lower_columns <- paste((1 - levels) / 2)
    upper_columns <- paste((1 + levels) / 2)
    for (i in seq_along(levels)) {
      #print(paste(temp_lvls[lower_columns[i]], temp_lvls[upper_columns[i]]))
      bounds[i, 1] <- temp_lvls[[lower_columns[i]]]
      bounds[i, 2] <- temp_lvls[[upper_columns[i]]]
    }
    
    agg_list[[td]][(p+1),,] <- bounds
    #print(agg_list[[td]][(p+1),,])
  }
  progress_counter = progress_counter+1
  
}

#plotEpicurve(data = rep.data)
#plotTrapezoid(data = rep.data)

library(jsonlite)

# Convert the named list to JSON and save to a file
json_data <- toJSON(agg_list, pretty = TRUE)
write(json_data, file = "data/model_predictions/RIVM_list_rec_365.json")



#############################################
########### RECENT OBSERVATIONS #############
#############################################
test_dates = t(read.csv("src/test_dates_recent.csv", header=F))
NUM_TIMING_OBS = 20
rownames(test_dates) <- 1:(length(test_dates))

agg_list_epinc_timing = list()
indiv_time_list = list()
for (i in 1:NUM_TIMING_OBS) {
  agg_list_epinc_timing[[test_dates[i]]] <- array(NA, dim = c(14, 9, 2)) # 7 CIs + med + min/max
  indiv_time_list[[test_dates[i]]] <- NA
}

levels <- c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)
source('src/rivm_utils/RIVM_functions.R')

progress_counter = 1
for(td in test_dates[135:length(test_dates)]) { # 134 already finished
  print(paste0("Date ",progress_counter,"/",length(test_dates), " (",td,")"))
  ## Initialize relevant data
  rep.data <- dataSetup(
    data         = df,
    start.date   = as.Date(td)-50, # Starting date of outbreak - 2013-01-01+(39-1)
    end.date     = as.Date(td)+50, # Ending date of outbreak (in real-time, leave NULL so end.date = nowcast.date)
    nowcast.date = as.Date(td),    # Nowcast date
    days.back    = 39,                    # Number of days back from nowcast.date to include in estimation procedure
    f.priordelay = f.priordelay)          # Prior reporting delay PMF
  
  model.setup <- modelSetup(
    data = rep.data,
    ord = 2,
    kappa = list(u = 1e6, b = 1e6, w = 0.01, s = 1e-6))
  
  nowcast.list <- nowcast(
    data = rep.data,
    model = model.setup,
    levels = levels)
  
  ncst <- nowcast.list$nowcast
  
  ## Go along 14 recent observations
  for(p in 0:13) {
    temp_date = format(as.Date(td) - p, format="%Y-%m-%d")
    #print(paste("Temp date to investigate", temp_date))
    temp_lvls = ncst[ncst$Date == temp_date, ]
    
    # Add levels to agg_list to use later, is array, index p
    bounds <- matrix(nrow = length(levels), ncol = 2)
    lower_columns <- paste((1 - levels) / 2)
    upper_columns <- paste((1 + levels) / 2)
    for (i in seq_along(levels)) {
      #print(paste(temp_lvls[lower_columns[i]], temp_lvls[upper_columns[i]]))
      bounds[i, 1] <- temp_lvls[[lower_columns[i]]]
      bounds[i, 2] <- temp_lvls[[upper_columns[i]]]
    }
    
    agg_list_recent[[td]][(p+1),,] <- bounds
    #print(agg_list[[td]][(p+1),,])
  }
  progress_counter = progress_counter+1
  indiv_time_list[[td]] <- start_time - Sys.time()
}

#plotEpicurve(data = rep.data)
#plotTrapezoid(data = rep.data)


#install.packages("jsonlite")
library(jsonlite)

# Convert the named list to JSON and save to a file
json_data_recent <- toJSON(agg_list_recent, pretty = TRUE)
write(json_data_recent, file = "data/model_predictions/RIVM_list_recent.json")

#############################################
############ Inference Timing ###############
#############################################
test_dates = t(read.csv("src/test_dates_recent.csv", header=F))
NUM_TIMING_OBS = 20
rownames(test_dates) <- 1:(length(test_dates))

levels <- c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)
lower_columns <- (1 - levels) / 2
upper_columns <- (1 + levels) / 2

agg_list_rivm_timing = list()
indiv_time_list = list()
for (i in 1:NUM_TIMING_OBS) {
  agg_list_rivm_timing[[test_dates[i]]] <- array(NA, dim = c(14, 9, 2)) # 7 CIs + med + min/max
  indiv_time_list[[test_dates[i]]] <- NA
}

progress_counter = 1
for(td in test_dates[1:NUM_TIMING_OBS]) {
  start_time = Sys.time()
  print(paste0("Date ", progress_counter, "/", NUM_TIMING_OBS, " (",td,")"))
  
  rep.data <- dataSetup(
    data         = df,
    start.date   = as.Date(td)-50, # Starting date of outbreak - 2013-01-01+(39-1)
    end.date     = as.Date(td)+50, # Ending date of outbreak (in real-time, leave NULL so end.date = nowcast.date)
    nowcast.date = as.Date(td),    # Nowcast date
    days.back    = 39,                    # Number of days back from nowcast.date to include in estimation procedure
    f.priordelay = f.priordelay)          # Prior reporting delay PMF
  
  model.setup <- modelSetup(
    data = rep.data,
    ord = 2,
    kappa = list(u = 1e6, b = 1e6, w = 0.01, s = 1e-6))
  
  nowcast.list <- nowcast(
    data = rep.data,
    model = model.setup,
    levels = levels)
  
  ncst <- nowcast.list$nowcast
  
  ## Go along 14 recent observations
  for(p in 0:13) {
    temp_date = format(as.Date(td) - p, format="%Y-%m-%d")
    temp_lvls = ncst[ncst$Date == temp_date, ]
    
    # Add levels to agg_list to use later, is array, index p
    bounds <- matrix(nrow = length(levels), ncol = 2)
    lower_columns <- paste((1 - levels) / 2)
    upper_columns <- paste((1 + levels) / 2)
    for (i in seq_along(levels)) {
      bounds[i, 1] <- temp_lvls[[lower_columns[i]]]
      bounds[i, 2] <- temp_lvls[[upper_columns[i]]]
    }
    
    agg_list_rivm_timing[[td]][(p+1),,] <- bounds
  }
  progress_counter = progress_counter+1
  indiv_time_list[[td]] <- start_time - Sys.time()
}

times_numeric <- sapply(indiv_time_list, function(x) abs(as.numeric(x, units = "secs")))

mean_time <- mean(times_numeric)
sd_time <- sd(times_numeric)

# Print them
cat("Mean time (seconds):", mean_time, "\n")
cat("Standard deviation (seconds):", sd_time, "\n")

library(jsonlite)

# Convert the named list to JSON and save to a file
json_times_numeric <- toJSON(times_numeric, pretty = TRUE)
write(json_times_numeric, file = "data/model_predictions/times_numeric_rivm.json")

