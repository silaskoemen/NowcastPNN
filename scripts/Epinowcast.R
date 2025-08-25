setwd("/Users/silaskoemen/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Imperial/NowcastPNN/")
library(cmdstanr)
#install_cmdstan(overwrite=TRUE)

library(epinowcast)
enw_set_cache(tools::R_user_dir(package = "epinowcast", "cache"), type = c('session', 'persistent'))
library(data.table)
library(ggplot2)
library(dplyr)

## Read in data, to compare also use SP data
full_df = read.csv("data/derived/DENGSP.csv")

full_df$DT_SIN_PRI <- as.Date(full_df$DT_SIN_PRI)
full_df$DT_NOTIFIC <- as.Date(full_df$DT_NOTIFIC)

full_df$year_DT_SIN_PRI <- as.numeric(format(full_df$DT_SIN_PRI, "%Y"))
full_df$year_DT_NOTIFIC <- as.numeric(format(full_df$DT_NOTIFIC, "%Y"))

# Filter rows where year is less than 2012 in either column
filtered_df <- full_df[(full_df$year_DT_SIN_PRI >= 2012) & (full_df$year_DT_NOTIFIC >= 2012) & (full_df$year_DT_SIN_PRI <= 2021) & (full_df$year_DT_NOTIFIC <= 2021), ]

# Optionally, remove the year columns if they are no longer needed
filtered_df$year_DT_SIN_PRI <- NULL
filtered_df$year_DT_NOTIFIC <- NULL

filtered_df <- rename(filtered_df, reference_date = DT_SIN_PRI, report_date = DT_NOTIFIC)
df <- filtered_df[complete.cases(filtered_df[, c('reference_date', 'report_date')]), ]

## Brauche 'confirm' column mit number per reference und report date
count_combinations <- function(df) {
  df %>%
    group_by(reference_date, report_date) %>%
    summarize(confirm = n()) %>%
    ungroup()
}

cumulative_count_combinations <- function(df) {
  cum_df <- df %>%
    group_by(reference_date, report_date) %>%
    summarize(confirm = n()) %>%
    arrange(reference_date, report_date) %>%
    mutate(cumulative_confirm = cumsum(confirm)) %>%
    ungroup()
  cum_df <- cum_df[c("reference_date", "report_date", "cumulative_confirm")]
  cum_df <- rename(cum_df, confirm = cumulative_confirm)
  return(cum_df)
}

# Example usage
# Assuming your dataframe is named 'data' and has columns 'reference_date' and 'report_date'
# data <- data.frame(reference_date = ..., report_date = ...)

# Apply the function to the dataframe
#result_df <- count_combinations(df)
cum_df <- cumulative_count_combinations(df)
#result_df = cum_df[c("reference_date", "report_date", "cumulative_confirm")]

View(cum_df %>% group_by(reference_date, report_date))
#result_df = rename(result_df, confirm = cumulative_confirm)
setorderv(cum_df, cols = c("reference_date", "report_date"))
cum_dt = data.table(cum_df)
print(cum_dt[reference_date > as.Date("2015-01-01")], n = 100)

library(tidyr)
library(lubridate)
library(purrr)


#############################################
######### RANDOM TEST OBSERVATIONS ##########
#############################################
PAST_UNITS <- 80
test_dates = t(read.csv("src/test_dates_recent.csv", header=F))
rownames(test_dates) <- 1:(length(test_dates))

agg_list_epinc = list()
for (i in 1:length(test_dates)) {
  agg_list_epinc[[test_dates[i]]] <- array(NA, dim = c(14, 9, 2)) # 7 CIs + med + min/max
}

levels <- c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)
lower_columns <- (1 - levels) / 2
upper_columns <- (1 + levels) / 2
## At times estimation diverged and the count cannot be estimated. The estimate is then replaced by the previous day's value and estimation is restarted from the day before.
progress_counter = 1
# agg_list_epinc[[test_dates[progress_counter]]] <- agg_list_epinc[[test_dates[progress_counter-1]]]; progress_counter = progress_counter+1
for(td in test_dates[progress_counter:length(test_dates)]) {
  print(paste0("Date ",progress_counter,"/",length(test_dates), " (",td,")"))
  
  target_date = as.Date(td)
  #data_tab = data.table(result_df)
  data_tab = cum_dt |> enw_filter_report_dates(latest_date = target_date+40)# |> enw_filter_reference_dates(latest_date = target_date, include_days = 30) # first enw_filter_reference_dates(earliest_date = target_date - 30, latest_date = target_date)
  data_tab = enw_complete_dates(data_tab, max_delay = 40)
  #data_tab = data_tab[!is.na(reference_date) & !is.na(report_date)]
  
  retro_tab <- data_tab |>
    enw_filter_report_dates(remove_days = 40) |>
    enw_filter_reference_dates(include_days = PAST_UNITS) # hat as available on day correctly
  
  # Soll wahrscheinlich available looking back plotten,
  # Ergo pro reference date group by und count?
  latest_tab <- data_tab |>
    enw_latest_data() |>
    enw_filter_reference_dates(remove_days = 40, include_days = PAST_UNITS)
  
  on_day_tab = data_tab[(reference_date > target_date - PAST_UNITS) & (reference_date <= target_date)]
  on_day_tab = on_day_tab[, sum(confirm), by = reference_date]
  setnames(on_day_tab, "V1", "confirm")
  
  retro_on_day = retro_tab[, sum(confirm), by = reference_date]
  setnames(retro_on_day, "V1", "confirm")
  
  pobs <- enw_preprocess_data(retro_tab, max_delay = 40)
  
  expectation_module <- enw_expectation(
    ~ 0 + (1 | day), data = pobs
  )
  reference_module <- enw_reference(~1, distribution = "lognormal", data = pobs)
  report_module <- enw_report(~ (1 | day_of_week), data = pobs)
  model <- enw_model(threads = TRUE)
  
  options(mc.cores = 8) # usable cores 
  nowcast <- epinowcast(data = pobs,
                        expectation = expectation_module,
                        reference = reference_module,
                        report = report_module,
                        fit = enw_fit_opts(
                          save_warmup = FALSE, pp = TRUE,
                          chains = 2, threads_per_chain = 2,
                          iter_sampling = 500, iter_warmup = 250,
                          show_messages = T
                        ),
                        model = model
  ) # 43.6 seconds
  
  # Looks like only starts nowcasting from reference target date (latest_date) onwards
  # Obviously not recorded bc not included in reference dates anymore
  samples <- summary(nowcast, type="nowcast_samples") # have chains * iter_sampling samples
  
  ## Use samples for quantiles, min, max (width) and all criteria
  
  for(p in 0:13) {
    temp_date = as.Date(td) - p
    #print(paste("Temp date to investigate", temp_date))
    target_samples = samples[samples$reference_date == temp_date]$sample
    
    # Add levels to agg_list to use later, is array, index p
    bounds <- matrix(nrow = length(levels), ncol = 2)
    
    for (i in seq_along(levels)) {
      #print(paste(temp_lvls[lower_columns[i]], temp_lvls[upper_columns[i]]))
      bounds[i, 1] <- unname(quantile(target_samples, probs=lower_columns[i], na.rm=T))
      bounds[i, 2] <- unname(quantile(target_samples, probs=upper_columns[i], na.rm=T))
    }
    
    agg_list_epinc[[td]][(p+1),,] <- bounds
  }
  progress_counter = progress_counter+1
}

library(jsonlite)

# Convert the named list to JSON and save to a file
json_epinc <- toJSON(agg_list_epinc, pretty = TRUE)
write(json_epinc, file = "data/model_predictions/epinowcast_list_rec_80.json")



#############################################
########### RECENT OBSERVATIONS #############
#############################################
test_dates = t(read.csv("src/test_dates_recent.csv", header=F))
rownames(test_dates) <- 1:(length(test_dates))

agg_list_epinc_timing = list()
for (i in 1:length(test_dates)) {
  agg_list_epinc_timing[[test_dates[i]]] <- array(NA, dim = c(14, 9, 2)) # 7 CIs + med + min/max
}

levels <- c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)
lower_columns <- (1 - levels) / 2
upper_columns <- (1 + levels) / 2

progress_counter = 1
for(td in test_dates[692:length(test_dates)]) {
  print(paste0("Date ",progress_counter,"/",length(test_dates), " (",td,")"))
  
  target_date = as.Date(td)
  #data_tab = data.table(result_df)
  data_tab = cum_dt |> enw_filter_report_dates(latest_date = target_date+40)# |> enw_filter_reference_dates(latest_date = target_date, include_days = 30) # first enw_filter_reference_dates(earliest_date = target_date - 30, latest_date = target_date)
  data_tab = enw_complete_dates(data_tab, max_delay = 40)
  #data_tab = data_tab[!is.na(reference_date) & !is.na(report_date)]
  # CUMULATIVE???, then would explain a lot, yes - cumulative - lorl
  
  retro_tab  <- data_tab |>
    enw_filter_report_dates(remove_days = 40) |>
    enw_filter_reference_dates(include_days = 40) # hat as available on day correctly
  
  # Soll wahrscheinlich available looking back plotten,
  # Ergo pro reference date group by und count?
  latest_tab <- data_tab |>
    enw_latest_data() |>
    enw_filter_reference_dates(remove_days = 40, include_days = 40)
  
  on_day_tab = data_tab[(reference_date > target_date - 40) & (reference_date <= target_date)]
  on_day_tab = on_day_tab[, sum(confirm), by = reference_date]
  setnames(on_day_tab, "V1", "confirm")
  
  retro_on_day = retro_tab[, sum(confirm), by = reference_date]
  setnames(retro_on_day, "V1", "confirm")
  
  pobs <- enw_preprocess_data(retro_tab, max_delay = 40)
  
  expectation_module <- enw_expectation(
    ~ 0 + (1 | day), data = pobs
  )
  reference_module <- enw_reference(~1, distribution = "lognormal", data = pobs)
  report_module <- enw_report(~ (1 | day_of_week), data = pobs)
  model <- enw_model(threads = TRUE)
  
  options(mc.cores = 8) # usable cores 
  nowcast <- epinowcast(data = pobs,
                        expectation = expectation_module,
                        reference = reference_module,
                        report = report_module,
                        fit = enw_fit_opts(
                          save_warmup = FALSE, pp = TRUE,
                          chains = 2, threads_per_chain = 2,
                          iter_sampling = 500, iter_warmup = 250,
                          show_messages = T
                        ),
                        model = model
  ) # 43.6 seconds
  
  # Looks like only starts nowcasting from reference target date (latest_date) onwards
  # Obviously not recorded bc not included in reference dates anymore
  samples <- summary(nowcast, type="nowcast_samples") # have chains * iter_sampling samples
  
  ## Use samples for quantiles, min, max (width) and all criteria
  
  for(p in 0:13) {
    temp_date = as.Date(td) - p
    #print(paste("Temp date to investigate", temp_date))
    target_samples = samples[samples$reference_date == temp_date]$sample
    
    # Add levels to agg_list to use later, is array, index p
    bounds <- matrix(nrow = length(levels), ncol = 2)
    
    for (i in seq_along(levels)) {
      #print(paste(temp_lvls[lower_columns[i]], temp_lvls[upper_columns[i]]))
      bounds[i, 1] <- unname(quantile(target_samples, probs=lower_columns[i], na.rm=T))
      bounds[i, 2] <- unname(quantile(target_samples, probs=upper_columns[i], na.rm=T))
    }
    
    agg_list_epinc_timing[[td]][(p+1),,] <- bounds
  }
  progress_counter = progress_counter+1
}

# delete 2010-07-07, wrongly created instead of 2020


#agg_list_epinc_timing <- agg_list_epinc_timing[1:(length(agg_list_epinc_timing)-1)] # accidentally added wrong date to list, remove 

library(jsonlite)

# Convert the named list to JSON and save to a file
json_epinc_recent <- toJSON(agg_list_epinc_timing, pretty = TRUE)
write(json_epinc_recent, file = "data/model_predictions/epinowcast_list_recent.json")


#############################################
############ Inference Timing ###############
#############################################
test_dates = t(read.csv("src/test_dates_recent.csv", header=F))
NUM_TIMING_OBS = 20
rownames(test_dates) <- 1:(length(test_dates))

levels <- c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)
lower_columns <- (1 - levels) / 2
upper_columns <- (1 + levels) / 2

agg_list_epinc_timing = list()
indiv_time_list = list()
for (i in 1:NUM_TIMING_OBS) {
  agg_list_epinc_timing[[test_dates[i]]] <- array(NA, dim = c(14, 9, 2)) # 7 CIs + med + min/max
  indiv_time_list[[test_dates[i]]] <- NA
}

progress_counter = 1
for(td in test_dates[1:NUM_TIMING_OBS]) {
  start_time = Sys.time()
  print(paste0("Date ", progress_counter, "/", NUM_TIMING_OBS, " (",td,")"))
  
  target_date = as.Date(td, format="%Y-%m-%d")
  #data_tab = data.table(result_df)
  data_tab = cum_dt |> enw_filter_report_dates(latest_date = target_date+40)# |> enw_filter_reference_dates(latest_date = target_date, include_days = 30) # first enw_filter_reference_dates(earliest_date = target_date - 30, latest_date = target_date)
  data_tab = enw_complete_dates(data_tab, max_delay = 40)
  #data_tab = data_tab[!is.na(reference_date) & !is.na(report_date)]
  
  retro_tab  <- data_tab |>
    enw_filter_report_dates(remove_days = 40) |>
    enw_filter_reference_dates(include_days = 40) # hat as available on day correctly
  
  # Soll wahrscheinlich available looking back plotten,
  # Ergo pro reference date group by und count?
  latest_tab <- data_tab |>
    enw_latest_data() |>
    enw_filter_reference_dates(remove_days = 40, include_days = 40)
  
  on_day_tab = data_tab[(reference_date > target_date - 40) & (reference_date <= target_date)]
  on_day_tab = on_day_tab[, sum(confirm), by = reference_date]
  setnames(on_day_tab, "V1", "confirm")
  
  retro_on_day = retro_tab[, sum(confirm), by = reference_date]
  setnames(retro_on_day, "V1", "confirm")
  
  pobs <- enw_preprocess_data(retro_tab, max_delay = 40)
  
  expectation_module <- enw_expectation(
    ~ 0 + (1 | day), data = pobs
  )
  reference_module <- enw_reference(~1, distribution = "lognormal", data = pobs)
  report_module <- enw_report(~ (1 | day_of_week), data = pobs)
  model <- enw_model(threads = TRUE)
  
  options(mc.cores = 8) # usable cores - change as necessary
  nowcast <- epinowcast(data = pobs,
                        expectation = expectation_module,
                        reference = reference_module,
                        report = report_module,
                        fit = enw_fit_opts(
                          save_warmup = FALSE, pp = TRUE,
                          chains = 2, threads_per_chain = 2,
                          iter_sampling = 500, iter_warmup = 250,
                          show_messages = T
                        ),
                        model = model
  ) # 43.6 seconds
  
  # Looks like only starts nowcasting from reference target date (latest_date) onwards
  # Obviously not recorded bc not included in reference dates anymore
  samples <- summary(nowcast, type="nowcast_samples") # have chains * iter_sampling samples
  
  ## Use samples for quantiles, min, max (width) and all criteria
  
  for(p in 0:13) {
    temp_date = as.Date(td) - p
    #print(paste("Temp date to investigate", temp_date))
    target_samples = samples[samples$reference_date == temp_date]$sample
    
    # Add levels to agg_list to use later, is array, index p
    bounds <- matrix(nrow = length(levels), ncol = 2)
    
    for (i in seq_along(levels)) {
      #print(paste(temp_lvls[lower_columns[i]], temp_lvls[upper_columns[i]]))
      bounds[i, 1] <- unname(quantile(target_samples, probs=lower_columns[i], na.rm=T))
      bounds[i, 2] <- unname(quantile(target_samples, probs=upper_columns[i], na.rm=T))
    }
    
    agg_list_epinc_timing[[td]][(p+1),,] <- bounds
  }
  progress_counter = progress_counter+1
  indiv_time_list[[td]] <- start_time - Sys.time()
}

# delete 2010-07-07, wrongly created instead of 2020
times_numeric <- sapply(indiv_time_list, function(x) abs(as.numeric(x, units = "secs")))

mean_time <- mean(times_numeric)
sd_time <- sd(times_numeric)

# Print them
cat("Mean time (seconds):", mean_time, "\n")
cat("Standard deviation (seconds):", sd_time, "\n")

library(jsonlite)

# Convert the named list to JSON and save to a file
json_times_numeric <- toJSON(times_numeric, pretty = TRUE)
write(json_times_numeric, file = "data/model_predictions/times_numeric_epinowcast.json")
