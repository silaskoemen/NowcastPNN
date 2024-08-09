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

full_df$year_DT_SIN_PRI <- format(full_df$DT_SIN_PRI, "%Y")
full_df$year_DT_NOTIFIC <- format(full_df$DT_NOTIFIC, "%Y")

# Filter rows where year is less than 2012 in either column
filtered_df <- full_df[full_df$year_DT_SIN_PRI >= 2012 & full_df$year_DT_NOTIFIC >= 2012 & full_df$year_DT_SIN_PRI <= 2021 & full_df$year_DT_NOTIFIC <= 2021, ]

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

test_dates = t(read.csv("src/test_dates.csv", header=F))
rownames(test_dates) <- 1:(length(test_dates))

agg_list_epinc = list()
for (i in 1:length(test_dates)) {
  agg_list_epinc[[test_dates[i]]] <- array(NA, dim = c(14, 9, 2)) # 7 CIs + med + min/max
}

levels <- c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)
lower_columns <- (1 - levels) / 2
upper_columns <- (1 + levels) / 2

progress_counter = 1
for(td in test_dates) {
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
  #plot(nowcast, latest_obs = latest_tab)
  samples <- summary(nowcast, type="nowcast_samples") # have chains * iter_sampling samples
  
  ## Use samples for quantiles, min, max (width) and all criteria
  
  for(p in 0:13) {
    temp_date = format(as.Date(td) - p, format="%Y-%m-%d")
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




######### SINGLE OBSERVATION TEST RUN #########

# Target date needs to be final date of reports NOT end of reference dates directly
target_date = as.Date("2015-03-16")
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
plot(nowcast, latest_obs = latest_tab)
samples <- summary(nowcast, type="nowcast_samples") # have chains * iter_sampling samples

## Use samples for quantiles, min, max (width) and all criteria
target_samples = samples[samples$reference_date == target_date]$sample
length(target_samples)

levels = c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)

## Make bounds for every day in current to -13, set in big matrix (14, 9, 2) and that per date as key
bounds <- matrix(nrow = length(levels), ncol = 2)
lower_columns <- (1 - levels) / 2
upper_columns <- (1 + levels) / 2

for (i in seq_along(levels)) {
  #print(paste(temp_lvls[lower_columns[i]], temp_lvls[upper_columns[i]]))
  bounds[i, 1] <- unname(quantile(target_samples, probs=lower_columns[i], na.rm=T))
  bounds[i, 2] <- unname(quantile(target_samples, probs=upper_columns[i], na.rm=T))
}


                           