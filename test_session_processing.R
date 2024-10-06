#################################################################################
# This code is written by Yizhan Gu
# University of California San Diego
# email: yig031@ucsd.edu
#################################################################################


####load packages and clear workspace####
rm(list = ls(all = TRUE))

libs <- c("tidyverse", "ggthemes", "gdata", "lubridate", "hms", "readxl")
invisible(lapply(libs, library, character.only = TRUE))

#options(digits = 4)

#################################################################################
dir <- "/Users/admin/Desktop/EV_program/2024Summer_EVResearch"
setwd(dir)

raw_test_data <- read_csv("/Users/admin/Desktop/EV_program/2024Summer_EVResearch/test_charging_sessions.csv")
colnames(raw_test_data)

stations <- read.csv("/Users/admin/Desktop/EV_program/Other/SD_County_Comparison_Chademo_CCS/Station_Table_modified_2.csv")
UCSD_noDCFC_stations <- stations %>% 
  filter(Max_Power <= 40) %>% 
  filter(grepl("UCSD", Station_Name, ignore.case = TRUE)) %>% 
  select(Station_Name, Max_Power)



test_data <- raw_test_data %>% 
  select("User ID", "Start Date", "End Date", "Energy (kWh)", "Charging Time (hh:mm:ss)", "Station Name", "Vehicle Make", "Vehicle Model") %>% 
  rename(driver_id = "User ID", total_energy_dispensed = "Energy (kWh)", session_start_time_la = "Start Date", session_end_time_la = "End Date", charging_time_secs = "Charging Time (hh:mm:ss)", station_name = "Station Name", vehicle_make = "Vehicle Make", vehicle_model = "Vehicle Model") %>% 
  mutate(charging_time_secs = as.numeric(lubridate::hms(charging_time_secs))) %>% 
  mutate(charging_end_time_la = charging_time_secs + session_start_time_la) %>% 
filter(complete.cases(.),
  total_energy_dispensed >= 1,
  #total_energy_dispensed <= 1.05 * vehicle_battery_capacity,
  difftime(session_end_time_la, session_start_time_la, units = "mins") >= 10,
  date(session_start_time_la) == date(session_end_time_la), # filter out overnight sessions
  station_name %in% UCSD_noDCFC_stations$Station_Name)


test_data$Hour_of_day_AT <- as.numeric(format(test_data$session_start_time_la, "%H"))
test_data$Hour_of_day_DT <- as.numeric(format(test_data$session_end_time_la, "%H"))
hist(test_data$Hour_of_day_AT, main = "AT of Hours of the Day", xlab = "Hour of Day")
hist(test_data$Hour_of_day_DT, main = "DT of Hours of the Day", xlab = "Hour of Day")


test_data$session_start_time_la <- format(test_data$session_start_time_la, format="%Y-%m-%dT%H:%M:%S")
test_data$session_end_time_la <- format(test_data$session_end_time_la, format="%Y-%m-%dT%H:%M:%S")
test_data$charging_end_time_la <- format(test_data$charging_end_time_la, format="%Y-%m-%dT%H:%M:%S")
write_csv(test_data, "clean_test_sessions.csv")

#################################################################################
raw_data <- read_csv("/Users/admin/Desktop/EV_program/Other/SD_County_Comparison_Chademo_CCS/SD_latest_original_sessions.csv")
# Convert the UTC datetime columns to POSIXct objects
raw_data$session_start_time_utc <- ymd_hms(raw_data$session_start_time_utc)
raw_data$session_end_time_utc <- ymd_hms(raw_data$session_end_time_utc)

# Convert UTC to Los Angeles time considering DST
raw_data$session_start_time_la <- with_tz(raw_data$session_start_time_utc, tzone = "America/Los_Angeles")
raw_data$session_end_time_la <- with_tz(raw_data$session_end_time_utc, tzone = "America/Los_Angeles")

data <- raw_data %>% 
  select(driver_id, session_start_time_la, session_end_time_la, total_energy_dispensed, charging_time_secs, station_name, vehicle_make, vehicle_model) %>% 
  filter(complete.cases(.),
    total_energy_dispensed >= 1,
    difftime(session_end_time_la, session_start_time_la, units = "mins") >= 10,
    date(session_start_time_la) == date(session_end_time_la), # filter out overnight sessions
    station_name %in% UCSD_noDCFC_stations$Station_Name)


data$Hour_of_day_AT <- as.numeric(format(data$session_start_time_la, "%H"))
data$Hour_of_day_DT <- as.numeric(format(data$session_end_time_la, "%H"))
hist(data$Hour_of_day_AT, main = "AT of Hours of the Day", xlab = "Hour of Day")
hist(data$Hour_of_day_DT, main = "DT of Hours of the Day", xlab = "Hour of Day")


data$charging_end_time_la <- data$charging_time_secs + data$session_start_time_la

# when saving to csv the timezone info is hidden and error may show when using other languages
# convert time to string
data$session_start_time_la <- format(data$session_start_time_la, format="%Y-%m-%dT%H:%M:%S")
data$session_end_time_la <- format(data$session_end_time_la, format="%Y-%m-%dT%H:%M:%S")
data$charging_end_time_la <- format(data$charging_end_time_la, format="%Y-%m-%dT%H:%M:%S")
write_csv(data, "clean_charging_sessions.csv")







