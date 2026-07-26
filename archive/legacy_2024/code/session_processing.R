#################################################################################
# This code is written by Yizhan Gu
# University of California San Diego
# email: yig031@ucsd.edu
#################################################################################


####load packages and clear workspace####
rm(list = ls(all = TRUE))

libs <- c("tidyverse", "ggthemes", "gdata", "lubridate", "hms", "readxl", "gridExtra")
invisible(lapply(libs, library, character.only = TRUE))

#options(digits = 4)

#################################################################################
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
project_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  normalizePath(getwd())
}
setwd(project_dir)

raw_test_data <- read_csv(file.path(project_dir, "test_charging_sessions.csv"))
colnames(raw_test_data)

external_data_dir <- file.path(
  dirname(project_dir),
  "Other",
  "SD_County_Comparison_Chademo_CCS"
)
stations <- read.csv(file.path(external_data_dir, "Station_Table_modified_2.csv"))
UCSD_noDCFC_stations <- stations %>% 
  filter(Max_Power <= 40) %>% 
  filter(grepl("UCSD", Station_Name, ignore.case = TRUE)) %>% 
  select(Station_Name, Max_Power)

UCSD_L2_stations <- stations %>% 
  filter(Max_Power <= 6.7) %>% 
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
raw_data <- read_csv(file.path(external_data_dir, "SD_latest_original_sessions.csv"))
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



#################################################################################
# Update latest dataset to 2024 Sep
raw_data_new <- read_csv("CP_UCSD_raw_Jul16_Sep24.csv")
data_new <- raw_data_new %>% 
  select("User ID", "Start Date", "End Date", "Energy (kWh)", "Charging Time (hh:mm:ss)", "Station Name", "Port Number") %>% 
  rename(driver_id = "User ID", total_energy_dispensed = "Energy (kWh)", session_start_time_la = "Start Date", session_end_time_la = "End Date", charging_time_secs = "Charging Time (hh:mm:ss)", station_name = "Station Name", port = "Port Number") %>% 
  mutate(session_start_time_la = mdy_hm(session_start_time_la), session_end_time_la = mdy_hm(session_end_time_la)) %>% 
  mutate(charging_time_secs = as.numeric(lubridate::hms(charging_time_secs))) %>% 
  mutate(charging_end_time_la = charging_time_secs + session_start_time_la) %>% 
  filter(complete.cases(.),
         total_energy_dispensed >= 1,
         #total_energy_dispensed <= 1.05 * vehicle_battery_capacity,
         difftime(session_end_time_la, session_start_time_la, units = "mins") >= 10,
         date(session_start_time_la) == date(session_end_time_la), # filter out overnight sessions
         station_name %in% UCSD_L2_stations$Station_Name)


# Plotting some statistics
daily_sessions <- data_new %>%
  mutate(date = as.Date(session_start_time_la)) %>%
  group_by(date) %>%
  summarise(sessions = n())

data_new$start_time <- as.POSIXct(format(data_new$session_start_time_la, "%H:%M:%S"), format = "%H:%M:%S") - hours(8)
data_new$end_time <- as.POSIXct(format(data_new$session_end_time_la, "%H:%M:%S"), format = "%H:%M:%S") - hours(8)

p1 <- ggplot(daily_sessions, aes(x = date, y = sessions)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "", x = "Date", y = "Sessions") +
  theme_minimal() +
  theme(legend.title = element_blank())

p2 <- ggplot(data_new, aes(x = start_time)) +
  geom_histogram(bins = 96, fill = "lightblue", color = "white") +
  scale_x_datetime(breaks = scales::date_breaks("1 hour"), labels = scales::date_format("%H")) +
  labs(title = "AT", x = "Time of Day", y = "Sessions") +
  theme_minimal()

p3 <- ggplot(data_new, aes(x = end_time)) +
  geom_histogram(bins = 96, fill = "lightblue", color = "white") +
  scale_x_datetime(breaks = scales::date_breaks("1 hour"), labels = scales::date_format("%H")) +
  labs(title = "DT", x = "Time of Day", y = "Sessions") +
  theme_minimal()

p <- grid.arrange(p1, p2, p3, ncol = 1)







# store
data_new$session_start_time_la <- format(data_new$session_start_time_la, format="%Y-%m-%dT%H:%M:%S")
data_new$session_end_time_la <- format(data_new$session_end_time_la, format="%Y-%m-%dT%H:%M:%S")
data_new$charging_end_time_la <- format(data_new$charging_end_time_la, format="%Y-%m-%dT%H:%M:%S")

data_train <- filter(data_new, year(session_start_time_la) < 2023)
data_test <- filter(data_new, year(session_start_time_la) >= 2023, month(session_start_time_la) <= 9)
  

  
# FIXME: Possibly still need to filter cause the session is suddenly decreasing since someday 2023
  
write_csv(data_train, "clean_charging_sessions.csv")
write_csv(data_test, "clean_test_sessions.csv")

ggsave("session_distriution.png", plot = p, width = 10, height = 6, dpi = 300)


