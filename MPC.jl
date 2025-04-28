#=
Author: Yizhan Gu
Description: This is the main file for the MPC project
Affiliation: University of California San Diego
Email: yig031@ucsd.edu
All rights reserved
=#
# NOTE: change the comments of VERSION based on performance of methods!

####################################################
# SECTION: Testing if Julia works and set working directory
print("hello world")
cd("/Users/admin/Desktop/EV_program/2024Summer_EVResearch")
print(pwd())

####################################################
# Importing the necessary packages
using Pkg
packages = ["JuMP", "Ipopt", "LinearAlgebra", "Plots", "Random", "CSV", "DataFrames", "Statistics", "StatsBase", "ProgressMeter", "Dates", "DataFramesMeta", "Distributions", "JLD", "GaussianMixtures", "Holidays", "AutoMLPipeline", "StatsPlots", "IJulia", "Gurobi", "Flux", "MLJ", "TimeSeries", "Metal", "Optimisers"]

Pkg.Registry.update()
for package in packages
    Pkg.add(package)
end

for package in packages
    try
        @eval using $(Symbol(package))
    catch
        print("Error: ", package, " is not loaded")
    end
end
print("All packages are successfully loaded")


# Add the TransformersLite package by https://github.com/LiorSinai/TransformersLite.jl?tab=readme-ov-file
# Pkg.rm("TransformersLite")
Pkg.develop(path="/Users/admin/Desktop/EV_program/2024Summer_EVResearch/TransformersLite")
# Pkg.activate("/Users/admin/Desktop/EV_program/2024Summer_EVResearch/TransformersLite")
Pkg.status()
Pkg.instantiate()

using TransformersLite

# Test if TransformersLite works
TransformerBlock(4, 32, 128; pdrop=0.1)


# Clear console
print("\033c") # Or REPL: Ctrl + L

####################################################
# Data preprocessing
CP_data = CSV.read("CP_UCSD_clean_Jul16_Sep24.csv", DataFrame)
print(first(CP_data))

CP_data_clean = select(CP_data,
"UserID" => :driver_id,
"StartDate" => :session_start_time_la,
"EndDate" => :session_end_time_la,
"Energy(kWh)" => :total_energy_dispensed,
"StationName" => :station_name,
"PortNumber" => :port,
"PortType" => :portyype
)

CP_data_clean = dropmissing(CP_data_clean)
CP_data_clean.session_start_time_la = DateTime.(CP_data_clean.session_start_time_la, dateformat"yyyy-mm-dd HH:MM:SS")
CP_data_clean.session_end_time_la = DateTime.(CP_data_clean.session_end_time_la, dateformat"yyyy-mm-dd HH:MM:SS")
CP_data_clean = filter(row -> !any(ismissing, row), CP_data_clean)
CP_data_clean = filter(row -> row.total_energy_dispensed >= 1, CP_data_clean)
CP_data_clean = filter(row -> dayofmonth(row.session_start_time_la) == dayofmonth(row.session_end_time_la), CP_data_clean)
CP_data_clean = filter(row -> 
    (hour(row.session_end_time_la) - hour(row.session_start_time_la)) * 60 + 
    (minute(row.session_end_time_la) - minute(row.session_start_time_la)) > 10, 
    CP_data_clean)
CP_data_clean = filter(row -> occursin("UCSD", row.station_name), CP_data_clean)
CP_data_clean = filter(row -> row.portyype == "Level 2", CP_data_clean)
CP_data_clean = filter(row -> row.total_energy_dispensed / ((Dates.value(row.session_end_time_la - row.session_start_time_la) / (3600 * 1000))) <= 6.6, CP_data_clean)
select!(CP_data_clean, Not(:portyype))

CP_data_clean_train = filter(row -> Dates.year(row.session_start_time_la) <= 2022 || (Dates.year(row.session_start_time_la) == 2023 && Dates.month(row.session_start_time_la) <= 6), CP_data_clean)
CP_data_clean_test = filter(row -> Dates.year(row.session_start_time_la) == 2023 && Dates.month(row.session_start_time_la) >= 7 && Dates.month(row.session_start_time_la) <= 9, CP_data_clean)
CSV.write("train_charging_sessions.csv", CP_data_clean_train)
CSV.write("test_charging_sessions.csv", CP_data_clean_test)


####################################################
# Read data and plot statistics
charging_sessions = CSV.read("train_charging_sessions.csv", DataFrame)
charging_sessions = sort(charging_sessions, [:session_start_time_la])
unique_stations_ports = CSV.read("unique_stations_ports.csv", DataFrame)


hist_AT_all = histogram(Time.(charging_sessions.session_start_time_la), bins = 96, alpha=0.5, label="AT", xlabel="AT", ylabel="Sessions")
hist_DT_all = histogram(Time.(charging_sessions.session_end_time_la), bins = 96, alpha=0.5, label="DT", xlabel="DT", ylabel="Sessions")
charging_sessions.date = Date.(charging_sessions.session_start_time_la)
daily_sessions = groupby(charging_sessions, :date)
daily_sessions_count = combine(daily_sessions, nrow)

bar_plot = bar(daily_sessions_count.date, 
    daily_sessions_count.nrow,
    xlabel = "", 
    ylabel = "Daily Sessions",
    legend = false,
    rotation = 45,
    alpha = 0.1,
    color = :blue
)

p_stat = plot(hist_AT_all, hist_DT_all, bar_plot, layout = (3, 1), size = (1000, 800))
savefig(p_stat, "Data_distribution.png")

select!(charging_sessions, Not(:date))
charger_sessions = combine(groupby(charging_sessions, [:station_name, :port]), 
                      :total_energy_dispensed => x -> collect(x) => :ED)

unique_stations_ports = unique(charger_sessions[:, [:station_name, :port]])
unique_stations_ports = sort(unique_stations_ports, [:station_name, :port])
# CSV.write("unique_stations_ports.csv", unique_stations_ports)



####################################################
# MPC optimization on EV charging cost minimization and peak shaving
#   V0G: Standard EV charging without any smart grid interaction.
#	V1G: Optimizes the charging process for cost savings or environmental benefits.
#	V2G: EV discharge to the grid.
# https://ieeexplore.ieee.org/document/10184283
# https://github.com/rdeits/DynamicWalking2018.jl/blob/master/notebooks/6.%20Optimization%20with%20JuMP.ipynb


# Electricity rates
# https://www.sdge.com/residential/pricing-plans/about-our-pricing-plans/whenmatters
r_energy_ur = 0.00671 # $/kWh
r_energy_summer_onpeak = 0.11957 + r_energy_ur # $/kWh
r_energy_summer_offpeak = 0.10008 + r_energy_ur # $/kWh
r_energy_winter_onpeak = 0.09955 + r_energy_ur # $/kWh
r_energy_winter_offpeak = 0.08835 + r_energy_ur # $/kWh
r_energy_midseason_onpeak = 0.5 * (r_energy_summer_onpeak + r_energy_winter_onpeak) # $/kWh
r_energy_midseason_offpeak = 0.5 * (r_energy_summer_offpeak + r_energy_winter_offpeak) # $/kWh
r_power_summer_onpeak = 9.78 + 19.14 # $/kW
r_power_winter_onpeak = 19.23 # $/kW
r_power_midseason_onpeak = 0.5 * (r_power_summer_onpeak + r_power_winter_onpeak) # $/kW
r_power_nc = 24.48 # $/kW

# Parameters
P_max = 6.6 # kW
T = 24 # horizon
N = 96 # steps
delta_t = T / N # time slot
N_start_idx = Int(16 / delta_t + 1)
N_end_idx = Int(21 / delta_t)
DWR_charge = 0.0

# Function to determine the day type
function datetype(time::DateTime)
    day = Dates.dayofweek(time)
    date = Date(time)
    usa_holidays = Holidays.UnitedStates()
    
    if date in usa_holidays
        return "holiday"
    elseif day in 1:5
        return "weekday"
    elseif day in 6:7
        return "weekend"
    end
end


function get_season(date::DateTime)
    month = Dates.month(date)
    if month in 3:5
        return "spring"
    elseif month in 6:8
        return "summer"
    elseif month in 9:11
        return "fall"
    else
        return "winter"
    end
end


####################################################
# MPC main function
function run_mpc(data_forecast::DataFrame, data_today::DataFrame, method::String, base::String)
    # Define rates based on season
    data_today.AT = ceil.(Float64.(Dates.hour.(data_today.session_start_time_la)) + Float64.(Dates.minute.(data_today.session_start_time_la)) / 60, digits=2)
    today_update = Date.(data_today.session_start_time_la[1])

    season = get_season(data_today.session_start_time_la[1])
    if season == "summer"
        r_energy_onpeak = r_energy_summer_onpeak
        r_energy_offpeak = r_energy_summer_offpeak
        r_power_onpeak = r_power_summer_onpeak
    elseif season == "winter"
        r_energy_onpeak = r_energy_winter_onpeak
        r_energy_offpeak = r_energy_winter_offpeak
        r_power_onpeak = r_power_winter_onpeak
    else
        r_energy_onpeak = r_energy_midseason_onpeak
        r_energy_offpeak = r_energy_midseason_offpeak
        r_power_onpeak = r_power_midseason_onpeak
    end

    # Size will be different for each time step
    L_mpc = zeros(N)
    P_mpc = Vector{Vector{Float64}}(undef, N)
    E_mpc = Vector{Vector{Float64}}(undef, N)
    E_tmp = Vector{Float64}()
    Optimal = zeros(Bool, N)
    Status = Int[]
    iterations = Int[]
    if base == "EV"
        N_ev_last = size(data_forecast, 1)
    elseif base == "Charger"
        N_charger = size(unique_stations_ports, 1)
    end

    # Setup Gurobi environment in case of printing license information every time
    GUROBI_ENV = Gurobi.Env()

    @showprogress for k in 1:N
        # SECTION: Preprocessing
        data_forecast_update = DataFrame()

        # No real-time update
        if method == "Perfect"
            data_forecast_update = copy(data_forecast)
        # Real-time update at every time step: only arrived EVs can be charged and considered in the MPC
        else
            update_time = (k - 1) * delta_t
            last_step_time = (k - 2) * delta_t

            arrived_sessions_today = filter(row -> row.AT <= update_time, data_today)
            arrived_sessions_forecast = filter(row -> row.AT <= update_time, data_forecast)
            arrived_sessions_today.ED = zeros(size(arrived_sessions_today, 1))
            arrived_sessions_today.DT = zeros(size(arrived_sessions_today, 1))
            # arrived_sessions_step = filter(row -> last_step_time < row.AT <= update_time, data_today)

            if method == "Noforecast"
                if !all(arrived_sessions_forecast.AT .== arrived_sessions_today.AT)
                    error("Noforecast method should have the same AT as perfect forecast")
                elseif isempty(arrived_sessions_forecast)
                    data_forecast_update = copy(data_forecast[1:2, :])
                    # data_forecast_update.AT .= min.(data_forecast_update.AT .+ 1, 23.99)
                    # data_forecast_update.DT .= min.(data_forecast_update.DT .+ 1, 23.99)
                    data_forecast_update.ED .= 0
                elseif !isempty(arrived_sessions_forecast)
                    data_forecast_update = copy(arrived_sessions_forecast)
                end
            elseif method in ["Persistence", "Statistic", "LSTM"]
                if isempty(arrived_sessions_today)
                    data_forecast_update = copy(data_forecast[1:2, :])
                    # postpone time so that unarrived EVs are not charged -- lead to index out of bounds and AT > DT
                    # data_forecast_update.AT .= min.(data_forecast_update.AT .+ 1, 23.99)
                    # data_forecast_update.DT .= min.(data_forecast_update.DT .+ 1, 23.99)
                    data_forecast_update.ED .= 0
                elseif !isempty(arrived_sessions_today)
                    # VERSION: Assume all real AT, DT, ED, PD are known after arrival of EVs to make sure the area of load plot is the same as the real data
                    arrived_sessions_today.DT = floor.(Float64.(Dates.hour.(arrived_sessions_today.session_end_time_la)) + Float64.(Dates.minute.(arrived_sessions_today.session_end_time_la)) / 60, digits=2)
                    arrived_sessions_today.ED = arrived_sessions_today.total_energy_dispensed

                    #= VERSION: Assume only real AT is known after arrival of EVs
                    for i in 1:size(arrived_sessions_today, 1)
                        closest_idx = argmin(abs.(arrived_sessions_forecast.AT .- arrived_sessions_today.AT[i]))
                        arrived_sessions_today.PD[i] = arrived_sessions_forecast.PD[closest_idx]
                        arrived_sessions_today.DT[i] = arrived_sessions_today.PD[i] + arrived_sessions_today.AT[i]
                        if method == "Persistence"
                            arrived_sessions_today.ED[i] = arrived_sessions_forecast.ED[closest_idx]
                        elseif method == "Statistic"
                            arrived_sessions_today.ED[i] = arrived_sessions_forecast.ED[closest_idx] * M # Difference
                        end
                    end
                    =#

                    noarrived_sessions_forecast = filter(row -> row.AT > update_time && row.ED > 0, data_forecast) 
                    # noarrived_sessions_forecast = filter(row -> row.ED >= 0.5 * P_max * delta_t, noarrived_sessions_forecast) # Otherwise too many variables
                    noarrived_sessions_today = copy(noarrived_sessions_forecast)
                    if method in ["Persistence", "Statistic"]
                        if isempty(arrived_sessions_forecast)
                            # Increase ED of the forecasted EVs
                            ED_candidate = noarrived_sessions_forecast.ED * (1 + k / N)^(1/2)
                            ED_max = max.(P_max * (floor.(Int, noarrived_sessions_forecast.DT / T * N) .- ceil.(Int, noarrived_sessions_forecast.AT / T * N) .+ 1) * delta_t, 0)
                            noarrived_sessions_today.ED = min.(ED_candidate, ED_max)
                        elseif !isempty(arrived_sessions_forecast)
                            # NOTE: for persistence forecast ED could be super large that full power charging cannot meet the ED, so take the sqrt
                            EV_ratio = size(arrived_sessions_today, 1) / size(arrived_sessions_forecast, 1)
                            # VERSION: change the ED accordingly
                            ED_candidate = noarrived_sessions_forecast.ED * EV_ratio^(1/2)
                            ED_max = max.(P_max * (floor.(Int, noarrived_sessions_forecast.DT / T * N) .- ceil.(Int, noarrived_sessions_forecast.AT / T * N) .+ 1) * delta_t, 0)
                            noarrived_sessions_today.ED = min.(ED_candidate, ED_max)
                            #= VERSION: partially use the historical session
                            if EV_ratio <= 1
                                # Randomly pick nonredundant rows from `noarrived_sessions_forecast`
                                num_to_pick = round(Int, EV_ratio * size(noarrived_sessions_forecast, 1))
                                noarrived_sessions_today = noarrived_sessions_forecast[rand(1:end, num_to_pick), :]
                            elseif EV_ratio > 1
                                # Randomly pick nonredundant rows and combine with `noarrived_sessions_forecast`
                                n_duplicate = floor(Int, EV_ratio) 
                                num_to_pick = round(Int, (EV_ratio - n_duplicate) * size(noarrived_sessions_forecast, 1))
                                additional_sessions = noarrived_sessions_forecast[rand(1:end, num_to_pick), :]
                                ED_candidate = noarrived_sessions_forecast.ED * sqrt(EV_ratio)
                                ED_max = max.(P_max * (floor.(Int, noarrived_sessions_forecast.PD / T * N) .-1) * delta_t, 0)
                                noarrived_sessions_forecast.ED = min.(ED_candidate, ED_max)
                                noarrived_sessions_today = vcat(noarrived_sessions_forecast, additional_sessions)
                            end
                            =#
                        end
                    elseif method == "LSTM"
                        forecast_station_port = unique(noarrived_sessions_forecast[:, [:station_name, :port]])
                        today_station_port = unique(arrived_sessions_today[:, [:station_name, :port]])
                        combined_sessions = DataFrame()
                        strange_site = DataFrame()
                        ED_threshold = 6
                        for site in eachrow(forecast_station_port)
                            if site in eachrow(today_station_port)
                                real_DT_max = maximum(arrived_sessions_today.DT[findall(row -> row.station_name == site.station_name && row.port == site.port, eachrow(arrived_sessions_today))])
                                noarrived_sessions_today = filter(row -> !(row.station_name == site.station_name && row.port == site.port && row.AT <= real_DT_max), noarrived_sessions_today)
                            end
                            # Combine the continuous forecast sessions of each station and port
                            site_sessions = filter(row -> row.station_name == site.station_name && row.port == site.port, noarrived_sessions_today)
                            if isempty(site_sessions)
                                continue
                            elseif size(site_sessions, 1) > 1
                                site_sessions[!, :AT_diff] .= vcat(0, diff(site_sessions.AT))
                                if all(site_sessions.AT_diff[2:end] .== 0.25)
                                    combined_site_sessions = site_sessions[1, :]
                                    combined_site_sessions.DT = site_sessions.DT[end]
                                    combined_site_sessions.ED = sum(site_sessions.ED)
                                    push!(combined_sessions, combined_site_sessions)
                                else
                                    print("Strange site sessions: ", site_sessions, "k:", k, "\n")
                                    strange_site = vcat(strange_site, site_sessions)
                                    site_sessions = filter(row -> row.ED >= 0.1, site_sessions)
                                    combined_sessions = vcat(combined_sessions, site_sessions)
                                end
                            elseif size(site_sessions, 1) == 1
                                site_sessions[!, :AT_diff] .= 0.0
                                push!(combined_sessions, site_sessions[1, :])
                            end
                        end
                        if !isempty(combined_sessions)
                            combined_sessions = select(combined_sessions, Not(:AT_diff))
                            noarrived_sessions_today = copy(combined_sessions)
                            #= Update the ED
                            EV_ratio = sum(arrived_sessions_today.ED) / sum(arrived_sessions_forecast.ED)
                            if EV_ratio >= ED_threshold
                                ED_candidate = noarrived_sessions_today.ED
                            else
                                ED_candidate = noarrived_sessions_today.ED * EV_ratio^(1/4)
                            end
                            ED_max = max.(P_max * (floor.(Int, noarrived_sessions_today.DT / T * N) .- ceil.(Int, noarrived_sessions_today.AT / T * N) .+ 1) * delta_t, 0)
                            noarrived_sessions_today.ED = min.(ED_candidate, ED_max)
                            =#
                        else
                            noarrived_sessions_today = copy(combined_sessions)
                        end
                    end
                    data_forecast_update = vcat(arrived_sessions_today, noarrived_sessions_today)
                end
            end
        end

        if size(data_forecast_update, 1) == 0
            error("Wrong preprocessing before MPC")
        end

        # SECTION: MPC optimization
        # model = Model(Gurobi.Optimizer)
        model_mpc = JuMP.Model(() -> Gurobi.Optimizer(GUROBI_ENV)) 
        # Set Gurobi solver options
        set_optimizer_attribute(model_mpc, "OutputFlag", 0)  # Suppress solver output
        set_optimizer_attribute(model_mpc, "Threads", 6)   # Use multiple threads

        #= model = Model(Ipopt.Optimizer)
        # Set Ipopt solver options
        set_optimizer_attribute(model, "max_iter", 2000)  # Set maximum number of iterations
        set_optimizer_attribute(model, "tol", 1e-4)       # Set tolerance for convergence
        set_optimizer_attribute(model, "acceptable_tol", 1e-3)  # Set acceptable tolerance
        set_optimizer_attribute(model, "print_level", 0)  # Set printing level (0: no output, 5: full output)
        =#
        # set_optimizer_attribute(model, "constr_viol_tol", 1e-3)  # Set constraint violation tolerance
        # set_optimizer_attribute(model, "warm_start_init_point", "yes")


        if base == "EV"
            N_ev = size(data_forecast_update, 1) # Combine the forecasted and arrived EVs

            if N_ev > N_ev_last
                flag = "more"
            elseif N_ev < N_ev_last
                flag = "fewer"
            elseif N_ev == N_ev_last
                flag = "same"
            end

            @variables model_mpc begin
                P[k:N, 1:N_ev] >= 0
                L[k:N] >= 0
                E[k:N, 1:N_ev] >= 0
                gamma_nc_k >= 0
                gamma_onpeak_k >= 0
            end
    
            # Constraints
            AT = data_forecast_update.AT # 0.00-23.99
            DT = data_forecast_update.DT # 0.00-23.98
            ED = data_forecast_update.ED
            # PD = data_forecast_update.PD
            AT_idx = ceil.(Int, AT / T * N) .+ 1 # Note: idx can only be 1-97 (00:00 as 0+1) for comparison with k
            DT_idx = floor.(Int, DT / T * N) .+ 1 # 1-96
    
            if k == 1
                @constraint(model_mpc, [i=1:N_ev], E[k, i] == 0) # Initial energy state
            elseif k >= 2
                if flag == "more"
                    E_tmp = vcat(E_tmp, zeros(N_ev - N_ev_last))
                elseif flag == "fewer"
                    E_tmp = E_tmp[1:N_ev]
                end
    
                @constraint(model_mpc, [i=1:N_ev], E[k, i] == E_tmp[i] + P[k, i] * delta_t) # E_tmp is the energy state vector stored at the previous loop of k
                if k <= N - 1  
                    @constraint(model_mpc, [t=k+1:N, i=1:N_ev], E[t, i] == E[t-1, i] + P[t, i] * delta_t)
                end
            end
            
            for i in 1:N_ev
                # Charing can happen at DT_idx, or otherwise for AT_idx[i] == DT_idx[i] the ED can never be reached
                # @constraint(model, [t=k:N], E[t, i] <= ED[i])
                # NOTE: Avoid using elseif in multiple cases!!!
                # @constraint(model, [t=k:N], 0 <= P[t, i] <= P_max)

                if AT_idx[i] > DT_idx[i]
                    print("AT", AT_idx, "\n", "DT", DT_idx, "\n", k, "\n")
                    error("AT should be less than DT")
                end
                if AT_idx[i] <= k && k <= DT_idx[i]
                    # @constraint(model, 0.9 * ED[i] <= E[DT_idx[i], i] <= ED[i]) # Non-strict
                    @constraint(model_mpc, E[DT_idx[i], i] == ED[i]) # Strict
                    @constraint(model_mpc, [t=k:DT_idx[i]], 0 <= P[t, i] <= P_max)
                    if DT_idx[i] <= N - 1
                        @constraint(model_mpc, [t=DT_idx[i]+1:N], P[t, i] == 0)
                    end
                else
                    if DT_idx[i] < k
                        @constraint(model_mpc, [t=k:N], P[t, i] == 0)
                    end
                    if  k < AT_idx[i]
                        @constraint(model_mpc, E[DT_idx[i], i] == ED[i])
                        @constraint(model_mpc, [t=k:AT_idx[i]-1], P[t, i] == 0)
                        @constraint(model_mpc, [t=AT_idx[i]:DT_idx[i]], 0 <= P[t, i] <= P_max)
                        if DT_idx[i] <= N - 1
                            @constraint(model_mpc, [t=DT_idx[i]+1:N], P[t, i] == 0)
                        end
                    end
                end
            end
        
            @constraint(model_mpc, [t=k:N], L[t] == sum(P[t, i] for i in 1:N_ev))
        elseif base == "Charger"
            combined_data_forecast_update = combine(groupby(data_forecast_update, [:station_name, :port]), 
            :AT => x -> collect(x) => :AT,
            :DT => x -> collect(x) => :DT,
            :ED => x -> collect(x) => :ED)

            chargers_update = combined_data_forecast_update[:, [:station_name, :port]]
            chargers_update = sort(chargers_update, :station_name)

            if !(all(row -> any(r -> r == row, eachrow(unique_stations_ports)), eachrow(chargers_update)))
                # CSV.write("chargers_update_wrong.csv", chargers_update)
                error("Charger not in the unique_stations_ports")
            end

            missing_rows = filter(row -> !(row in eachrow(chargers_update)), eachrow(unique_stations_ports))
            missing_rows = DataFrame(missing_rows) 
            missing_rows.AT = fill(23.75, nrow(missing_rows))
            missing_rows.DT = fill(23.75, nrow(missing_rows))
            missing_rows.ED = fill(0.0, nrow(missing_rows))
            missing_rows = combine(groupby(missing_rows, [:station_name, :port]), 
            :AT => x -> collect(x) => :AT,
            :DT => x -> collect(x) => :DT,
            :ED => x -> collect(x) => :ED)

            combined_data_forecast_update = vcat(combined_data_forecast_update, missing_rows)
            combined_data_forecast_update = sort(combined_data_forecast_update, [:station_name, :port])

            if size(combined_data_forecast_update, 1) != size(unique_stations_ports, 1)
                error("Charger size mismatch")
            end
            
            @variables model_mpc begin
                P[k:N, 1:N_charger] >= 0
                L[k:N] >= 0
                E[k:N, 1:N_charger] >= 0
                gamma_nc_k >= 0
                gamma_onpeak_k >= 0
            end

            ED_vector = zeros(Float64, N_charger)
            
            # Constraints
            for i in 1:N_charger
                # Extract the AT and DT for the j-th charger
                AT_values = getfield.(combined_data_forecast_update.AT_function[i], 1)
                DT_values = getfield.(combined_data_forecast_update.DT_function[i], 1)
                ED_values = getfield.(combined_data_forecast_update.ED_function[i], 1)
                busy_idx = []
                ED = 0 # Total energy demand of a charger

                if k == 1
                    @constraint(model_mpc, E[k, i] == 0) # Initial energy state
                elseif k >= 2
                    @constraint(model_mpc, E[k, i] == E_tmp[i] + P[k, i] * delta_t)
                    if k <= N - 1  
                        @constraint(model_mpc, [t=k+1:N], E[t, i] == E[t-1, i] + P[t, i] * delta_t)
                    end
                end

                # FIXME: charger forecast not know the number of j, more loose version at the end of day meet ED
                for j in eachindex(AT_values)
                    at_idx = ceil(Int, AT_values[j] / T * N) + 1
                    dt_idx = floor(Int, DT_values[j] / T * N) + 1
                    ed = ED_values[j]
                    ED += ed
                    push!(busy_idx, at_idx:dt_idx)

                    if at_idx > dt_idx
                        print("AT", at_idx, "\n", "DT", dt_idx, "\n", k, "\n")
                        error("AT should be less than DT")
                    end
                    if k <= dt_idx
                        @constraint(model_mpc, E[dt_idx, i] == ED)
                    end
                end

                ED_vector[i] = ED
                busy_idx = reduce(vcat, busy_idx)
                vacant_idx = setdiff(1:N, busy_idx)
                valid_vacant_idx = intersect(vacant_idx, k:N)

                @constraint(model_mpc, [t=k:N], 0 <= P[t, i] <= P_max)
                @constraint(model_mpc, [t in valid_vacant_idx], P[t, i] == 0)
                
            end
            @constraint(model_mpc, [t=k:N], L[t] == sum(P[t, i] for i in 1:N_charger))

        end

        # Peak horizons
        Index_onpeak = []
        Index_offpeak = []

        if k < N_start_idx
            Index_onpeak = N_start_idx:N_end_idx
            Index_offpeak = vcat(k:N_start_idx-1, N_end_idx+1:N)
        elseif N_start_idx <= k <= N_end_idx
            Index_onpeak = k:N_end_idx
            Index_offpeak = N_end_idx+1:N
        elseif k > N_end_idx
            Index_onpeak = []
            Index_offpeak = k:N
        end        

        if sort(vcat(Index_onpeak, Index_offpeak)) != k:N
            print("Index_onpeak: ", Index_onpeak, "\n", "Index_offpeak: ", Index_offpeak, "\n", "k: ", k, "\n")
            error("Index_onpeak and Index_offpeak error")
        end

        @constraint(model_mpc, [t=k:N], gamma_nc_k >= L[t])
        # @constraint(model, gamma_nc_k .>= L)
        # @constraint(model, [t=k:N], gamma_nc_k == maximum(L[t])) # This is not working for IPOPT

        if !isempty(Index_onpeak)
            @constraint(model_mpc, [t in Index_onpeak], gamma_onpeak_k >= L[t])
            # @constraint(model, gamma_onpeak_k == maximum(L[t] for t in Index_onpeak))
        elseif isempty(Index_onpeak)
            @constraint(model_mpc, gamma_onpeak_k == 0)
        end
        

        # Objective function
        @expression(model_mpc, demand_charge_k, r_power_nc * gamma_nc_k + r_power_onpeak * gamma_onpeak_k)

        if isempty(Index_onpeak)
            @expression(model_mpc, energy_charge_k, delta_t * sum(r_energy_offpeak * L[t] for t in Index_offpeak))
        elseif !isempty(Index_onpeak)
            @expression(model_mpc, energy_charge_k, delta_t * sum(r_energy_offpeak * L[t] for t in Index_offpeak) + 
                                            delta_t * sum(r_energy_onpeak * L[t] for t in Index_onpeak))
        end

                                            
        # TODO: DWR of the other charges is not included

        if k == 1
            @expression(model_mpc, E_dispensed, 0)
        else
            if base == "EV" 
                @expression(model_mpc, E_dispensed, sum(E_tmp[i] for i in 1:N_ev))
            elseif base == "Charger"
                @expression(model_mpc, E_dispensed, sum(E_tmp[i] for i in 1:N_charger))
            end
        end

        @expression(model_mpc, other_charge_k, 0.0578 * (demand_charge_k + energy_charge_k) + (0.0058 + 0.00058 + 0.0003) * E_dispensed + 0.0688 * DWR_charge)

        @expression(model_mpc, J_k, demand_charge_k + energy_charge_k + other_charge_k)

        @objective(model_mpc, Min, J_k)

        optimize!(model_mpc)

        L_mpc[k] = value(L[k])
        P_mpc[k] = value.(P[k, :])
        E_mpc[k] = value.(E[k, :])
        E_tmp = copy(E_mpc[k])

        # NOTE: If some are not optimal, normally the constraints are conflicting
        Optimal[k] = (termination_status(model_mpc) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED]) ? 1 : 0
        if Optimal[k] == 0
            push!(Status, Int(termination_status(model_mpc)))
            push!(iterations, k)
        end

        if base == "Charger"
            ED_plot = ED_vector
            p0 = plot(ED_plot, label="ED", title="Energy Demand Check", xticks=1:20:length(ED_plot), size = (800, 600))
        elseif base == "EV"
            N_ev_last = N_ev
            ED_plot = ED
            p0 = plot(ED_plot, label="ED", title="Energy Demand Check", xticks=1:5:length(ED_plot), size = (600, 600))
        end
        
        plot!(p0, E_tmp, label="E_tmp")
        path = "ED_check/$today_update/$base/$method/"
        if !isdir(path)
            mkpath(path)
        end
        savefig(p0, path * "$k.png")
    end
    # print("MPC optimization is done with time: ", ceil(toc - tic), " seconds\n")
    print("Optimal Found: ", sum(Optimal), " out of ", N, "\n")
    # https://github.com/jump-dev/JuMPTutorials.jl/blob/master/notebook/introduction/solvers_and_solutions.ipynb
    print("k of not optimal: ", iterations, "\n")
    print("Status of not optimal: ", Status, "\n")

    return L_mpc, P_mpc, E_mpc
end



####################################################
# Forecast is initialized daily and updated with the latest data at each time step

# Daily updates:
# V0G: Dumb charging with max power
# Perfect forecast: AT, DT, ED, Nev are known, and real data is used
# No forecast: Only do MPC with the arrival data
# Persistence forecast: AT, DT, ED, Nev are the same as the previous daily data
# Statistic forecast: AT, DT, Nev are the aggregate of M previous daily data, ED is divided by M

# Real-time updates:
# KNN forecast: AT, DT, ED are updated by combination of real data and forecast
# GMM forecast: AT, DT, ED are forecasted by GMM model
# ML forecast: AT, DT, ED are forecasted by ML model


# TODO: GMM forecast
function predict_gmm(data_input)
    # Function to compute GMM density
    function gmm_pdf(gmm, x)
        density = 0.0
        for i in 1:gmm.n
            mu = gmm.μ[i]
            sigma = sqrt(gmm.Σ[i])
            weight = gmm.w[i]
            density += weight * pdf(Normal(mu, sigma), x)
        end
        return density
    end
    # Load the GMM models from the file
    models = load("best_gmms.jld")  # Adjust file extension based on actual file type
    # Extract the individual GMM models from the loaded data
    best_gmms = models["models"]
    best_gmm_AT, best_gmm_DT, best_gmm_ED = best_gmms[1:3]
    # Prepare arrays to store predictions
    n = size(data_input, 1)
    predict_gmm_AT = zeros(n)
    predict_gmm_DT = zeros(n)
    predict_gmm_ED = zeros(n)
    # For each data point, find the most likely component and use its mean as the prediction, and then update with Ben's method
    # https://docs.google.com/presentation/d/1EMBE8Me50NhXHq-kFVkho-nn2YyRwtEi_801p6mAHo4/edit?pli=1#slide=id.g1ee332c0660_1_245
    # TODO: Not sure how to update the prediction with Ben's method
    for i in 1:n
        # Extract the data point
        x = [data_input.AT[i], data_input.DT[i], data_input.ED[i]]
        # Compute the likelihood of the data point under each GMM
        likelihood_AT = gmm_pdf(best_gmm_AT, x[1])
        likelihood_DT = gmm_pdf(best_gmm_DT, x[2])
        likelihood_ED = gmm_pdf(best_gmm_ED, x[3])
        # Find the most likely component for each feature
        component_AT = argmax([likelihood_AT])
        component_DT = argmax([likelihood_DT])
        component_ED = argmax([likelihood_ED])
        # Use the mean of the most likely component as the prediction
        predict_gmm_AT[i] = best_gmm_AT.μ[component_AT]
        predict_gmm_DT[i] = best_gmm_DT.μ[component_DT]
        predict_gmm_ED[i] = best_gmm_ED.μ[component_ED]
    end
    return predict_gmm_AT, predict_gmm_DT, predict_gmm_ED
end



####################################################
# SECTION: Forecast function
function forecast_ev(data_today::DataFrame, method::String, updated_sessions::DataFrame)
    data_forecast = copy(data_today)
    forecast_list = ["Perfect", "Persistence", "Statistic", "Noforecast"]
    if method ∉ forecast_list
        error("Invalid forecast method")
    elseif method in ["Perfect", "Noforecast"]
        # The AT, DT, ED, PD should comply with or be more precise than the step length of MPC
        data_forecast.AT = ceil.(Float64.(Dates.hour.(data_forecast.session_start_time_la)) + Float64.(Dates.minute.(data_forecast.session_start_time_la)) / 60, digits=2)
        data_forecast.DT = floor.(Float64.(Dates.hour.(data_forecast.session_end_time_la)) + Float64.(Dates.minute.(data_forecast.session_end_time_la)) / 60, digits=2)
        data_forecast.ED = data_forecast.total_energy_dispensed
        # data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)
    elseif method == "Persistence"
        # Just do the MPC with data from the previous smart day (weekday, weekend, holiday)
        target_time = data_today.session_start_time_la[1]
        target_day_type = datetype(target_time)
        same_type_sessions = updated_sessions[(updated_sessions.type .== target_day_type) .& (Date.(updated_sessions.session_start_time_la) .< Date(target_time)), :]
        if nrow(same_type_sessions) == 0
            error("No valid sessions found for the same day type")
        end
        closest_idx = argmin(abs.(Date.(same_type_sessions.session_start_time_la) .- Date(target_time)))
        closest_date = Date(same_type_sessions.session_start_time_la[closest_idx])
        persistence_sessions = same_type_sessions[Date.(same_type_sessions.session_start_time_la) .== closest_date, :]
        data_forecast = persistence_sessions
        data_forecast.AT = ceil.(Float64.(Dates.hour.(data_forecast.session_start_time_la)) + Float64.(Dates.minute.(data_forecast.session_start_time_la)) / 60, digits=2)
        data_forecast.DT = floor.(Float64.(Dates.hour.(data_forecast.session_end_time_la)) + Float64.(Dates.minute.(data_forecast.session_end_time_la)) / 60, digits=2)
        data_forecast.ED = data_forecast.total_energy_dispensed
        # data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)
    elseif method == "Statistic"
        # Pick the closest M days
        M = 3
        target_time = data_today.session_start_time_la[1]
        target_day_type = datetype(target_time)
        same_type_sessions = updated_sessions[(updated_sessions.type .== target_day_type) .& (Date.(updated_sessions.session_start_time_la) .< Date(target_time)), :]
        if nrow(same_type_sessions) == 0
            error("No valid sessions found for the same day type")
        end
        session_dates = unique(Date.(same_type_sessions.session_start_time_la))
        differences = abs.(session_dates .- Date(target_time))
        closest_idxs = sortperm(differences)[1:min(M, length(differences))]
        closest_dates = session_dates[closest_idxs]
        mask = in.(Date.(same_type_sessions.session_start_time_la), Ref(closest_dates))
        closest_sessions = same_type_sessions[mask, :]
        data_forecast = closest_sessions
        data_forecast.AT = ceil.(Float64.(Dates.hour.(data_forecast.session_start_time_la)) + Float64.(Dates.minute.(data_forecast.session_start_time_la)) / 60, digits=2)
        data_forecast.DT = floor.(Float64.(Dates.hour.(data_forecast.session_end_time_la)) + Float64.(Dates.minute.(data_forecast.session_end_time_la)) / 60, digits=2)
        data_forecast.ED = data_forecast.total_energy_dispensed / M
        # data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)
    end

    return data_forecast
end

# NOTE: for compatibility with MPC, output will be also in session not charger form
function forecast_charger(data_today::DataFrame, method::String, updated_sessions::DataFrame)
    data_forecast = copy(data_today)
    forecast_list = ["Perfect", "Noforecast", "Persistence", "LSTM", "Transformer"]
    if method ∉ forecast_list
        error("Invalid forecast method")
    elseif method in ["Perfect", "Noforecast"]
        data_forecast.AT = ceil.(Float64.(Dates.hour.(data_forecast.session_start_time_la)) + Float64.(Dates.minute.(data_forecast.session_start_time_la)) / 60, digits=2)
        data_forecast.DT = floor.(Float64.(Dates.hour.(data_forecast.session_end_time_la)) + Float64.(Dates.minute.(data_forecast.session_end_time_la)) / 60, digits=2)
        data_forecast.ED = data_forecast.total_energy_dispensed
    elseif method in ["LSTM", "Transformer"]
        # For each charger train NN model respectively
        Random.seed!(17)
        device = gpu_device()
        N_days = 30
        ED_min = 0.0
        ED_max = P_max * delta_t
        target_time = data_today.session_start_time_la[1]
        today_update = Date(target_time)
        sessions_windowed = updated_sessions[(Date.(updated_sessions.session_start_time_la) .>= Date(target_time) - Day(N_days)), :]
        sessions_windowed[!, :start_interval] = ceil.(sessions_windowed.session_start_time_la, Minute(15))
        sessions_windowed[!, :end_interval] = floor.(sessions_windowed.session_end_time_la, Minute(15))
        time_bins = DataFrame(interval=collect(DateTime(Date(minimum(sessions_windowed.start_interval))):Minute(15):DateTime(Date(maximum(sessions_windowed.end_interval)))+Hour(23)+Minute(45)))
        ref_time = minimum(time_bins.interval)
        time_bins[!, :time_encoded] .= Dates.value.(time_bins.interval .- ref_time) ./ (60*15*10^3)  # Normalize by 15-min intervals
        period = T / delta_t
        time_bins.time_sin .= sin.(2π * time_bins.time_encoded / period)
        time_bins.time_cos .= cos.(2π * time_bins.time_encoded / period)
        X = hcat(time_bins.time_sin, time_bins.time_cos)' # Only time is taken as input
        X = convert(Array{Float32}, X)
        sequence_length = N  # 1 day of data
        num_sequences = size(X, 2) ÷ sequence_length
        X_train = [X[:, (i-1)*sequence_length+1:i*sequence_length, :] for i in 1:num_sequences]
        X_train = cat(X_train..., dims=3)

        # Create the model
        Flux.reset!(model_ml)
        if method == "LSTM"
            model_ml = Flux.Chain(
                    LSTM(2 => 64),               # or Recur(LSTMCell(2, 64))
                    Dense(64 => 32, tanh),
                    Dense(32 => 2, relu)
                )
        elseif method == "Transformer"
            position_encoding = PositionEncoding(32)
            add_position_encoding(x) = x .+ position_encoding(x)
            model_ml = Flux.Chain(
                Embedding(1000 => 32), # vocab length is 1000
                add_position_encoding, # can also make anonymous
                Dropout(0.1),
                TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
                TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
                Dense(32, 1),
                FlattenLayer(),
                Dense(10, 3) # sentence length is 10, 3 labels
                )
        end

        loss(x, y) = Flux.mse(model_ml(x), y)
        combined_forecast = DataFrame(interval=DateTime[], occupancy=Float64[], ED=Float64[], station_name=String[], port=Int64[])
        tic = time()
        epochs = 50

        # Forecast today
        forecast = DataFrame(interval=collect(DateTime(today_update):Minute(15):DateTime(today_update) + Hour(23) + Minute(45)))
        forecast.time_encoded .= Dates.value.(forecast.interval .- ref_time) ./ (60*15*10^3)
        forecast.time_sin = sin.(2π * forecast.time_encoded / period)
        forecast.time_cos = cos.(2π * forecast.time_encoded / period)
        forecast_X = hcat(forecast.time_sin, forecast.time_cos)'
        forecast_X = reshape(forecast_X, size(forecast_X, 1), size(forecast_X, 2), 1)
        forecast_X = convert(Array{Float32}, forecast_X)


        @showprogress for charger in eachrow(unique_stations_ports)
            charger_idx = findfirst(==(charger), eachrow(unique_stations_ports))
            time_bins.occupancy .= 0
            time_bins.ED .= 0.0
            charger_sessions = filter(row -> row.station_name == charger.station_name && row.port == charger.port, sessions_windowed)
            if nrow(charger_sessions) != 0
                for row_idx in 1:nrow(time_bins)
                    interval = time_bins.interval[row_idx]
                    mask = (charger_sessions.start_interval .<= interval) .& (interval .<= charger_sessions.end_interval)
                    if any(mask)
                        mask_idx = findfirst(mask) # can only have 1 idx
                        time_bins.occupancy[row_idx] = 1
                        N_intervals = Dates.value(charger_sessions.end_interval[mask_idx] - charger_sessions.start_interval[mask_idx]) / (60*15*10^3)
                        # NOTE: The energy dispensed is divided by N_intervals + 1 to avoid the case of 0 and we assume at DT idx power can be dispatched
                        time_bins.ED[row_idx] = charger_sessions.total_energy_dispensed[mask_idx] / (N_intervals + 1)
                    end
                end
            else
                # print("\nNo history sessions found for $(charger.station_name) port $(charger.port)") # keep their occupancy and ED as 0
                continue;
            end
            # NN model could be accelerated by Metal.jl but not solved yet
            # dev = Metal.device()
            
            y_occupancy = time_bins.occupancy
            y_ED = (time_bins.ED .- ED_min) ./ (ED_max - ED_min)  # Normalize ED to [0,1]
            y = hcat(y_occupancy, y_ED)'
            y = convert(Array{Float32}, y)
            # Reshape (features, sequence length, batch size)
            y_train = [y[:, (i-1)*sequence_length+1:i*sequence_length, :] for i in 1:num_sequences]
            # Flatten the list of sequences into a single batch
            y_train = cat(y_train..., dims=3)
            loader = Flux.DataLoader((X_train, y_train), batchsize=15, shuffle=true)
            
            
            losses = []
            Flux.reset!(model_ml)
            opt = Flux.Adam()
            state = Flux.setup(opt, model_ml)
            state = Flux.setup(OptimiserChain(WeightDecay(0.42), Adam(0.1)), model) # with l2 regularization

            for epoch in 1:epochs
                Flux.train!(model_ml, loader, state) do m, x, y
                    Flux.mse(m(x), y)
                end
                push!(losses, loss(X_train, y_train))
            end




            # Plot the loss curve
            p_loss = plot(losses, label="Loss", title="Loss Curve", size=(800, 600))
            path = "ML/$today_update/$(unique_stations_ports.station_name[charger_idx])/port $(unique_stations_ports.port[charger_idx])/LSTM/"
            if !isdir(path)
                mkpath(path)
            end
            savefig(p_loss, path * "loss.png")

            # Forecast today's occupancy and ED of this charger
            predictions = model_ml(forecast_X) |> cpu
            # TODO: maybe ensure the occupancy is in binary with threshold?
            forecast[!, :occupancy] .= vec(predictions[1, :, 1])
            forecast[!, :normalized_ED] .= vec(predictions[2, :, 1]).* (ED_max - ED_min) .+ ED_min
            forecast[!, :station_name] .= charger.station_name
            forecast[!, :port] .= charger.port

            # Plot the forecasted occupancy and ED
            p_occupancy_ED = plot(forecast.interval, forecast.occupancy, label="Occupancy", title="Occupancy Forecast", size=(800, 600))
            plot!(forecast.interval, forecast.normalized_ED, label="ED", title="LSTM Forecast", size=(800, 600))
            savefig(p_occupancy_ED, path * "forecast.png")
            combined_forecast = vcat(combined_forecast, forecast)
        end
        toc = time()
        print("ML forecast for $today_update with $method is done in $(ceil(toc - tic)) seconds\n")
 
        # Convert to EV session form
        combined_forecast[!, :ED] .= combined_forecast.occupancy .* combined_forecast.normalized_ED
        combined_forecast = filter(row -> row.ED >= 0 && row.occupancy >= 0, combined_forecast)
        # Add required columns for MPC input
        combined_forecast[!, :driver_id] .= "dummy"  # Placeholder, replace with actual driver IDs if available
        combined_forecast.AT = ceil.(Float64.(Dates.hour.(combined_forecast.interval)) + Float64.(Dates.minute.(combined_forecast.interval)) / 60, digits=2)
        combined_forecast.DT = min.(combined_forecast.AT .+ 0.25, 23.75)
        combined_forecast[!, :type] .= datetype.(combined_forecast.interval)
        combined_forecast[!, :session_start_time_la] .= combined_forecast.interval
        combined_forecast[!, :session_end_time_la] .= combined_forecast.interval .+ Minute(15)
        combined_forecast[!, :total_energy_dispensed] .= combined_forecast.ED
        combined_forecast = select(combined_forecast, [:driver_id, :session_start_time_la, :session_end_time_la, :total_energy_dispensed, :AT, :DT, :ED, :station_name, :port, :type])

        data_forecast = copy(combined_forecast)
    end
    return data_forecast
end

# V0G for comparison
function run_V0G(data_input::DataFrame)
    L_V0G = zeros(N)
    P_V0G = zeros(N, size(data_input, 1))
    data_input.AT = ceil.(Float64.(Dates.hour.(data_input.session_start_time_la)) + Float64.(Dates.minute.(data_input.session_start_time_la)) / 60, digits=2)
    data_input.DT = floor.(Float64.(Dates.hour.(data_input.session_end_time_la)) + Float64.(Dates.minute.(data_input.session_end_time_la)) / 60, digits=2)
    data_input.ED = data_input.total_energy_dispensed

    for k in 1:N
        for i in 1:size(data_input, 1)
            Energy_dispensed = sum(P_V0G[1:k-1, i]) * delta_t
            Remaining_energy = data_input.ED[i] - Energy_dispensed
            # Check charging condition and apply adjusted final charging power if needed
            if data_input.AT[i] <= (k - 1) * delta_t <= data_input.DT[i] && Energy_dispensed < data_input.ED[i]
                # If remaining energy is less than max power * delta_t, adjust to just meet ED
                if Remaining_energy < P_max * delta_t
                    P_V0G[k, i] = Remaining_energy / delta_t
                else
                    P_V0G[k, i] = P_max
                end
            else
                P_V0G[k, i] = 0
            end
        end
    end
    select!(data_input, Not([:AT, :DT, :ED]))

    L_V0G = sum(P_V0G, dims=2)
    return L_V0G
end

# Daily update main function
function daily_update(data_input::DataFrame, method::String, base::String)
    # days = unique(Date.(data_input.session_start_time_la))
    updated_sessions = copy(charging_sessions)
    updated_sessions.type = datetype.(updated_sessions.session_start_time_la)
    L_mpc_dict = Dict{Date, Any}()
    P_mpc_dict = Dict{Date, Any}()
    E_mpc_dict = Dict{Date, Any}()
    L_V0G_dict = Dict{Date, Any}()

    println("Daily update with $method\n")
    tic = time()
    for today_update in days
        data_today = filter(row -> Dates.Date(row.session_start_time_la) == today_update, data_input)
        L_V0G_dict[today_update] = run_V0G(data_today)
        data_today.type = datetype.(data_today.session_start_time_la)
        tmp_sessions = vcat(updated_sessions, data_today)
        if base == "EV"
            data_forecast = forecast_ev(data_today, method, updated_sessions)
        elseif base == "Charger"
            data_forecast = forecast_charger(data_today, method, updated_sessions)
        end
        
        L_mpc_tmp, P_mpc_tmp, E_mpc_tmp = run_mpc(data_forecast, data_today, method, base);
        updated_sessions = copy(tmp_sessions)
        L_mpc_dict[today_update] = L_mpc_tmp
        P_mpc_dict[today_update] = P_mpc_tmp
        E_mpc_dict[today_update] = E_mpc_tmp
    end
    toc = time()
    println("Daily update with $method in $(ceil(toc - tic)) seconds\n")
    
    return L_V0G_dict, L_mpc_dict, P_mpc_dict, E_mpc_dict
end

####################################################
# SECTION: Testing
# https://github.com/rdeits/DynamicWalking2018.jl/blob/master/notebooks/6.%20Optimization%20with%20JuMP.ipynb
# Test data is unknown with training data known
test_sessions = CSV.read("test_charging_sessions.csv", DataFrame)
test_sessions = sort(test_sessions, [:session_start_time_la])
start_date = Date(minimum(test_sessions.session_start_time_la))
end_date = start_date + Day(3)
data_test_candidate = filter(row -> start_date <= row.session_start_time_la <= end_date, test_sessions)
data_test_candidate = sort(data_test_candidate, [:session_start_time_la])

# NOTE: Since we shorten the time period for charging, some real ED > P_max * delta_t * (DT_idx - AT_idx)
data_test_candidate.AT = ceil.(Float64.(Dates.hour.(data_test_candidate.session_start_time_la)) + Float64.(Dates.minute.(data_test_candidate.session_start_time_la)) / 60, digits=2)
data_test_candidate.DT = floor.(Float64.(Dates.hour.(data_test_candidate.session_end_time_la)) + Float64.(Dates.minute.(data_test_candidate.session_end_time_la)) / 60, digits=2)
data_test_candidate.AT_idx = ceil.(Int, data_test_candidate.AT / T * N) .+ 1
data_test_candidate.DT_idx = floor.(Int, data_test_candidate.DT / T * N) .+ 1
data_test_candidate.ED = data_test_candidate.total_energy_dispensed
data_test_candidate.overMaxPower = data_test_candidate.ED .> P_max * delta_t * (data_test_candidate.DT_idx - data_test_candidate.AT_idx .+ 1)
data_test = data_test_candidate[data_test_candidate.overMaxPower .== false, :]
data_test = data_test[:, 1:6]
days = unique(Date.(data_test.session_start_time_la))





# VERSION: Pick the method and base from below for testing
method_list_EV = ["Perfect", "Noforecast", "Persistence", "Statistic"]
method_list_Charger = ["Perfect", "Noforecast", "LSTM", "Transformer"]
base_list = ["EV", "Charger"]
data_input = copy(data_test)
method = "Perfect"
base = "Charger"

# EV testing
L_V0G, L_mpc_perfect, P_mpc_perfect, E_mpc_perfect = daily_update(data_test, "Perfect", "EV");
L_V0G, L_mpc_noforecast, P_mpc_noforecast, E_mpc_noforecast = daily_update(data_test, "Noforecast", "EV");
L_V0G, L_mpc_persistence, P_mpc_persistence, E_mpc_persistence = daily_update(data_test, "Persistence", "EV");
L_V0G, L_mpc_statistic, P_mpc_statistic, E_mpc_statistic = daily_update(data_test, "Statistic", "EV");

# Charger testing
L_V0G, L_mpc_perfect, P_mpc_perfect, E_mpc_perfect = daily_update(data_test, "Perfect", "Charger");
L_V0G, L_mpc_noforecast, P_mpc_noforecast, E_mpc_noforecast = daily_update(data_test, "Noforecast", "Charger");
L_V0G, L_mpc_lstm, P_mpc_lstm, E_mpc_lstm = daily_update(data_test, "LSTM", "Charger");
L_V0G, L_mpc_transformer, P_mpc_transformer, E_mpc_transformer = daily_update(data_test, "Transformer", "Charger");



####################################################
# SECTION: Results

# Load plot
function load_plot()
    all_intervals = DataFrame(interval = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)])
    floored_AT = Dates.floor.(data_test.session_start_time_la, Minute(15))
    floored_DT = Dates.floor.(data_test.session_end_time_la, Minute(15))
    counts_AT = combine(groupby(DataFrame(interval = floored_AT), :interval), nrow => :count)
    counts_DT = combine(groupby(DataFrame(interval = floored_DT), :interval), nrow => :count)
    df_AT = leftjoin(all_intervals, counts_AT, on=:interval)
    df_AT[!, :count] .= coalesce.(df_AT[!, :count], 0)
    df_DT = leftjoin(all_intervals, counts_DT, on=:interval)
    df_DT[!, :count] .= coalesce.(df_DT[!, :count], 0)

    df_all_loads = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], V0G = vec(vcat(values(L_V0G)...)), Perfect = vec(vcat(values(L_mpc_perfect)...)), Noforecast = vec(vcat(values(L_mpc_noforecast)...)), Persistence = vec(vcat(values(L_mpc_persistence)...)), Statistic = vec(vcat(values(L_mpc_statistic)...)), LSTM = vec(vcat(values(L_mpc_lstm)...)))

    xlim = extrema(vcat(df_AT.interval, df_DT.interval, df_all_loads.time))
    hist_AT = bar(df_AT.interval, df_AT.count, xlabel="AT", ylabel="Sessions", label="AT", color=:lightblue, alpha=0.7, size=(1400, 600), dpi=300, xlims=xlim)
    hist_DT = bar(df_DT.interval, df_DT.count, xlabel="DT", ylabel="Sessions", label="DT", color=:lightblue, alpha=0.7, size=(1400, 600), dpi=300, xlims=xlim)

    p_load_combined = plot(df_all_loads.time, df_all_loads.V0G,
        label="V0G",  # Add cost to label
        xlabel="Timestamp",
        ylabel="Load (kW)",
        size=(1400, 600),
        dpi=300,
        legend=:topright,
        legendfontsize=10,
        xlims=xlim)
    plot!(df_all_loads.time, df_all_loads.Perfect, label="Perfect Forecast")
    plot!(df_all_loads.time, df_all_loads.Noforecast, label="No Forecast")
    plot!(df_all_loads.time, df_all_loads.Persistence, label="Persistence Forecast")
    plot!(df_all_loads.time, df_all_loads.Statistic, label="Statistic Forecast")
    plot!(df_all_loads.time, df_all_loads.LSTM, label="LSTM Forecast")

    p = plot(hist_AT, hist_DT, p_load_combined, layout=grid(3, 1, heights=[0.2 ,0.2, 0.6]), size=(3000, 1000))
    return p
end

p_load = load_plot()
savefig(p_load, "Load_All.png")



# Energy checking
function daily_energy()
    Energy_real = Dict{Date, Float64}()
    Energy_V0G = Dict{Date, Float64}()
    Energy_perfect = Dict{Date, Float64}()
    Energy_noforecast = Dict{Date, Float64}()
    Energy_persistence = Dict{Date, Float64}()
    Energy_statistic = Dict{Date, Float64}()
    Energy_lstm = Dict{Date, Float64}()
    for today_update in days
        data_today = data_test[Date.(data_test.session_start_time_la) .== today_update, :]
        Energy_real[today_update] = sum(data_today.total_energy_dispensed)
        Energy_V0G[today_update] = sum(L_V0G[today_update]) * delta_t
        Energy_perfect[today_update] = sum(L_mpc_perfect[today_update]) * delta_t
        Energy_noforecast[today_update] = sum(L_mpc_noforecast[today_update]) * delta_t
        Energy_persistence[today_update] = sum(L_mpc_persistence[today_update]) * delta_t
        Energy_statistic[today_update] = sum(L_mpc_statistic[today_update]) * delta_t
        Energy_lstm[today_update] = sum(L_mpc_lstm[today_update]) * delta_t
    end
 
    df_energy_all = DataFrame(
    day = days,
    Real = [Energy_real[day] for day in days],
    V0G = [Energy_V0G[day] for day in days],
    Perfect = [Energy_perfect[day] for day in days],
    Noforecast = [Energy_noforecast[day] for day in days],
    Persistence = [Energy_persistence[day] for day in days],
    Statistic = [Energy_statistic[day] for day in days],
    LSTM = [Energy_lstm[day] for day in days]
    )
    return df_energy_all
end

df_energy_all = daily_energy()
CSV.write("Energy_All.csv", df_energy_all)


# Cost calculation
function daily_cost(Load_input)
    Cost_dict = Dict{Date, Any}()
    for i in eachindex(days)
        today_update = days[i]
        cost = 0.0
        season = get_season(DateTime(today_update))
        if season == "summer"
            r_energy_onpeak = r_energy_summer_onpeak
            r_energy_offpeak = r_energy_summer_offpeak
            r_power_onpeak = r_power_summer_onpeak
        elseif season == "winter"
            r_energy_onpeak = r_energy_winter_onpeak
            r_energy_offpeak = r_energy_winter_offpeak
            r_power_onpeak = r_power_winter_onpeak
        else
            r_energy_onpeak = r_energy_midseason_onpeak
            r_energy_offpeak = r_energy_midseason_offpeak
            r_power_onpeak = r_power_midseason_onpeak
        end

        L_daily = Load_input[today_update]
        Index_onpeak = N_start_idx:N_end_idx
        Index_offpeak = vcat(1:N_start_idx-1, N_end_idx+1:N)
        gamma_nc = maximum(L_daily[t] for t in 1:N)
        gamma_onpeak = maximum(L_daily[t] for t in Index_onpeak)
        demand_charge = r_power_nc * gamma_nc + r_power_onpeak * gamma_onpeak
        energy_charge = delta_t * sum(r_energy_offpeak * L_daily[t] for t in Index_offpeak) + 
                                            delta_t * sum(r_energy_onpeak * L_daily[t] for t in Index_onpeak)
        # other_charge = 0.0578 * (demand_charge + energy_charge) + (0.0058 + 0.00058 + 0.0003) * df_energy_all.Real[i] + 0.0688 * DWR_charge
        cost = demand_charge + energy_charge
        Cost_dict[today_update] = cost
    end
    return Cost_dict
end

Cost_V0G = daily_cost(L_V0G)
Cost_perfect = daily_cost(L_mpc_perfect)
Cost_noforecast = daily_cost(L_mpc_noforecast)
Cost_persistence = daily_cost(L_mpc_persistence)
Cost_statistic = daily_cost(L_mpc_statistic)
Cost_lstm = daily_cost(L_mpc_lstm)

df_cost = DataFrame(day = days,
                    V0G = collect(values(Cost_V0G)),
                    Perfect = collect(values(Cost_perfect)),
                    Noforecast = collect(values(Cost_noforecast)),
                    Persistence = collect(values(Cost_persistence)),
                    Statistic = collect(values(Cost_statistic)),
                    LSTM = collect(values(Cost_lstm)))

p_cost = groupedbar(df_cost.day, [df_cost.V0G df_cost.Perfect df_cost.Noforecast df_cost.Persistence df_cost.Statistic df_cost.LSTM], 
                    label=["V0G" "Perfect" "Noforecast" "Persistence" "Statistic" "LSTM"], xlabel="Day", ylabel="Cost", 
                    bar_width=0.7, size=(800, 600), dpi=300, legend=:top, legendfontsize=5,
                    title="Cost Comparison")

savefig(p_cost, "Cost_All.png")
CSV.write("Cost_All.csv", df_cost)