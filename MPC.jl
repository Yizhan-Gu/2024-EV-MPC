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
packages = ["JuMP", "Ipopt", "LinearAlgebra", "Plots", "Random", "CSV", "DataFrames", "Statistics", "StatsBase", "ProgressMeter", "Dates", "DataFramesMeta", "Distributions", "JLD", "GaussianMixtures", "Holidays", "AutoMLPipeline", "StatsPlots", "IJulia", "Gurobi", "Flux", "MLJ", "TimeSeries", "Metal", "Optimisers", "OrderedCollections"]

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
    data_today.AT = ceil.(Float64.(Dates.hour.(data_today.session_start_time_la)) + Float64.(Dates.minute.(data_today.session_start_time_la)) / 60, digits=2)
    today_update = Date.(data_today.session_start_time_la[1])

    # Define rates based on season
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
        model_mpc = JuMP.Model(() -> Gurobi.Optimizer(GUROBI_ENV)) 
        set_optimizer_attribute(model_mpc, "OutputFlag", 0)  # Suppress solver output
        set_optimizer_attribute(model_mpc, "Threads", 6)   # Use multiple threads
        #=
        set_optimizer_attribute(model, "max_iter", 2000)  # Set maximum number of iterations
        set_optimizer_attribute(model, "tol", 1e-4)       # Set tolerance for convergence
        set_optimizer_attribute(model, "acceptable_tol", 1e-3)  # Set acceptable tolerance
        set_optimizer_attribute(model, "print_level", 0)  # Set printing level (0: no output, 5: full output)
        set_optimizer_attribute(model, "constr_viol_tol", 1e-3)  # Set constraint violation tolerance
        set_optimizer_attribute(model, "warm_start_init_point", "yes")
        =#

        data_forecast_update = DataFrame()

        if base == "EV"
            # SECTION: Preprocessing
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
                elseif method in ["Persistence", "Statistic"]
                    if isempty(arrived_sessions_today)
                        data_forecast_update = copy(data_forecast[1:2, :])
                        # postpone time so that unarrived EVs are not charged -- lead to index out of bounds and AT > DT
                        # data_forecast_update.AT .= min.(data_forecast_update.AT .+ 1, 23.99)
                        # data_forecast_update.DT .= min.(data_forecast_update.DT .+ 1, 23.99)
                        data_forecast_update.ED .= 0
                    elseif !isempty(arrived_sessions_today)
                        # NOTE: Assume all real AT, DT, ED, PD are known after arrival of EVs to make sure the area of load plot is the same as the real data
                        arrived_sessions_today.DT = floor.(Float64.(Dates.hour.(arrived_sessions_today.session_end_time_la)) + Float64.(Dates.minute.(arrived_sessions_today.session_end_time_la)) / 60, digits=2)
                        arrived_sessions_today.ED = arrived_sessions_today.total_energy_dispensed

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
                        end
                        data_forecast_update = vcat(arrived_sessions_today, noarrived_sessions_today)
                    end
                end
            end

            if size(data_forecast_update, 1) == 0
                error("Wrong preprocessing before MPC")
            end

            # SECTION: MPC optimization
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
                E_tmp = zeros(N_ev)
            elseif k >= 2
                if flag == "more"
                    E_tmp = vcat(E_tmp, zeros(N_ev - N_ev_last))
                elseif flag == "fewer"
                    E_tmp = E_tmp[1:N_ev]
                end
            end
            @constraint(model_mpc, [i=1:N_ev], E[k, i] == E_tmp[i] + P[k, i] * delta_t) # E_tmp is the energy state vector stored at the previous loop of k
            if k <= N - 1  
                @constraint(model_mpc, [t=k+1:N, i=1:N_ev], E[t, i] == E[t-1, i] + P[t, i] * delta_t)
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
            @variables model_mpc begin
                P[k:N, 1:N_charger] >= 0
                L[k:N] >= 0
                E[k:N, 1:N_charger] >= 0
                gamma_nc_k >= 0
                gamma_onpeak_k >= 0
            end

            # SECTION: Preprocessing
            if method == "Perfect"
                data_forecast_update = copy(data_forecast)
            else
                update_time = (k - 1) * delta_t
                arrived_sessions_today = filter(row -> row.AT <= update_time, data_today)
                arrived_sessions_today.DT = floor.(Float64.(Dates.hour.(arrived_sessions_today.session_end_time_la)) + Float64.(Dates.minute.(arrived_sessions_today.session_end_time_la)) / 60, digits=2)
                arrived_sessions_today.ED = arrived_sessions_today.total_energy_dispensed
                arrived_charger_today = ev_to_charger(arrived_sessions_today)

                # check if ED > 0 but sum of oc_matrix = 0
                ed_charger = arrived_charger_today.ED
                oc_charger = vec(sum(reduce(hcat, arrived_charger_today.oc_matrix)', dims=2))
                error_idx = findall(i -> ed_charger[i] > 0 && oc_charger[i] == 0, 1:length(ed_charger))
                if !isempty(error_idx)
                    println("error charger indices: ", error_idx, "\n")
                    error("ED > 0 but oc_matrix = 0")
                end
                
                if method == "Noforecast"
                    data_forecast_update = arrived_charger_today
                elseif method in ["Persistence", "LSTM", "Transformer"]
                    # Need to update the data_forecast_update
                    data_forecast_update = copy(data_forecast)
                    if size(data_forecast_update, 1) != N_charger
                        error("The number of chargers in the forecast data is not equal to the number of chargers in the arrived data")
                    end
                    charger_ed = zeros(N_charger)
                    valid_occupied_idx = [Int[] for _ in 1:N_charger]
                    valid_vacant_idx = [Int[] for _ in 1:N_charger]
                    modified_forecast_ed = zeros(N_charger)

                    for i_ch in 1:N_charger
                        oc_arrived = arrived_charger_today.oc_matrix[i_ch]
                        oc_forecast = data_forecast_update.oc_matrix[i_ch]
                        # Find the last occupied timestep in the arrived data
                        last_occupied_idx = findlast(==(1), oc_arrived)
                        last_occupied_idx = last_occupied_idx === nothing ? 0 : last_occupied_idx                        
                        combined_oc = collect(vcat(oc_arrived[1:last_occupied_idx], oc_forecast[last_occupied_idx+1:end]))
                        busy_idx = findall(x -> x == 1, combined_oc)
                        # NOTE: 2 cases: 1) last_occupied_idx >= k, 2) last_occupied_idx < k
                        if sum(oc_forecast) == 0 || isempty(oc_forecast[max(k, last_occupied_idx)+1:end])
                            valid_occupied_idx[i_ch] = collect(intersect(busy_idx, k:N))
                            modified_forecast_ed[i_ch] = 0
                        elseif k <= last_occupied_idx
                            valid_occupied_idx[i_ch] = collect(intersect(busy_idx, k:N))
                            modified_forecast_ed[i_ch] = data_forecast_update.ED[i_ch] * sum(oc_forecast[last_occupied_idx+1:end]) / sum(oc_forecast)
                            @constraint(model_mpc, E[last_occupied_idx, i_ch] == arrived_charger_today.ED[i_ch]) # To make sure only arrived ED is charged
                        elseif k > last_occupied_idx # including no arrived EVs that last_occupied_idx = 0
                            valid_occupied_idx[i_ch] = collect(intersect(busy_idx, k+1:N))
                            modified_forecast_ed[i_ch] = data_forecast_update.ED[i_ch] * sum(oc_forecast[k+1:end]) / sum(oc_forecast)
                        end

                        valid_vacant_idx[i_ch] = collect(setdiff(k:N, valid_occupied_idx[i_ch]))
                        charger_ed[i_ch] = arrived_charger_today.ED[i_ch] + modified_forecast_ed[i_ch]
                    end
                end
            end

            if k == 1
                E_tmp = zeros(N_charger)
            end

            if method in ["Perfect", "Noforecast"]
                ED = data_forecast_update.ED
                valid_occupied_idx = [Int[] for _ in 1:N_charger]
                valid_vacant_idx = [Int[] for _ in 1:N_charger]
                for i_ch in 1:N_charger
                    busy_idx = findall(x -> x == 1, data_forecast_update.oc_matrix[i_ch])
                    valid_occupied_idx[i_ch] = collect(intersect(busy_idx, k:N))
                    valid_vacant_idx[i_ch] = collect(setdiff(k:N, valid_occupied_idx[i_ch]))
                end
            else
                ED = charger_ed
            end
            ED_max = zeros(N_charger)

            # SECTION: MPC optimization
            # Constraints
            for i in 1:N_charger
                @constraint(model_mpc, E[k, i] == E_tmp[i] + P[k, i] * delta_t)
                if k <= N - 1  
                    @constraint(model_mpc, [t=k+1:N], E[t, i] == E[t-1, i] + P[t, i] * delta_t)
                end

                # Check ED valivity
                ED_max[i] = E_tmp[i] + P_max * length(valid_occupied_idx[i]) * delta_t
                if ED[i] > ED_max[i] + 1e-4 # Computing small error but does not affect solving MPC
                    print("ED is too large ", "forecast ED:", ED[i], "ED_max: ", ED_max[i], "arrived ED:", arrived_charger_today.ED[i], "\n")
                    ED[i] = ED_max[i]
                end
                
                if E_tmp[i] > ED[i] + 1e-4
                    print("ED is too small", "ED", ED[i], "E_tmp", E_tmp[i], "\n")
                end

                @constraint(model_mpc, [t in valid_occupied_idx[i]], 0 <= P[t, i] <= P_max)
                @constraint(model_mpc, [t in valid_vacant_idx[i]], P[t, i] == 0)
                # @constraint(model_mpc, [t in valid_occupied_idx], 0 <= E[t, i] <= ED[i])
                
                if !isempty(valid_occupied_idx[i])
                    # @constraint(model_mpc, E_tmp[i] + sum(P[t, i] * delta_t for t in valid_occupied_idx) == ED[i]) # TODO: discuss with Adil
                    @constraint(model_mpc, E[valid_occupied_idx[i][end], i] == ED[i])
                end

            end
            @constraint(model_mpc, [t=k:N], L[t] == sum(P[t, i] for i in 1:N_charger))

        end # End of base EV or Charger

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

        # FIXME: the gamma should be maxof the daily previous values and the current value
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
        E_tmp = value.(Array(E[k, :]))

        # If some are not optimal, normally the constraints are conflicting
        Optimal[k] = (termination_status(model_mpc) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED]) ? 1 : 0
        if Optimal[k] == 0
            push!(Status, Int(termination_status(model_mpc)))
            push!(iterations, k)
        end

        if base == "Charger"
            p0 = plot(ED, label="ED", title="Energy Demand Check", xticks=1:20:length(ED), size = (800, 600))
        elseif base == "EV"
            N_ev_last = N_ev
            p0 = plot(ED, label="ED", title="Energy Demand Check", xticks=1:5:length(ED), size = (600, 600))
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

    return L_mpc
end



####################################################
# Forecast is initialized daily and updated with the latest data at each time steps

# V0G: Dumb charging with max power
# Perfect forecast: AT, DT, ED, Nev are known, and real data is used
# No forecast: Only do MPC with the arrival data
# Persistence forecast: AT, DT, ED, Nev are the same as the previous daily data
# Statistic forecast: AT, DT, Nev are the aggregate of M previous daily data, ED is divided by M
# ML forecast: AT, DT, ED are forecasted by ML (NN) model


# TODO: GMM forecast
function predict_gmm(data_input)
    print("Predicting GMM...\n")
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
    end

    # NOTE: Since we shorten the time period for charging, some real ED > P_max * delta_t * (DT_idx - AT_idx)
    data_forecast.AT_idx = ceil.(Int, data_forecast.AT / T * N) .+ 1
    data_forecast.DT_idx = floor.(Int, data_forecast.DT / T * N) .+ 1
    data_forecast.overMaxPower = data_forecast.ED .> P_max * delta_t * (data_forecast.DT_idx - data_forecast.AT_idx .+ 1)
    data_forecast = data_forecast[data_forecast.overMaxPower .== false, :]
    select!(data_forecast, Not(:AT_idx, :DT_idx, :overMaxPower))

    return data_forecast
end

function ev_to_charger(data_forecast::DataFrame)
    time_bins = 0.0:delta_t:delta_t*(N-1)
    bin_count = length(time_bins)
    n_chargers = size(unique_stations_ports, 1)
    oc_matrix = zeros(n_chargers, bin_count)
    ED_vector = zeros(Float64, n_chargers)
    charger_idx = Dict((row.station_name, row.port) => i for (i, row) in enumerate(eachrow(unique_stations_ports)))
    
    for row in eachrow(data_forecast)
        key = (row.station_name, row.port)
        haskey(charger_idx, key) || continue
        idx = charger_idx[key]
        start_bin = findfirst(t -> t ≥ row.AT, time_bins)
        end_bin   = findlast(t -> t ≤ row.DT, time_bins)
        if isnothing(start_bin) || isnothing(end_bin) || start_bin > end_bin
            continue
        end
        # NOTE: If many sessions of the same charger, no overlap should happen
        ED_vector[idx] += row.ED
        oc_matrix[idx, start_bin:end_bin] .= 1
    end

    if any(oc_matrix .> 1)
        error("Overlapping sessions detected in the occupancy matrix.\n")
    end

    df = DataFrame(
        station_name = unique_stations_ports.station_name,
        port = unique_stations_ports.port,
        ED = ED_vector,
        oc_matrix = [oc_matrix[i, :] for i in 1:n_chargers]
    )

    return df
end



function forecast_charger(data_today::DataFrame, method::String, updated_sessions::DataFrame)
    forecast_list = ["Perfect", "Noforecast", "Persistence", "LSTM", "Transformer"]
    if method ∉ forecast_list
        error("Invalid forecast method")
    elseif method in ["Perfect", "Noforecast", "Persistence"]
        data_forecast = forecast_ev(data_today, method, updated_sessions)
        data_forecast = ev_to_charger(data_forecast)
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

        # Forecast today
        forecast = DataFrame(interval=collect(DateTime(today_update):Minute(15):DateTime(today_update) + Hour(23) + Minute(45)))
        forecast.time_encoded .= Dates.value.(forecast.interval .- ref_time) ./ (60*15*10^3)
        forecast.time_sin = sin.(2π * forecast.time_encoded / period)
        forecast.time_cos = cos.(2π * forecast.time_encoded / period)
        forecast_X = hcat(forecast.time_sin, forecast.time_cos)'
        forecast_X = reshape(forecast_X, size(forecast_X, 1), size(forecast_X, 2), 1)
        forecast_X = convert(Array{Float32}, forecast_X)

        combined_forecast = DataFrame(station_name = String[], port = Int[], AT = Float64[], DT = Float64[], ED = Float64[])
        tic = time()
        epochs = 30

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
                print("\nNo history sessions found for $(charger.station_name) port $(charger.port)") # keep their occupancy and ED as 0
                # continue
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

            # Create the model
            if method == "LSTM"
                model_ml = Flux.Chain(
                        LSTM(2 => 64),               # or Recur(LSTMCell(2, 64))
                        Dense(64 => 32, tanh),
                        Dense(32 => 2, relu)
                    )
            elseif method == "Transformer"
                # FIXME: not working
                position_encoding = PositionEncoding(32)
                add_position_encoding(x) = x .+ position_encoding(x)
                model_ml = Flux.Chain(
                    Embedding(2 => 32), # vocab length is 1000
                    add_position_encoding, # can also make anonymous
                    Dropout(0.1),
                    TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
                    TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
                    Dense(32 => 2)
                    )
            end
            
            losses = []
            # Flux.reset!(model_ml)
            opt = Flux.Adam()
            state = Flux.setup(opt, model_ml)
            # FIXME: state = Flux.setup(OptimiserChain(WeightDecay(0.42), Adam(0.1)), model) # with l2 regularization

            for epoch in 1:epochs
                Flux.train!(model_ml, loader, state) do m, x, y
                    Flux.Losses.mse(m(x), y)
                end
                push!(losses, Flux.Losses.mse(X_train, y_train))
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
            forecast[!, :occupancy] .= vec(predictions[1, :, 1])
            forecast[!, :real_ED] .= vec(predictions[2, :, 1]).* (ED_max - ED_min) .+ ED_min
            forecast[!, :station_name] .= charger.station_name
            forecast[!, :port] .= charger.port

            # Plot the forecasted occupancy and ED
            p_occupancy_ED = plot(forecast.interval, forecast.occupancy, label="Occupancy", title="Occupancy Forecast", size=(800, 600))
            plot!(forecast.interval, forecast.real_ED, label="ED", title="LSTM Forecast", size=(800, 600))
            savefig(p_occupancy_ED, path * "forecast.png")

            # NOTE: assume occupied when occupancy > 0.1
            idx_occupied = findall(row -> row.occupancy > 1e-1, eachrow(forecast))
            if isempty(idx_occupied)
                forecast_station = DataFrame(station_name = charger.station_name, port = charger.port, AT = 23.75, DT = 23.75, ED = 0.0)
            else
                forecast_station = DataFrame(station_name = charger.station_name, port = charger.port, AT = first(idx_occupied) / N * T, DT = last(idx_occupied) / N * T, ED = sum(forecast[!, :real_ED]))
            end
            combined_forecast = vcat(combined_forecast, forecast_station)
        end
        toc = time()
        print("ML forecast for $today_update with $method is done in $(ceil(toc - tic)) seconds\n")

        if size(combined_forecast, 1) != size(unique_stations_ports, 1)
            error("Charger size mismatch")
        end

        data_forecast = ev_to_charger(combined_forecast)
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
    L_V0G_dict = Dict{Date, Any}()
    Forecast_dict = Dict{Date, Any}()

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
        
        L_mpc_tmp = run_mpc(data_forecast, data_today, method, base);
        updated_sessions = copy(tmp_sessions)
        L_mpc_dict[today_update] = L_mpc_tmp
        Forecast_dict[today_update] = data_forecast
    end
    toc = time()
    println("Daily update with $method in $(ceil(toc - tic)) seconds\n")
    
    return L_V0G_dict, L_mpc_dict, Forecast_dict
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


# Pick the method and base from below for testing
method_list_EV = ["Perfect", "Noforecast", "Persistence", "Statistic"]
method_list_Charger = ["Perfect", "Noforecast", "Persistence", "LSTM", "Transformer"]
base_list = ["EV", "Charger"]
data_input = copy(data_test)
method = "Persistence"
base = "Charger"

# EV testing
L_V0G, L_ev_perfect, forecast_ev_perfect = daily_update(data_test, "Perfect", "EV");
L_V0G, L_ev_noforecast, forecast_ev_noforecast = daily_update(data_test, "Noforecast", "EV");
L_V0G, L_ev_persistence, forecast_ev_persistence = daily_update(data_test, "Persistence", "EV");
L_V0G, L_ev_statistic, forecast_ev_statistic = daily_update(data_test, "Statistic", "EV");

# Charger testing
L_V0G, L_charger_perfect, forecast_charger_perfect = daily_update(data_test, "Perfect", "Charger");
L_V0G, L_charger_noforecast, forecast_charger_noforecast = daily_update(data_test, "Noforecast", "Charger");
L_V0G, L_charger_persistence, forecast_charger_persistence = daily_update(data_test, "Persistence", "Charger");
L_V0G, L_charger_lstm, forecast_charger_lstm = daily_update(data_test, "LSTM", "Charger");
L_V0G, L_charger_transformer, forecast_charger_transformer = daily_update(data_test, "Transformer", "Charger");



####################################################
# SECTION: Results

L_plot = OrderedDict{String, Any}([
    "V0G" => L_V0G,
    "Perfect_ev" => L_ev_perfect,
    "Noforecast_ev" => L_ev_noforecast,
    "Persistence_ev" => L_ev_persistence,
    "Statistic_ev" => L_ev_statistic,
    "Perfect_charger" => L_charger_perfect,
    "Noforecast_charger" => L_charger_noforecast,
    "Persistence_charger" => L_charger_persistence,
    "LSTM_charger" => L_charger_lstm
    # "Transformer_charger" => L_charger_transformer
])

# Save the load results
folder = "Load_mpc"
isdir(folder) || mkdir(folder)

for (name, date_dict) in L_plot
    dates = sort(collect(keys(date_dict)))
    n_rows = length(dates)
    n_cols = length(date_dict[dates[1]])
    data = Matrix{Float64}(undef, n_rows, n_cols)
    row_labels = String[]
    for (i, d) in enumerate(dates)
        data[i, :] .= round.(date_dict[d], digits=4)
        push!(row_labels, string(d))
    end
    df = DataFrame(data, :auto)
    df = hcat(DataFrame(Date=row_labels), df)
    CSV.write(joinpath(folder, "Load_$name.csv"), df)
end

Forecasts = OrderedDict{String, Any}([
    "Perfect_ev" => forecast_ev_perfect,
    "Noforecast_ev" => forecast_ev_noforecast,
    "Persistence_ev" => forecast_ev_persistence,
    "Statistic_ev" => forecast_ev_statistic,
    "Perfect_charger" => forecast_charger_perfect,
    "Noforecast_charger" => forecast_charger_noforecast,
    "Persistence_charger" => forecast_charger_persistence,
    "LSTM_charger" => forecast_charger_lstm
    # "Transformer_charger" => forecast_charger_transformer
])

# Save the forecast results
folder_forecast = "Forecasts"
isdir(folder_forecast) || mkdir(folder_forecast)

for (name, date_dict) in Forecasts
    for (d, df) in date_dict
        filename = joinpath(folder_forecast, "Forecast_$(name)_$(d).csv")
        CSV.write(filename, df)
    end
end



# Load plot
function load_plot(L_plot; methods_to_show::Vector{String}, show_hist::Bool=true)
    all_intervals = DataFrame(interval = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)])
    floored_AT = Dates.floor.(data_test.session_start_time_la, Minute(15))
    floored_DT = Dates.floor.(data_test.session_end_time_la, Minute(15))
    counts_AT = combine(groupby(DataFrame(interval = floored_AT), :interval), nrow => :count)
    counts_DT = combine(groupby(DataFrame(interval = floored_DT), :interval), nrow => :count)
    df_AT = leftjoin(all_intervals, counts_AT, on=:interval)
    df_AT[!, :count] .= coalesce.(df_AT[!, :count], 0)
    df_DT = leftjoin(all_intervals, counts_DT, on=:interval)
    df_DT[!, :count] .= coalesce.(df_DT[!, :count], 0)

    df_all_loads = DataFrame(
        time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)]
    )
    for method in methods_to_show
        df_all_loads[!, method] = vec(vcat([L_plot[method][d] for d in sort(collect(keys(L_plot[method]))) ]...))
    end

    xlim = extrema(vcat(df_AT.interval, df_DT.interval, df_all_loads.time))
    if show_hist
        hist_AT = bar(df_AT.interval, df_AT.count, xlabel="AT", ylabel="Sessions", label="AT", color=:lightblue, alpha=0.7, size=(1400, 600), dpi=300, xlims=xlim)
        hist_DT = bar(df_DT.interval, df_DT.count, xlabel="DT", ylabel="Sessions", label="DT", color=:lightblue, alpha=0.7, size=(1400, 600), dpi=300, xlims=xlim)
    end

    p_load_combined = plot(df_all_loads.time, df_all_loads[!, methods_to_show[1]],
        label=methods_to_show[1],
        xlabel="Timestamp",
        ylabel="Load (kW)",
        size=(3000, 600),
        dpi=300,
        legend=:topright,
        legendfontsize=10,
        xlims=xlim)
    for method in methods_to_show[2:end]
        plot!(df_all_loads.time, df_all_loads[!, method], label=method)
    end

    if show_hist
        return plot(hist_AT, hist_DT, p_load_combined, layout=grid(3, 1, heights=[0.2 ,0.2, 0.6]), size=(3000, 1000))
    else
        return p_load_combined
    end
end

p_load_ev = load_plot(L_plot, methods_to_show=["V0G", "Perfect_ev", "Noforecast_ev", "Persistence_ev", "Statistic_ev"], show_hist = true)
p_load_charger = load_plot(L_plot, methods_to_show=["V0G", "Perfect_charger", "Noforecast_charger", "Persistence_charger", "LSTM_charger"], show_hist = true)
p_load = plot(p_load_ev, p_load_charger, layout=grid(2, 1), size=(3000, 2000))
savefig(p_load, "Load_All.png")



# Energy checking
function daily_energy(L_plot)
    Energy_dict = Dict{String, Dict{Date, Float64}}()
    methods = keys(L_plot)
    
    for method in methods
        Energy_dict[method] = Dict{Date, Float64}()
        for today_update in days
            Energy_dict[method][today_update] = sum(L_plot[method][today_update]) * delta_t
        end
    end

    df_energy_all = DataFrame(day = days)
    for method in methods
        df_energy_all[!, method] = [Energy_dict[method][day] for day in days]
    end

    return df_energy_all
end

df_energy_all = daily_energy(L_plot)
CSV.write("Energy_All.csv", df_energy_all)



# Cost calculation
function daily_cost(L_plot)
    Cost_dict = Dict{String, Dict{Date, Float64}}()
    methods = keys(L_plot)
    for method in methods
        Cost_dict[method] = Dict{Date, Float64}()
        for today_update in days
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

            L_daily = L_plot[method][today_update]
            Index_onpeak = N_start_idx:N_end_idx
            Index_offpeak = vcat(1:N_start_idx-1, N_end_idx+1:N)
            gamma_nc = maximum(L_daily[t] for t in 1:N)
            gamma_onpeak = maximum(L_daily[t] for t in Index_onpeak)
            demand_charge = r_power_nc * gamma_nc + r_power_onpeak * gamma_onpeak
            energy_charge = delta_t * sum(r_energy_offpeak * L_daily[t] for t in Index_offpeak) + 
                                                delta_t * sum(r_energy_onpeak * L_daily[t] for t in Index_onpeak)
            other_charge = 0.0578 * (demand_charge + energy_charge) + (0.0058 + 0.00058 + 0.0003) * sum(L_daily) * delta_t
            cost = demand_charge + energy_charge + other_charge
            Cost_dict[method][today_update] = cost
        end
    end

    df_cost = DataFrame(day = days)
    for method in methods
        df_cost[!, method] = [Cost_dict[method][day] for day in days]
    end

    return df_cost
end

df_cost = daily_cost(L_plot)
CSV.write("Cost_All.csv", df_cost)



p_cost = groupedbar(
    string.(df_cost.day),
    hcat([df_cost[!, method] for method in names(df_cost)[2:end]]...),
    label=permutedims(methods),
    xlabel="Day",
    ylabel="Cost (USD)",
    bar_width=0.7,
    size=(1000, 600),
    dpi=300,
    legend=:topleft,
    title="Cost Comparison",
    bar_position = :dodge
)
savefig(p_cost, "Cost_All.png")


# Forecast error
Charger_forecast = OrderedDict{String, Any}([
    "Perfect" => forecast_charger_perfect,
    "Noforecast" => forecast_charger_noforecast,
    "Persistence" => forecast_charger_persistence,
    "LSTM" => forecast_charger_lstm
])

# TODO: maybe better plot, violin?
# https://docs.juliaplots.org/dev/generated/statsplots/
function forecast_error(Charger_forecast)
    ed_metrics = Dict{String, Dict{Date, Dict{String, Float64}}}()

    for method in keys(Charger_forecast)
        ed_metrics[method] = Dict{Date, Dict{String, Float64}}()
        for today_update in days
            forecast = Charger_forecast[method][today_update]
            perfect = Charger_forecast["Perfect"][today_update]

            abs_error = abs.(forecast.ED .- perfect.ED)
            mse = mean(abs_error .^ 2)
            mae = mean(abs_error)
            rmse = sqrt(mse)

            ed_metrics[method][today_update] = Dict(
                "MSE" => mse,
                "MAE" => mae,
                "RMSE" => rmse,
            )
        end
    end
    return ed_metrics
end

ed_metrics = forecast_error(Charger_forecast)

function flatten_ed_metrics(ed_metrics)
    rows = []
    for (method, date_dict) in ed_metrics
        for (date, metric_dict) in date_dict
            for (metric, value) in metric_dict
                push!(rows, (method=method, date=date, metric=metric, value=value))
            end
        end
    end
    return DataFrame(rows)
end

df_metrics = flatten_ed_metrics(ed_metrics)

@df df_metrics groupedbar(:date, :value, group = :method, bar_position = :dodge,
                          layout = (1, length(keys(ed_metrics))), by = :metric,
                          legend = :top, xlabel = "Date", ylabel = "Error",
                          title = "Forecast", size = (2000, 600),
                          dpi = 300, bar_width = 0.7,
                          titlefontsize = 12, labelfontsize = 10,
                          tickfontsize = 10, xtickfontsize = 10,
                          ytickfontsize = 10)

savefig("Forecast_Error.png")














       