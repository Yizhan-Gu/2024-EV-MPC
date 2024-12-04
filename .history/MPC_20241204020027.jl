#=
Author: Yizhan Gu
Date: 2024-08-13
Description: This is the main file for the MPC project
Affiliation: University of California San Diego
Email: yig031@ucsd.edu
All rights reserved
=#

####################################################
# SECTION: Testing if Julia works and set working directory
print("hello world")
cd("/Users/admin/Desktop/EV_program/2024Summer_EVResearch")
print(pwd())

####################################################
# Importing the necessary packages
using Pkg
packages = ["JuMP", "Ipopt", "LinearAlgebra", "Plots", "Random", "CSV", "DataFrames", "Statistics", "StatsBase", "ProgressMeter", "Dates", "DataFramesMeta", "Distributions", "JLD", "GaussianMixtures", "Holidays", "AutoMLPipeline", "StatsPlots"]

for package in packages
    Pkg.add(package)
end # Install the necessary packages

for package in packages
    try
        @eval using $(Symbol(package))
        # println("Successfully loaded: $package")
    catch e
        println("Error loading package: $package - $e")
    end
end
print("All packages are successfully loaded")

# Clear console
print("\033c") # Or REPL: Ctrl + L

####################################################
# Data preprocessing

charging_sessions = CSV.read("clean_charging_sessions.csv", DataFrame)
charging_sessions = sort(charging_sessions, [:session_start_time_la])

# Plot statistics
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


####################################################
# MPC optimization on EV charging cost minimization and peak shaving
#   V0G: Standard EV charging without any smart grid interaction.
#	V1G: Optimizes the charging process for cost savings or environmental benefits.
#	V2G: Advanced grid integration where the EV plays an active role in energy markets and grid stability.
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
T_start_idx = Int(16 / delta_t + 1)
T_end_idx = Int(21 / delta_t)
DWR_charge = 0.0


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
# SECTION: MPC optimization function
# VERSION: Charger based modeling with Adil's method
function rum_mpc_charger(data_input::DataFrame, data_today::DataFrame, method::String)
    # TODO: The input of data should also be changed with station and charger
    1
end









# TODO: for testing
data_input = copy(data_forecast)


# VERSION: EV based modeling with Graham's method
function run_mpc_ev(data_input::DataFrame, data_today::DataFrame, method::String)
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

    data_today.AT = ceil.(Float64.(Dates.hour.(data_today.session_start_time_la)) + Float64.(Dates.minute.(data_today.session_start_time_la)) / 60, digits=2)
    N_ev_last = size(data_input, 1)

    # Size will be different for each time step
    L_mpc = zeros(N)
    P_mpc = Vector{Vector{Float64}}(undef, N)
    E_mpc = Vector{Vector{Float64}}(undef, N)
    E_tmp = Vector{Float64}()
    # P_mpc = zeros(N, N_ev)
    # E_mpc = zeros(N, N_ev)
    # E_tmp = zeros(N_ev)
    Optimal = zeros(Bool, N)
    Infeasible = zeros(Bool, N)

    @showprogress for k in 1:N
        # No real-time update
        if (method == "Perfect")
            data_forecast_update = copy(data_input)
        # Real-time update at every time step: only arrived EVs can be charged and considered in the MPC
        else
            data_forecast_update = DataFrame()
            update_time = (k - 1) * delta_t
            last_step_time = (k - 2) * delta_t

            arrived_sessions_today = filter(row -> row.AT <= update_time, data_today)
            arrived_sessions_forecast = filter(row -> row.AT <= update_time, data_input)
            arrived_sessions_today.ED = zeros(size(arrived_sessions_today, 1))
            arrived_sessions_today.DT = zeros(size(arrived_sessions_today, 1))
            arrived_sessions_today.PD = zeros(size(arrived_sessions_today, 1))

            arrived_sessions_step = filter(row -> last_step_time < row.AT <= update_time, data_today)

            if method == "Noforecast"
                if size(arrived_sessions_forecast, 1) != size(arrived_sessions_today, 1)
                    error("Noforecast method should have the same as perfect forecast")
                elseif isempty(arrived_sessions_forecast)
                    data_forecast_update = copy(data_input)
                    data_forecast_update.AT .= 24
                    data_forecast_update.DT .= 24
                elseif !isempty(arrived_sessions_forecast)
                    data_forecast_update = copy(arrived_sessions_forecast)
                end
            elseif (method == "Persistence+KNN" || method == "Statistic+KNN")
                if isempty(arrived_sessions_today)
                    data_forecast_update = copy(data_input)
                    # postpone time so that unarrived EVs are not charged
                    data_forecast_update.AT .= 24
                    data_forecast_update.DT .= 24
                    # data_forecast_update.ED .= 0
                elseif !isempty(arrived_sessions_today)
                    # VERSION: Assume all real AT, DT, ED, PD are known after arrival of EVs to make sure the area of load plot is the same as the real data
                    arrived_sessions_today.DT = floor.(Float64.(Dates.hour.(arrived_sessions_today.session_end_time_la)) + Float64.(Dates.minute.(arrived_sessions_today.session_end_time_la)) / 60, digits=2)
                    arrived_sessions_today.ED = arrived_sessions_today.total_energy_dispensed
                    arrived_sessions_today.PD = ceil.(Dates.value.(arrived_sessions_today.charging_end_time_la - arrived_sessions_today.session_start_time_la) / (3600*1000), digits=2)

                    #= VERSION: Assume only real AT is known after arrival of EVs
                    for i in 1:size(arrived_sessions_today, 1)
                        closest_idx = argmin(abs.(arrived_sessions_forecast.AT .- arrived_sessions_today.AT[i]))
                        arrived_sessions_today.PD[i] = arrived_sessions_forecast.PD[closest_idx]
                        arrived_sessions_today.DT[i] = arrived_sessions_today.PD[i] + arrived_sessions_today.AT[i]
                        if method == "Persistence+KNN"
                            arrived_sessions_today.ED[i] = arrived_sessions_forecast.ED[closest_idx]
                        elseif method == "Statistic+KNN"
                            arrived_sessions_today.ED[i] = arrived_sessions_forecast.ED[closest_idx] * M # Difference
                        end
                    end
                    =#

                    noarrived_sessions_forecast = filter(row -> row.AT > update_time, data_input) 
                    if !isempty(arrived_sessions_forecast)
                        # Instead of changing EV number, change the ED of the forecasted EVs
                        noarrived_sessions_forecast.ED = noarrived_sessions_forecast.ED * size(arrived_sessions_today, 1) / size(arrived_sessions_forecast, 1)
                    elseif isempty(arrived_sessions_forecast)
                        # Increase ED of the forecasted EVs
                        noarrived_sessions_forecast.ED = noarrived_sessions_forecast.ED * (1 + k / N)
                    end

                    data_forecast_update = vcat(arrived_sessions_today, noarrived_sessions_forecast)
                end
            end
        end


        N_ev = size(data_forecast_update, 1) # Combine the forecasted and arrived EVs
        model = Model(Ipopt.Optimizer)
        # Set solver options
        set_optimizer_attribute(model, "max_iter", 3000)  # Set maximum number of iterations
        set_optimizer_attribute(model, "tol", 1e-4)       # Set tolerance for convergence
        set_optimizer_attribute(model, "acceptable_tol", 1e-3)  # Set acceptable tolerance
        set_optimizer_attribute(model, "print_level", 2)  # Set printing level (0: no output, 5: full output)
        # set_optimizer_attribute(model, "constr_viol_tol", 1e-3)  # Set constraint violation tolerance
        set_optimizer_attribute(model, "warm_start_init_point", "yes")

        if N_ev >= N_ev_last
            flag = "more"
        elseif N_ev < N_ev_last
            flag = "fewer"
        end

        @variables model begin
            P[k:N, 1:N_ev] >= 0
            L[k:N] >= 0
            E[k:N, 1:N_ev] >= 0
            gamma_nc_k >= 0
            gamma_onpeak_k >= 0
        end

        # Constraints
        # @constraint(model, [t=k:N, i=1:N_ev], P[t, i] <= M * (t >= AT_idx[i]))
        # @constraint(model, [t=k:N, i=1:N_ev], P[t, i] <= M * (t <= DT_idx[i]))


        AT = data_forecast_update.AT # 0.00-23.99
        DT = data_forecast_update.DT # 0.00-23.98
        ED = data_forecast_update.ED
        PD = data_forecast_update.PD
        AT_idx = floor.(Int, AT / T * N) .+ 2 # Note: idx can only be 1-96 for comparison with k
        DT_idx = floor.(Int, DT / T * N) .+ 1

        if k == 1
            @constraint(model, [i=1:N_ev], E[k, i] == 0)
        elseif k >= 2
            if flag == "more"
                E_tmp = vcat(E_tmp, zeros(N_ev - N_ev_last))
            elseif flag == "fewer"
                #= Changing the real ED is illegal!
                # ED = ED .- sum(E_tmp[N_ev+1:N_ev_last]) / N_ev # will lead to negative energy demand
                for i in 1:N_ev
                    ED[i] -= sum(E_tmp[N_ev+1:N_ev_last]) / sum(ED) * ED[i] # will only change when time tends to be the end b.c. EVs are charged by time
                end
                =#
                E_tmp = E_tmp[1:N_ev]
                #= if any(E_tmp - ED .> 1e-4)
                    diff = ED - E_tmp
                    println("Difference between ED and E_tmp: ", diff)
                    p1 = plot(ED, label="ED", title="Energy Demand Check")
                    plot!(p1, E_tmp, label="E_tmp")
                    display(p1)  # Force plot to display before error
                    error("Negative energy demand")
                end
                =#
            end

            @constraint(model, [i=1:N_ev], E[k, i] == E_tmp[i] + P[k, i] * delta_t) # E_tmp is the energy state vector stored at the previous loop of k
            @constraint(model, [t=k+1:N, i=1:N_ev], E[t, i] == E[t-1, i] + P[t, i] * delta_t)
        end
        

        for i in 1:N_ev
            # VERSION: Assume charing can happen at DT_idx, or otherwise for AT_idx[i] == DT_idx[i] the ED can never be reached
            # @constraint(model, [t=k:N], E[t, i] <= ED[i])
            @constraint(model, [t=k:N], 0 <= P[t, i] <= P_max)
            if AT_idx[i] > DT_idx[i]
                print(AT_idx, " ", DT_idx, "\n", k)
                error("AT should be less than DT")
            # elseif AT_idx[i] == DT_idx[i]
                # @constraint(model, [t=k:N], P[t, i] == 0)
            elseif AT_idx[i] <= k && DT_idx[i] >= k
                # @constraint(model, 0.8 * ED[i] <= E[DT_idx[i], i] <= ED[i]) # Non-strict
                @constraint(model, E[DT_idx[i], i] == ED[i]) # Strict
                @constraint(model, [t=DT_idx[i]+1:N], P[t, i] == 0)
            elseif DT_idx[i] < k
                @constraint(model, [t=k:N], P[t, i] == 0)
            elseif AT_idx[i] > k
                @constraint(model, [t=k:AT_idx[i]-1], P[t, i] == 0)
            end
        end
    
        @constraint(model, [t=k:N], L[t] == sum(P[t, i] for i in 1:N_ev))

        # Peak horizons and costs
        Index_onpeak = []
        Index_offpeak = []

        if k < T_start_idx
            Index_onpeak = T_start_idx:T_end_idx
            Index_offpeak = vcat(k:T_start_idx-1, T_end_idx+1:N)
        elseif T_start_idx <= k <= T_end_idx
            Index_onpeak = k:T_end_idx
            Index_offpeak = T_end_idx+1:N
        elseif k > T_end_idx
            Index_onpeak = []
            Index_offpeak = k:N
        end        

        @constraint(model, [t=k:N], gamma_nc_k >= L[t])
        # @constraint(model, [t=k:N], gamma_nc_k == maximum(L[t])) # This is not working for IPOPT

        if !isempty(Index_onpeak)
            @constraint(model, [t in Index_onpeak], gamma_onpeak_k >= L[t])
            # @constraint(model, gamma_onpeak_k == maximum(L[t] for t in Index_onpeak))
        elseif isempty(Index_onpeak)
            @constraint(model, gamma_onpeak_k == 0)
        end
        

        # Objective function
        # Define the demand charge as an expression
        @expression(model, demand_charge_k, r_power_nc * gamma_nc_k + r_power_onpeak * gamma_onpeak_k)

        # Combine the components into the total energy charge
        if isempty(Index_onpeak)
            @expression(model, energy_charge_k, delta_t * sum(r_energy_offpeak * L[t] for t in Index_offpeak))
        elseif !isempty(Index_onpeak)
            @expression(model, energy_charge_k, delta_t * sum(r_energy_offpeak * L[t] for t in Index_offpeak) + 
                                            delta_t * sum(r_energy_onpeak * L[t] for t in Index_onpeak))
        end

                                            

        # Define the other charges as an expression
        # TODO: Some details of the other charges are not clear
        @expression(model, E_total, sum(ED[i] for i in 1:N_ev))

        @expression(model, other_charge_k, 0.0578 * (demand_charge_k + energy_charge_k) +
                        (0.0058 + 0.00058 + 0.0003) * E_total +
                        0.0688 * DWR_charge)

        # Define the objective function as an expression
        @expression(model, J_k, demand_charge_k + energy_charge_k + other_charge_k)

        @objective(model, Min, J_k)

        optimize!(model)

        L_mpc[k] = value(L[k])
        P_mpc[k] = value.(P[k, :])
        E_mpc[k] = value.(E[k, :])
        E_tmp = E_mpc[k]
        N_ev_last = N_ev # Update the number of EVs for the next loop

        # P_mpc[k, :] = value.(P[k, :])
        # E_mpc[k, :] = value.(E[k, :])
        # E_tmp = E_mpc[k, :] # Update the energy state vector for the next loop

        # It's ok some of the solutions are not optimal, but all should be feasible
        Optimal[k] = (termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED) ? 1 : 0
        Infeasible[k] = (termination_status(model) == MOI.INFEASIBLE) ? 1 : 0
    end
    # print("MPC optimization is done with time: ", ceil(toc - tic), " seconds\n")
    print("Optimal Found: ", sum(Optimal), " out of ", N, "\n")
    print("Infeasible Found: ", sum(Infeasible), " out of ", N, "\n")

    return L_mpc, P_mpc, E_mpc
end



####################################################
# Forecast is run daily and updated with the latest data at each time step

# Daily updates:
# V0G: Dumb charging with max power
# Perfect forecast: AT, DT, ED, Nev are known, and true data is used
# Persistence forecast: AT, DT, ED, Nev are the same as the previous daily data, Nev needs to be updated

# Real-time updates:
# KNN forecast: AT, DT, ED are updated by KNN and Nev is updated by rules
# GMM forecast: AT, DT, ED are forecasted by GMM model


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


# Function to predict using the most likely component
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
    best_gmm_AT, best_gmm_DT, best_gmm_ED, best_gmm_PD = best_gmms[1:4]

    # Prepare arrays to store predictions
    n = size(data_input, 1)
    predict_gmm_AT = zeros(n)
    predict_gmm_DT = zeros(n)
    predict_gmm_ED = zeros(n)
    predict_gmm_PD = zeros(n)

    # For each data point, find the most likely component and use its mean as the prediction, and then update with Ben's method
    # https://docs.google.com/presentation/d/1EMBE8Me50NhXHq-kFVkho-nn2YyRwtEi_801p6mAHo4/edit?pli=1#slide=id.g1ee332c0660_1_245
    # TODO: Not sure how to update the prediction with Ben's method
    for i in 1:n
        # Extract the data point
        x = [data_input.AT[i], data_input.DT[i], data_input.ED[i], data_input.PD[i]]
        # Compute the likelihood of the data point under each GMM
        likelihood_AT = gmm_pdf(best_gmm_AT, x[1])
        likelihood_DT = gmm_pdf(best_gmm_DT, x[2])
        likelihood_ED = gmm_pdf(best_gmm_ED, x[3])
        likelihood_PD = gmm_pdf(best_gmm_PD, x[4])
        # Find the most likely component for each feature
        component_AT = argmax([likelihood_AT])
        component_DT = argmax([likelihood_DT])
        component_ED = argmax([likelihood_ED])
        component_PD = argmax([likelihood_PD])
        # Use the mean of the most likely component as the prediction
        predict_gmm_AT[i] = best_gmm_AT.μ[component_AT]
        predict_gmm_DT[i] = best_gmm_DT.μ[component_DT]
        predict_gmm_ED[i] = best_gmm_ED.μ[component_ED]
        predict_gmm_PD[i] = best_gmm_PD.μ[component_PD]
    end

    return predict_gmm_AT, predict_gmm_DT, predict_gmm_ED, predict_gmm_PD
end



####################################################
# SECTION: Forecast function
function forecast(data_input::DataFrame, method::String, updated_sessions::DataFrame)
    data_today = copy(data_input)

    forecast_list = ["Perfect", "GMM", "Persistence+KNN", "Statistic+KNN", "Noforecast"]
    if method ∉ forecast_list
        error("Invalid forecast method")
    elseif method == "Perfect" || method == "Noforecast"
        # The AT, DT, ED, PD should comply with or be more precise than the step length of MPC
        data_input.AT = ceil.(Float64.(Dates.hour.(data_input.session_start_time_la)) + Float64.(Dates.minute.(data_input.session_start_time_la)) / 60, digits=2)
        data_input.DT = floor.(Float64.(Dates.hour.(data_input.session_end_time_la)) + Float64.(Dates.minute.(data_input.session_end_time_la)) / 60, digits=2)
        data_input.ED = data_input.total_energy_dispensed
        data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)
    elseif method == "Persistence+KNN"
        # Just do the MPC with data from the previous smart day (weekday, weekend, holiday)
        target_time = data_input.session_start_time_la[1]
        target_day_type = datetype(target_time)
        same_type_sessions = updated_sessions[(updated_sessions.type .== target_day_type) .& (Date.(updated_sessions.session_start_time_la) .< Date(target_time)), :]
        if nrow(same_type_sessions) == 0
            error("No valid sessions found for the same day type")
        end

        closest_idx = argmin(abs.(Date.(same_type_sessions.session_start_time_la) .- Date(target_time)))
        closest_date = Date(same_type_sessions.session_start_time_la[closest_idx])
        persistence_sessions = same_type_sessions[Date.(same_type_sessions.session_start_time_la) .== closest_date, :]

        data_input = persistence_sessions
        data_input.AT = ceil.(Float64.(Dates.hour.(data_input.session_start_time_la)) + Float64.(Dates.minute.(data_input.session_start_time_la)) / 60, digits=2)
        data_input.DT = floor.(Float64.(Dates.hour.(data_input.session_end_time_la)) + Float64.(Dates.minute.(data_input.session_end_time_la)) / 60, digits=2)
        data_input.ED = data_input.total_energy_dispensed
        data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)

    elseif method == "GMM"
        data_input.AT, data_input.DT, data_input.ED, data_input.PD = predict_gmm(data_input) 

    elseif method == "Statistic+KNN"
        target_time = data_input.session_start_time_la[1]
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
        data_input = closest_sessions
        data_input.AT = ceil.(Float64.(Dates.hour.(data_input.session_start_time_la)) + Float64.(Dates.minute.(data_input.session_start_time_la)) / 60, digits=2)
        data_input.DT = floor.(Float64.(Dates.hour.(data_input.session_end_time_la)) + Float64.(Dates.minute.(data_input.session_end_time_la)) / 60, digits=2)
        data_input.ED = data_input.total_energy_dispensed / M
        data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)
    end

    return data_input, data_today
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
            Energy_dispensed = sum(P_V0G[1:k, i]) * delta_t
            Remaining_energy = data_input.ED[i] - Energy_dispensed

            # Check charging condition and apply adjusted final charging power if needed
            if data_input.AT[i] <= k * delta_t <= data_input.DT[i] && Energy_dispensed < data_input.ED[i]
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

    L_V0G = sum(P_V0G, dims=2)
    return L_V0G
end


# Daily update main function
function daily_update(data_input::DataFrame, method::String)
    days = unique(Date.(data_input.session_start_time_la))
    updated_sessions = copy(charging_sessions)
    updated_sessions.type = datetype.(updated_sessions.session_start_time_la)
    L_mpc_dict = Dict{Date, Any}()
    P_mpc_dict = Dict{Date, Any}()
    E_mpc_dict = Dict{Date, Any}()
    L_V0G_dict = Dict{Date, Any}()

    tic = time()
    for today_update in days
        data_today = filter(row -> Dates.Date(row.session_start_time_la) == today_update, data_input)
        L_V0G_dict[today_update] = run_V0G(data_today)
        data_today.type = datetype.(data_today.session_start_time_la)
        select!(data_today, Not([:AT, :DT, :ED]))
        tmp_sessions = vcat(updated_sessions, data_today)
        data_forecast, data_today = forecast(data_today, method, updated_sessions)
        
        L_mpc_tmp, P_mpc_tmp, E_mpc_tmp = run_mpc_ev(data_forecast, data_today, method)
        updated_sessions = copy(tmp_sessions)

        #println("Day: ", today_update, " Number of EV: ", size(data_today, 1), "\n")
        #println("Day: ", today_update, " Number of EV in daily forecast: ", size(data_forecast, 1), "\n")
        #println("Day: ", today_update, " Number of EV after last real-time update: ", length(E_mpc_tmp[N]), "\n")
        L_mpc_dict[today_update] = L_mpc_tmp
        P_mpc_dict[today_update] = P_mpc_tmp
        E_mpc_dict[today_update] = E_mpc_tmp
    end
    toc = time()
    println("Daily update run time with $method in $(ceil(toc - tic)) seconds\n")
    
    return L_V0G_dict, L_mpc_dict, P_mpc_dict, E_mpc_dict
end


####################################################
# SECTION: Testing
# https://github.com/rdeits/DynamicWalking2018.jl/blob/master/notebooks/6.%20Optimization%20with%20JuMP.ipynb
# Test data is unknown with all other data known
test_sessions = CSV.read("clean_test_sessions.csv", DataFrame)
test_sessions_filtered = filter(row -> 
    (Dates.hour(row.session_end_time_la) + Dates.minute(row.session_end_time_la) / 60) - 
    (Dates.hour(row.session_start_time_la) + Dates.minute(row.session_start_time_la) / 60) > 0.25,
    test_sessions)
test_sessions = sort(test_sessions, [:session_start_time_la])
start_date = Date(minimum(test_sessions.session_start_time_la)) + Day(2)
end_date = start_date + Day(1)
data_test = filter(row -> start_date <= row.session_start_time_la <= end_date, test_sessions)
data_test = sort(data_test, [:session_start_time_la])


# TODO: For testing
data_input = copy(data_test)
method = "Noforecast"

# Pick the closest M days in "Statistic+KNN"
M = 3
# VERSION: Pick the method from below
method_list = ["Perfect", "Noforecast", "Persistence+KNN", "Statistic+KNN"]
days = unique(Date.(data_test.session_start_time_la))


L_V0G, L_mpc_perfect, P_mpc_perfect, E_mpc_perfect = daily_update(data_test, "Perfect");
L_V0G, L_mpc_noforecast, P_mpc_noforecast, E_mpc_noforecast = daily_update(data_test, "Noforecast");
L_V0G, L_mpc_persistence_knn, P_mpc_persistence_knn, E_mpc_persistence_knn = daily_update(data_test, "Persistence+KNN");
L_V0G, L_mpc_statistic_knn, P_mpc_statistic_knn, E_mpc_statistic_knn = daily_update(data_test, "Statistic+KNN");


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

    df_load_V0G  = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], load = vec(vcat(values(L_V0G)...)))
    df_load_perfect = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], load = vcat(values(L_mpc_perfect)...))
    df_load_noforecast = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], load = vcat(values(L_mpc_noforecast)...))
    df_load_persistence_knn = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], load = vcat(values(L_mpc_persistence_knn)...))
    df_load_statistic_knn = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], load = vcat(values(L_mpc_statistic_knn)...))

    xlim = extrema(vcat(df_AT.interval, df_DT.interval, df_load_V0G.time))
    hist_AT = bar(df_AT.interval, df_AT.count, xlabel="AT", ylabel="Sessions", label="AT", color=:lightblue, alpha=0.7, size=(1400, 600), dpi=300, xlims=xlim)
    hist_DT = bar(df_DT.interval, df_DT.count, xlabel="DT", ylabel="Sessions", label="DT", color=:lightblue, alpha=0.7, size=(1400, 600), dpi=300, xlims=xlim)
    # hist_AT = histogram(data_test.session_start_time_la, bins = length(days) * N, alpha=0.5, label="AT", xlabel="AT", ylabel="Sessions")
    # hist_DT = histogram(data_test.session_end_time_la, bins = length(days) * N, alpha=0.5, label="DT", xlabel="DT", ylabel="Sessions")

    p_load_combined = plot(df_load_V0G.time, df_load_V0G.load,
        label="V0G",  # Add cost to label
        xlabel="Timestamp",
        ylabel="Load (kW)",
        size=(1400, 600),
        dpi=300,
        legend=:topright,
        legendfontsize=10,
        xlims=xlim)

    plot!(df_load_perfect.time, df_load_perfect.load, label="Perfect Forecast")
    plot!(df_load_noforecast.time, df_load_noforecast.load, label="No Forecast")
    plot!(df_load_persistence_knn.time, df_load_persistence_knn.load, label="Persistence+KNN Forecast")
    plot!(df_load_statistic_knn.time, df_load_statistic_knn.load, label="Statistic+KNN Forecast")
    p = plot(hist_AT, hist_DT, p_load_combined, layout=grid(3, 1, heights=[0.2 ,0.2, 0.6]), size=(3000, 1000))
    return p
end

p_load = load_plot()
savefig(p_load, "Load_All.png")



# Energy checking
# FIXME: Still not correct with less energy dispensed, even if using the perfect forecast into the other methods of MPC

function daily_energy()
    Energy_real = Dict{Date, Float64}()
    Energy_V0G = Dict{Date, Float64}()
    Energy_perfect = Dict{Date, Float64}()
    Energy_noforecast = Dict{Date, Float64}()
    Energy_persistence_knn = Dict{Date, Float64}()
    Energy_statistic_knn = Dict{Date, Float64}()
    for today in days
        data_today = data_test[Date.(data_test.session_start_time_la) .== today, :]
        Energy_real[today] = sum(data_today.total_energy_dispensed)
        Energy_V0G[today] = sum(L_V0G[today]) * delta_t
        Energy_perfect[today] = sum(L_mpc_perfect[today]) * delta_t
        Energy_noforecast[today] = sum(L_mpc_noforecast[today]) * delta_t
        Energy_persistence_knn[today] = sum(L_mpc_persistence_knn[today]) * delta_t
        Energy_statistic_knn[today] = sum(L_mpc_statistic_knn[today]) * delta_t
    end
    df_energy_all = DataFrame(
    day = days,
    Real = [Energy_real[day] for day in days],
    V0G = [Energy_V0G[day] for day in days],
    Perfect = [Energy_perfect[day] for day in days],
    Noforecast = [Energy_noforecast[day] for day in days],
    Persistence_KNN = [Energy_persistence_knn[day] for day in days],
    Statistic_KNN = [Energy_statistic_knn[day] for day in days]
    )
    return df_energy_all
end

df_energy_all = daily_energy()
CSV.write("Energy.csv", df_energy_all)



# Cost calculation
function daily_cost(Load_input::Dict{Date, Array{Float64, 1}})
    Cost_dict = Dict{Date, Any}()
    for today in days
        cost = 0.0
        season = get_season(DateTime(today))
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

        L_daily = Load_input[today]
        Index_onpeak = T_start_idx:T_end_idx
        Index_offpeak = vcat(1:T_start_idx-1, T_end_idx+1:N)
        gamma_nc = maximum(L_daily[t] for t in 1:N)
        gamma_onpeak = maximum(L_daily[t] for t in Index_onpeak)
        demand_charge = r_power_nc * gamma_nc + r_power_onpeak * gamma_onpeak
        energy_charge = delta_t * sum(r_energy_offpeak * L_daily[t] for t in Index_offpeak) + 
                                            delta_t * sum(r_energy_onpeak * L_daily[t] for t in Index_onpeak)
        other_charge = 0.0578 * (demand_charge + energy_charge) +
                        (0.0058 + 0.00058 + 0.0003) * real_ED +
                        0.0688 * DWR_charge
        cost = demand_charge + energy_charge + other_charge
        Cost_dict[today] = cost
    end
    return Cost_dict
end

Cost_V0G = daily_cost(L_V0G)
Cost_perfect = daily_cost(L_mpc_perfect)
Cost_noforecast = daily_cost(L_mpc_noforecast)
Cost_persistence_knn = daily_cost(L_mpc_persistence_knn)
Cost_statistic_knn = daily_cost(L_mpc_statistic_knn)

df_cost = DataFrame(day = days,
                    V0G = collect(values(Cost_V0G)),
                    Perfect = collect(values(Cost_perfect)),
                    Noforecast = collect(values(Cost_noforecast)),
                    Persistence_KNN = collect(values(Cost_persistence_knn)),
                    Statistic_KNN = collect(values(Cost_statistic_knn)))

p_cost = groupedbar(df_cost.day, [df_cost.V0G df_cost.Perfect df_cost.Noforecast df_cost.Persistence_KNN df_cost.Statistic_KNN], 
                    label=["V0G" "Perfect" "Noforecast" "Persistence+KNN" "Statistic+KNN"], xlabel="Day", ylabel="Cost", 
                    bar_width=0.7, size=(800, 600), dpi=300, legend=:topright,             # Move legend to top-right corner
                    title="Cost Comparison")

savefig(p_cost, "Cost_All.png")
CSV.write("Cost.csv", df_cost)