#=
Author: Yizhan Gu
Date: 2024-08-13
Description: This is the main file for the MPC project
Affiliation: University of California San Diego
Email: yig031@ucsd.edu
All rights reserved
=#


####################################################
# Testing if Julia works and set working directory
print("hello world")
cd("/Users/admin/Desktop/EV_program/2024Summer_EVResearch")
print(pwd())

####################################################
# Importing the necessary packages
using Pkg
packages = ["JuMP", "Ipopt", "LinearAlgebra", "Plots", "Random", "CSV", "DataFrames", "Statistics", "StatsBase", "ProgressMeter", "Dates", "DataFramesMeta", "Distributions", "JLD", "GaussianMixtures"]

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

#= Importing the necessary files
charging_sessions = CSV.read("/Users/admin/Desktop/EV_program/2023Fall_TotalEnergies/data_sessions.csv", DataFrame)
charging_sessions.session_start_time_la = DateTime.(charging_sessions.session_start_time_la, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.session_end_time_la = DateTime.(charging_sessions.session_end_time_la, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.charging_end_time_la = DateTime.(charging_sessions.charging_end_time_la, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.Time_of_day = Time.(charging_sessions.session_start_time_la)
charging_sessions.AT_day = Dates.Time.(charging_sessions.session_start_time_la)
charging_sessions.DT_day = Dates.Time.(charging_sessions.session_end_time_la)
select!(charging_sessions, Not(:Time_of_day))

# Save the updated DataFrame to a CSV file
CSV.write("clean_charging_sessions.csv", charging_sessions)
=#

charging_sessions = CSV.read("clean_charging_sessions.csv", DataFrame)

names(charging_sessions)
show(first(charging_sessions, 1), allcols=true)




####################################################
# MPC optimization on EV charging cost minimization and peak shaving (V1G)
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
# Shrinking MPC is run daily
function run_mpc(data_input::DataFrame)
    season = get_season(data_input.session_start_time_la[1]) # Assume all sessions happen in the same season

    AT = data_input.AT
    DT = data_input.DT
    ED = data_input.ED
    PD = data_input.PD
    forecasted_n_EV = data_input.forecasted_n_EV[1]

    L_mpc = zeros(N)
    P_mpc = zeros(N, forecasted_n_EV)
    E_mpc = zeros(N, forecasted_n_EV)

    # Define rates based on season
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

    # Initialize variables
    M = 1e6
    AT_idx = floor.(Int, AT / T * N)
    DT_idx = floor.(Int, DT / T * N)

    # Optimize with loop through each time slot b.c. change of objective function
    tic = time()
    @showprogress for k in 1:N
        model = Model(Ipopt.Optimizer)
        # Set solver options
        set_optimizer_attribute(model, "max_iter", 1000)  # Set maximum number of iterations
        set_optimizer_attribute(model, "tol", 1e-6)       # Set tolerance for convergence
        set_optimizer_attribute(model, "acceptable_tol", 1e-4)  # Set acceptable tolerance
        set_optimizer_attribute(model, "print_level", 3)  # Set printing level (0: no output, 5: full output)

        @variables model begin
            P[k:N, 1:forecasted_n_EV] >= 0
            L[k:N] >= 0
            E[k:N, 1:forecasted_n_EV] >= 0
            gamma_nc_k >= 0
            gamma_onpeak_k >= 0
        end

        # Constraints

        # @constraint(model, [t=k:N, i=1:forecasted_n_EV], P[t, i] <= M * (t >= AT_idx[i]))
        # @constraint(model, [t=k:N, i=1:forecasted_n_EV], P[t, i] <= M * (t <= DT_idx[i]))

        for i in 1:forecasted_n_EV
            @constraint(model, [t=k:N], 0 <= P[t, i] <= P_max)
            # @constraint(model, [t=k:N], E[t, i] <= ED[i])

            if DT_idx[i] >= k
                @constraint(model, E[DT_idx[i], i] == ED[i])
            elseif DT_idx[i] < k
                @constraint(model, [t=k:N], P[t, i] == 0)
            end

            if AT_idx[i] > k
                @constraint(model, [t=k:AT_idx[i]-1], P[t, i] == 0)
            end

            # Initialize the energy state
            if k == 1
                @constraint(model, E[k, i] == 0)
            end

            if k <= N-1
                @constraint(model, [t=k+1:N], E[t, i] == E[t-1, i] + P[t, i] * delta_t)
            end
        end
    
        @constraint(model, [t=k:N], L[t] == sum(P[t, i] for i in 1:forecasted_n_EV))

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

        # @constraint(model, [t=k:N], gamma_nc_k >= L[t])
        @constraint(model, [t=k:N], gamma_nc_k == maximum(L[t]))

        if !isempty(Index_onpeak)
            @constraint(model, gamma_onpeak_k == maximum(L[t] for t in Index_onpeak))
        elseif isempty(Index_onpeak)
            @constraint(model, gamma_onpeak_k == 0)
        end
        

        # Objective function
        # Define the demand charge as an expression
        @expression(model, demand_charge_k, r_power_nc * gamma_nc_k + r_power_onpeak * gamma_onpeak_k)

        # Combine the components into the total energy charge
        @expression(model, energy_charge_k, delta_t * sum(r_energy_offpeak * L[t] for t in Index_offpeak) + 
                                            delta_t * sum(r_energy_onpeak * L[t] for t in Index_onpeak))


        # Define the other charges as an expression
        # TODO: the total energy use in a month is not known and assumed to be the sum of all ED in 30 days
        @expression(model, other_charge_k, 0.0578 * (demand_charge_k + energy_charge_k) +
                        (0.0058 + 0.00058 + 0.0003) * sum(ED[i] for i in 1:forecasted_n_EV) * 30 +
                        0.0688 * DWR_charge)

        # Define the objective function as an expression
        @expression(model, J_k, demand_charge_k + energy_charge_k + other_charge_k)

        @objective(model, Min, J_k)

        optimize!(model)

        L_mpc[k] = value(L[k])
        P_mpc[k, :] = value.(P[k, :])
        E_mpc[k, :] = value.(E[k, :])
    end
    toc = time()
    print("MPC optimization is done with time: ", ceil(toc - tic), " seconds")

    return L_mpc, P_mpc, E_mpc
end



####################################################
# Forecast is run daily

# Perfect forecast: AT, DT, ED, PD are known, and true data in dataset is used
# Persistence forecast: AT, DT, ED, PD are the same as the previous daily data
# GMM forecast: AT, DT, ED, PD are forecasted by GMM model

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


# Function to predict using the most likely component
function predict_gmm(data_input)
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

# TODO: For testing
data_input = data_test
method = "Perfect"
#

function forecast(data_input::DataFrame, method::String)
    forecast_list = ["Perfect", "Persistence", "GMM"]
    if method ∉ forecast_list
        error("Invalid forecast method")
    elseif method == "Perfect"
        # The AT, DT, ED, PD should comply with or be more precise than the step length of MPC
        data_input.AT = ceil.(Float64.(Dates.hour.(data_input.session_start_time_la)) + Float64.(Dates.minute.(data_input.session_start_time_la)) / 60, digits=2)
        data_input.DT = ceil.(Float64.(Dates.hour.(data_input.session_end_time_la)) + Float64.(Dates.minute.(data_input.session_end_time_la)) / 60, digits=2)
        data_input.ED = data_input.total_energy_dispensed
        data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)
        data_input.forecasted_n_EV .= size(data_input, 1)
        return nothing
    elseif method == "Persistence" # TODO: Need to update
        data_last_day = charging_sessions[
            Date.(charging_sessions.session_start_time_la) .== Date(data_input.session_start_time_la[1] - Day(1)), :
        ]
        data_input.AT .= ceil(mean(Float64.(Dates.hour.(data_last_day.session_start_time_la))), digits=2)
        data_input.DT .= ceil(mean(Float64.(Dates.hour.(data_last_day.session_end_time_la))), digits=2)
        data_input.ED .= mean(data_last_day.total_energy_dispensed)
        data_input.PD .= ceil(mean(Dates.value.(data_last_day.charging_end_time_la - data_last_day.session_start_time_la) / (3600*1000)), digits=2)
        data_input.forecasted_n_EV .= size(data_last_day, 1)
        return nothing
    elseif method == "GMM"
        data_input.AT, data_input.DT, data_input.ED, data_input.PD = predict_gmm(data_input) 
        data_input.forecasted_n_EV .= size(data_input, 1)
        return nothing
    end
end



####################################################
# Start MPC optimization
# Load test data
stations = first(unique(charging_sessions.station_name), 1)
start_date = minimum(charging_sessions.session_start_time_la)
end_date = start_date + Day(1)

data_test = filter(row -> start_date <= row.session_start_time_la <= end_date &&
                              row.station_name in stations, charging_sessions)
data_test = sort(data_test, [:station_name, :session_start_time_la])


# Run Forecast, MPC and Plot results
# https://github.com/rdeits/DynamicWalking2018.jl/blob/master/notebooks/6.%20Optimization%20with%20JuMP.ipynb

# select method from ["Perfect", "Persistence", "GMM"]
forecast(data_test, "Perfect")

L_mpc, P_mpc, E_mpc = run_mpc(data_test)


# Plot the results
hist_AT = histogram(data_test.AT, bins=1:24, alpha=0.5, label="AT", xlabel="AT", ylabel="Number of Sessions", xticks=1:24)

hist_DT = histogram(data_test.DT, bins=1:24, alpha=0.5, label="DT", xlabel="DT", ylabel="Number of Sessions", xticks=1:24)

p_load_MPC = plot(1:N, L_mpc,
         label="Perfect Forecast MPC",
         xlabel="Time Index",
         ylabel="Load (kW)",
         title="MPC Load Profile",
         size=(800, 600),  # Set the size of the plot (width, height) in pixels
         dpi=300)          # Set the DPI (dots per inch)

p_power_MPC = plot(1:N, P_mpc[:, 1],
         label="Perfect Forecast MPC sample 1",
         xlabel="Time Index",
         ylabel="Power (kW)",
         title="MPC Power Profile",
         size=(800, 600),  # Set the size of the plot (width, height) in pixels
         dpi=300)          # Set the DPI (dots per inch)

p_energy_MPC = plot(1:N, E_mpc[:, 1],
         label="Perfect Forecast MPC sample 1",
         xlabel="Time Index",
         ylabel="Energy (kWh)",
         title="MPC Energy Profile",
         size=(800, 600),  # Set the size of the plot (width, height) in pixels
         dpi=300)          # Set the DPI (dots per inch)

p = plot(hist_AT, hist_DT, p_load_MPC, p_power_MPC, p_energy_MPC, layout=(5, 1), size=(1000, 800), dpi=300)

savefig(p, "Load_MPC.png")


