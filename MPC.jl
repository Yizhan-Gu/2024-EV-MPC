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
packages = ["JuMP", "Ipopt", "LinearAlgebra", "Plots", "Random", "CSV", "DataFrames", "Statistics", "StatsBase", "ProgressMeter", "Dates", "DataFramesMeta", "Distributions"]

for package in packages
    Pkg.add(package)
end # Install the necessary packages

for package in packages
    eval(Meta.parse("using $package"))
end
print("All packages are successfully loaded")

# Clear console
print("\033c") # Or REPL: Ctrl + L

####################################################
# Data preprocessing

# Importing the necessary files
charging_sessions = CSV.read("/Users/admin/Desktop/EV_program/2023Fall_TotalEnergies/data_sessions.csv", DataFrame)
charging_sessions.session_start_time_pacific = DateTime.(charging_sessions.session_start_time_pacific, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.session_end_time_pacific = DateTime.(charging_sessions.session_end_time_pacific, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.charging_end_time_pacific = DateTime.(charging_sessions.charging_end_time_pacific, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.Time_of_day = Time.(charging_sessions.session_start_time_pacific)

names(charging_sessions)
show(first(charging_sessions, 1), allcols=true)



####################################################
# MPC optimization on EV charging cost minimization and peak shaving
# GMM forecast number of EVs in terms of AT and ED
# https://ieeexplore.ieee.org/document/10184283
# https://github.com/rdeits/DynamicWalking2018.jl/blob/master/notebooks/6.%20Optimization%20with%20JuMP.ipynb

# Electricity rates
# https://www.sdge.com/residential/pricing-plans/about-our-pricing-plans/whenmatters
r_energy_ur = 0.00671 # $/kWh
r_energy_summer_onpeak = 0.11957 + r_energy_ur # $/kWh
r_energy_summer_offpeak = 0.10008 + r_energy_ur # $/kWh
r_energy_winter_onpeak = 0.09955 + r_energy_ur # $/kWh
r_energy_winter_offpeak = 0.08835 + r_energy_ur # $/kWh
r_power_summer_onpeak = 9.78 + 19.14 # $/kW
r_power_winter_onpeak = 19.23 # $/kW
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

# Initialize empty variables
L_mpc = zeros(N)

# Assume everything is known and only test if MPC works
function run_mpc(data_input)
    season = get_season(data_input.session_start_time_pacific[1]) # Assume all sessions happen in the same season
    ED = data_input.total_energy_dispensed
    AT = Float64.(Dates.hour.(data_input.session_start_time_pacific))
    DT = Float64.(Dates.hour.(data_input.session_end_time_pacific))
    forecasted_n_EV = size(data_input, 1)

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
        r_energy_onpeak = r_energy_ur
        r_energy_offpeak = r_energy_ur
        r_power_onpeak = r_power_nc
    end

    # Initialize variables
    M = 1e6

    # Optimize with loop through each time slot b.c. change of objective function
    @showprogress for k in 1:N
        model = Model(Ipopt.Optimizer)
        # Set solver options
        set_optimizer_attribute(model, "max_iter", 1000)  # Set maximum number of iterations
        set_optimizer_attribute(model, "tol", 1e-6)       # Set tolerance for convergence
        set_optimizer_attribute(model, "acceptable_tol", 1e-4)  # Set acceptable tolerance
        set_optimizer_attribute(model, "print_level", 2)  # Set printing level (0: no output, 5: full output)

        @variables model begin
            P[k:N, 1:forecasted_n_EV] >= 0
            L[k:N] >= 0
            E[k:N, 1:forecasted_n_EV] >= 0
            gamma_nc_k >= 0
            gamma_onpeak_k >= 0
        end

        # Constraints
        # Constraints for P[t, i]
        @constraint(model, [t=k:N, i=1:forecasted_n_EV], 0 <= P[t, i] <= P_max)
        @constraint(model, [t=k:N, i=1:forecasted_n_EV], P[t, i] <= M * (t * delta_t >= AT[i]) + M * (t * delta_t <= DT[i]))
    
        # Initial and final energy constraints
        @constraint(model, [i=1:forecasted_n_EV], E[N, i] == ED[i])
        @constraint(model, [i=1:forecasted_n_EV], E[k, i] == 0)  # Ensure initial condition for k
    
        # Energy balance constraint with boundary check
        @constraint(model, [t=k+1:N, i=1:forecasted_n_EV], E[t, i] == E[t-1, i] + P[t, i] * delta_t)
        # @constraint(model, [t=k:N, i=1:forecasted_n_EV], E[t, i] >= 0)
    
        # Load constraints
        @constraint(model, [t=k:N], L[t] == sum(P[t, i] for i in 1:forecasted_n_EV))
        # @constraint(model, [t=k:N], L[t] >= 0)

        # Peak horizons and costs
        Index_onpeak = []
        Index_offpeak = []

        if k < T_start_idx
            Index_onpeak = T_start_idx:T_end_idx
            Index_offpeak = vcat(k:T_start_idx, T_end_idx:N)
        elseif T_start_idx <= k < T_end_idx
            Index_onpeak = k:T_end_idx
            Index_offpeak = T_end_idx:N
        elseif k >= T_end_idx
            Index_onpeak = []
            Index_offpeak = k:N
        end

        @constraint(model, [t=k:N], gamma_nc_k >= L[t])
        if !isempty(Index_onpeak)
            @constraint(model, [t in Index_onpeak], gamma_onpeak_k >= L[t])
        elseif isempty(Index_onpeak)
            @constraint(model, gamma_onpeak_k == 0)
        end
        

        # Objective function
        # Define the demand charge as an expression
        @expression(model, demand_charge_k, r_power_nc * gamma_nc_k + r_power_onpeak * gamma_onpeak_k)

        # Initialize expressions for energy charge components
        energy_offpeak_charge_k = @expression(model, sum(delta_t * r_energy_offpeak * L[t] for t in Index_offpeak))
        energy_onpeak_charge_k = @expression(model, sum(delta_t * r_energy_onpeak * L[t] for t in Index_onpeak))

        # Combine the components into the total energy charge
        @expression(model, energy_charge_k, energy_offpeak_charge_k + energy_onpeak_charge_k)


        # Define the other charges as an expression
        @expression(model, other_charge_k, 0.0578 * (demand_charge_k + energy_charge_k) +
                        (0.0058 + 0.00058 + 0.0003) * delta_t * sum(L[t] for t in k:N) +
                        0.0688 * DWR_charge)

        # Define the objective function as an expression
        @expression(model, J_k, demand_charge_k + energy_charge_k + other_charge_k)

        @objective model Min J_k

        optimize!(model)

        L_mpc[k] = value(L[k])
    end
end





####################################################
# Sample: Persistence forecast for testing
# Assume AT, DT, ED, PD are known

# 1. Load data
stations = first(unique(charging_sessions.station_name), 10)
start_date = minimum(charging_sessions.session_start_time_pacific)
end_date = start_date + Month(3)

data_mpc = filter(row -> start_date <= row.session_start_time_pacific <= end_date &&
                              row.station_name in stations, charging_sessions)
data_mpc = sort(data_mpc, [:station_name, :session_start_time_pacific])

# 2. Run MPC
run_mpc(data_mpc)

# 3. Plot results
# https://github.com/rdeits/DynamicWalking2018.jl/blob/master/notebooks/6.%20Optimization%20with%20JuMP.ipynb

p = plot(1:N, L_mpc,
         label="Perfect Forecast MPC",
         xlabel="Time Index",
         ylabel="Load (kW)",
         title="MPC Load Profile",
         size=(800, 600),  # Set the size of the plot (width, height) in pixels
         dpi=300)          # Set the DPI (dots per inch)

# Saving the plot as an image file with the specified DPI
savefig(p, "mpc_load_profile.png")

# 4. Repeat regularly with forecasted data

