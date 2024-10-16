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
packages = ["JuMP", "Ipopt", "LinearAlgebra", "Plots", "Random", "CSV", "DataFrames", "Statistics", "StatsBase", "ProgressMeter", "Dates", "DataFramesMeta", "Distributions", "JLD", "GaussianMixtures", "Holidays", "AutoMLPipeline"]

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
charging_sessions = sort(charging_sessions, [:session_start_time_la])




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
# SECTION: MPC optimization function
# TODO: create a new MPC model that includes the station information and handle several EVs charging at the station in the same day

function run_mpc(data_input::DataFrame, method::String)
    season = get_season(data_input.session_start_time_la[1]) # All daily sessions happen in the same season

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

    AT = data_input.AT
    DT = data_input.DT
    ED = data_input.ED
    PD = data_input.PD
    forecasted_n_EV = size(data_input, 1)
    AT_idx = floor.(Int, AT / T * N) .+ 1
    DT_idx = floor.(Int, DT / T * N)

    L_mpc = zeros(N)
    P_mpc = zeros(N, forecasted_n_EV)
    E_mpc = zeros(N, forecasted_n_EV)
    E_tmp = zeros(forecasted_n_EV)

    # Optimize with loop through each time slot b.c. change of objective function
    Optimal = zeros(Bool, N)

    @showprogress for k in 1:N
        # FIXME: Data updater at every time step
        if method == "KNN"

            update_time = k * delta_t
            data_arrival = data_input[(data_input.AT .<= update_time), :]
            AT = data_arrival.AT
            DT = data_arrival.DT
            ED = data_arrival.ED
            PD = data_arrival.PD
            forecasted_n_EV = size(data_arrival, 1)
            AT_idx = floor.(Int, AT / T * N) .+ 1 # shorten the PD for assumption
            DT_idx = floor.(Int, DT / T * N)
        end


        model = Model(Ipopt.Optimizer)
        # Set solver options
        set_optimizer_attribute(model, "max_iter", 1000)  # Set maximum number of iterations
        set_optimizer_attribute(model, "tol", 1e-4)       # Set tolerance for convergence
        set_optimizer_attribute(model, "acceptable_tol", 1e-3)  # Set acceptable tolerance
        set_optimizer_attribute(model, "print_level", 2)  # Set printing level (0: no output, 5: full output)

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
                if DT_idx[i] <= N-1
                    @constraint(model, [t=DT_idx[i]+1:N], P[t, i] == 0)
                end
            elseif DT_idx[i] < k
                @constraint(model, [t=k:N], P[t, i] == 0)
            end

            if AT_idx[i] > k
                @constraint(model, [t=k:AT_idx[i]-1], P[t, i] == 0)
            end

            # Initialize the energy state
            if k == 1
                @constraint(model, E[k, i] == 0)
            elseif k >=2
                @constraint(model, E[k, i] == E_tmp[i] + P[k, i] * delta_t) #E_tmp is the energy state vector stored at the previous loop of k
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
        @expression(model, energy_charge_k, delta_t * sum(r_energy_offpeak * L[t] for t in Index_offpeak) + 
                                            delta_t * sum(r_energy_onpeak * L[t] for t in Index_onpeak))

                                            

        # Define the other charges as an expression
        # FIXME: the total energy use in a month is not known and assumed to be the sum of all ED in 30 days
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
        E_tmp = E_mpc[k, :] # Update the energy state vector for the next loop

        # TODO: export the total cost and compare

        Optimal[k] = (termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED) ? 1 : 0
    end
    # print("MPC optimization is done with time: ", ceil(toc - tic), " seconds\n")
    print("Optimal Found: ", sum(Optimal), " out of ", N, "\n")

    return L_mpc, P_mpc, E_mpc
end



####################################################
# Forecast is run daily and updated with the latest data at each time slot

# Perfect forecast: AT, DT, ED, PD are known, and true data in dataset is used
# Persistence forecast: AT, DT, ED, PD are the same as the previous daily data
# KNN forecast: AT, DT, ED, PD are forecasted by KNN
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


# Forecast function
function forecast(data_input::DataFrame, method::String, updated_sessions::DataFrame)
    forecast_list = ["Perfect", "Persistence", "GMM", "Decentralized", "KNN"]
    if method ∉ forecast_list
        error("Invalid forecast method")
    elseif method == "Perfect"
        # The AT, DT, ED, PD should comply with or be more precise than the step length of MPC
        data_input.AT = ceil.(Float64.(Dates.hour.(data_input.session_start_time_la)) + Float64.(Dates.minute.(data_input.session_start_time_la)) / 60, digits=2)
        data_input.DT = ceil.(Float64.(Dates.hour.(data_input.session_end_time_la)) + Float64.(Dates.minute.(data_input.session_end_time_la)) / 60, digits=2)
        data_input.ED = data_input.total_energy_dispensed
        data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)
    elseif method == "KNN"
        data_input.AT = zeros(nrow(data_input))
        data_input.DT = zeros(nrow(data_input))
        data_input.ED = zeros(nrow(data_input))
        data_input.PD = zeros(nrow(data_input))
        # FIXME: N_EV should be updated?

        target_time = data_input.session_start_time_la[1]
        target_day_type = datetype(target_time)
        same_type_sessions = updated_sessions[(updated_sessions.type .== target_day_type) .& (Date.(updated_sessions.session_start_time_la) .< Date(target_time)), :]
        if nrow(same_type_sessions) == 0
            error("No valid sessions found for the same day type")
        end

        closest_idx = argmin(abs.(Date.(same_type_sessions.session_start_time_la) .- Date(target_time)))
        closest_date = Date(same_type_sessions.session_start_time_la[closest_idx])
        persistence_sessions = same_type_sessions[Date.(same_type_sessions.session_start_time_la) .== closest_date, :]







        for i in 1:nrow(data_input)
            target_time = data_input.session_start_time_la[i]
            last_day_sessions = updated_sessions[Date.(updated_sessions.session_start_time_la) .== Date(target_time) - Day(1), :]
            
            if nrow(last_day_sessions) == 0
                error("No valid sessions found for the last day")
            end
            
            time_diffs = abs.(Time.(last_day_sessions.session_start_time_la) .- Time(target_time))
            closest_idx = argmin(time_diffs)
            data_last_day_closest = last_day_sessions[closest_idx, :]

            data_input.AT[i] = ceil(Float64(Dates.hour(data_last_day_closest.session_start_time_la)) + Float64(Dates.minute(data_last_day_closest.session_start_time_la)) / 60, digits=2)
            data_input.DT[i] = ceil(Float64(Dates.hour(data_last_day_closest.session_end_time_la)) + Float64(Dates.minute(data_last_day_closest.session_end_time_la)) / 60, digits=2)
            data_input.ED[i] = data_last_day_closest.total_energy_dispensed
            data_input.PD[i] = ceil(Dates.value(data_last_day_closest.charging_end_time_la - data_last_day_closest.session_start_time_la) / (3600*1000), digits=2)

            # data_input.forecasted_n_EV[i] = i
        end
    elseif method == "Persistence"
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
        data_input.DT = ceil.(Float64.(Dates.hour.(data_input.session_end_time_la)) + Float64.(Dates.minute.(data_input.session_end_time_la)) / 60, digits=2)
        data_input.ED = data_input.total_energy_dispensed
        data_input.PD = ceil.(Dates.value.(data_input.charging_end_time_la - data_input.session_start_time_la) / (3600*1000), digits=2)

    elseif method == "GMM"
        data_input.AT, data_input.DT, data_input.ED, data_input.PD = predict_gmm(data_input) 

    elseif method == "Decentralized" # TODO: Talk with Avik
        data_input.AT = zeros(nrow(data_input))
    end

    return data_input
end

# V0G for comparison
function run_V0G(data_input::DataFrame)
    L_V0G = zeros(N)
    P_V0G = zeros(N, size(data_input, 1))
    data_input.AT = ceil.(Float64.(Dates.hour.(data_input.session_start_time_la)) + Float64.(Dates.minute.(data_input.session_start_time_la)) / 60, digits=2)
    data_input.DT = ceil.(Float64.(Dates.hour.(data_input.session_end_time_la)) + Float64.(Dates.minute.(data_input.session_end_time_la)) / 60, digits=2)
    data_input.ED = data_input.total_energy_dispensed

    for k in 1:N
        for i in 1:size(data_input, 1)
            Energy_dispensed = sum(P_V0G[1:k, i]) * delta_t
            # Assume more energy can be dispensed than the actual data
            if data_input.AT[i] <= k * delta_t && k * delta_t <= data_input.DT[i] && Energy_dispensed < data_input.ED[i]
                P_V0G[k, i] = P_max
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
    updated_sessions = charging_sessions
    updated_sessions.type = datetype.(updated_sessions.session_start_time_la)
    L_mpc_dict = Dict{Date, Any}()
    P_mpc_dict = Dict{Date, Any}()
    E_mpc_dict = Dict{Date, Any}()
    L_V0G_dict = Dict{Date, Any}()

    tic = time()
    for day in days
        data_tmp = filter(row -> Dates.Date(row.session_start_time_la) == day, data_input)
        L_V0G_dict[day] = run_V0G(data_tmp)
        data_tmp = forecast(data_tmp, method, updated_sessions)
        new_sessions = select(data_tmp, Not([:AT, :DT, :ED, :PD]))
        new_sessions.type = datetype.(new_sessions.session_start_time_la)
        updated_sessions = vcat(updated_sessions, new_sessions)
        L_mpc_tmp, P_mpc_tmp, E_mpc_tmp = run_mpc(data_tmp, method)

        println("Day: ", day, " L_mpc_tmp size: ", size(L_mpc_tmp))
        println("Day: ", day, " P_mpc_tmp size: ", size(P_mpc_tmp))
        println("Day: ", day, " E_mpc_tmp size: ", size(E_mpc_tmp))
        L_mpc_dict[day] = L_mpc_tmp
        P_mpc_dict[day] = P_mpc_tmp
        E_mpc_dict[day] = E_mpc_tmp

    end
    toc = time()
    print("Daily update is done with time: ", ceil(toc - tic), " seconds\n")
    
    return L_V0G_dict, L_mpc_dict, P_mpc_dict, E_mpc_dict, days
end





####################################################
# SECTION: Testing
# https://github.com/rdeits/DynamicWalking2018.jl/blob/master/notebooks/6.%20Optimization%20with%20JuMP.ipynb
# Test data is unknown with all other data known
test_sessions = CSV.read("clean_test_sessions.csv", DataFrame)
test_sessions = sort(test_sessions, [:session_start_time_la])
start_date = Date(minimum(test_sessions.session_start_time_la))
end_date = start_date + Day(2)
data_test = filter(row -> start_date <= row.session_start_time_la <= end_date, test_sessions)
data_test = sort(data_test, [:session_start_time_la])

# TODO: For testing
data_input = data_test
method = "Persistence"

# TODO: select method from ["Perfect", "Persistence", "GMM", "Decentralized", "KNN"]
L_V0G_all, L_mpc_perfect, P_mpc_perfect, E_mpc_perfect, days = daily_update(data_test, "Perfect")
L_V0G_all, L_mpc_persistence, P_mpc_persistence, E_mpc_persistence, days = daily_update(data_test, "Persistence")



####################################################
# SECTION: Plot the test results in time order

hist_AT = histogram(data_test.session_start_time_la, bins = length(days) * N, alpha=0.5, label="AT", xlabel="AT", ylabel="Sessions")

hist_DT = histogram(data_test.session_end_time_la, bins = length(days) * N, alpha=0.5, label="DT", xlabel="DT", ylabel="Sessions")

df_load_perfect = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], load = vcat(values(L_mpc_perfect)...))
df_load_persistence = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], load = vcat(values(L_mpc_persistence)...))
df_load_V0G  = DataFrame(time = [DateTime(minimum(days)) + Minute(15 * (i - 1)) for i in 1:N * length(days)], load = vec(vcat(values(L_V0G_all)...)))

p_load_combined = plot(df_load_perfect.time, df_load_perfect.load,
    label="Perfect Forecast MPC",
    xlabel="Timestamp",
    ylabel="Load (kW)",
    size=(800, 600),
    dpi=300,
    legend=:topright)

# Add Persistence Forecast load to the same plot
plot!(df_load_persistence.time, df_load_persistence.load,
    label="Persistence Forecast MPC")

# Add V0G load to the same plot
plot!(df_load_V0G.time, df_load_V0G.load,
    label="V0G")   

p = plot(hist_AT, hist_DT, p_load_combined, layout=(3, 1), size=(1000, 800))

savefig(p, "Load_All.png")


