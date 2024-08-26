#=
Author: Yizhan Gu
Date: 2024-07-16
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
packages = ["JuMP", "Ipopt", "LinearAlgebra", "Plots", "Random", "CSV", "DataFrames", "Statistics", "StatsBase", "JLD", "Dates", "DataFramesMeta", "GaussianMixtures", "Distributions", "KernelDensity"]

for package in packages
    Pkg.add(package)
end # Install the necessary packages

for package in packages
    eval(Meta.parse("using $package"))
end
print("All packages are successfully loaded")

# Clear console
print("\033c") # Or REPL: Ctrl + L
# Note: to clear workspace, use Ctrl + D

####################################################
# Sample code for JuMP optimization with Ipopt
function test_jump()
    m = Model(Ipopt.Optimizer)
    @variable(m, x >= 0)
    @variable(m, y >= 0)
    @constraint(m, x + y == 1)
    @objective(m, Min, x^2 + y^2)
    optimize!(m)

    println("Objective value: ", JuMP.objective_value(m))
    println("x = ", JuMP.value(x))
    println("y = ", JuMP.value(y))
end

test_jump()

####################################################
# Data preprocessing

#= Importing the necessary files
charging_sessions = CSV.read("/Users/admin/Desktop/EV_program/2023Fall_TotalEnergies/data_sessions.csv", DataFrame)

charging_sessions.session_start_time_pacific = DateTime.(charging_sessions.session_start_time_pacific, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.session_end_time_pacific = DateTime.(charging_sessions.session_end_time_pacific, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.charging_end_time_pacific = DateTime.(charging_sessions.charging_end_time_pacific, dateformat"yyyy-mm-ddTHH:MM:SSZ")
charging_sessions.Time_of_day = Time.(charging_sessions.session_start_time_pacific)
=#

charging_sessions = CSV.read("clean_charging_sessions.csv", DataFrame)

names(charging_sessions)
show(first(charging_sessions, 1), allcols=true)

#= Convert datetime to Float64
data_AT = Float64.(Dates.value.(charging_sessions.session_start_time_pacific))
data_DT = Float64.(Dates.value.(charging_sessions.session_end_time_pacific))
data_PD = Float64.(Dates.value.(charging_sessions.PD))
=#
data_ED = charging_sessions.total_energy_dispensed


# TODO: Select time resolution manually
flag = "hour"
# flag = "minute"

if flag == "hour"
    charging_sessions.AT = Float64.(Dates.hour.(charging_sessions.session_start_time_pacific))
    charging_sessions.DT = Float64.(Dates.hour.(charging_sessions.session_end_time_pacific))
    charging_sessions.PD = ceil.(Dates.value.(charging_sessions.charging_end_time_pacific - charging_sessions.session_start_time_pacific) / (3600*1000), digits=0)
    # charging_sessions.PD = Float64.(Int64.(charging_sessions.PD))
elseif flag == "minute"
    charging_sessions.AT = Float64.(Dates.hour.(charging_sessions.session_start_time_pacific)) * 60 + Float64.(Dates.minute.(charging_sessions.session_start_time_pacific))
    charging_sessions.DT = Float64.(Dates.hour.(charging_sessions.session_end_time_pacific)) * 60 + Float64.(Dates.minute.(charging_sessions.session_end_time_pacific))
    charging_sessions.PD = ceil.(Dates.value.(charging_sessions.charging_end_time_pacific - charging_sessions.session_start_time_pacific) / (60*1000), digits=0)
    # charging_sessions.PD = Float64.(Int64.(charging_sessions.PD))
end


data_AT = charging_sessions.AT
data_DT = charging_sessions.DT
data_PD = charging_sessions.PD

####################################################
# GMM density modeling
# https://juliapackages.com/p/gaussianmixtures

# Try different number of GMM components
components = [1, 2, 3, 4, 5]
gmm_AT = Dict()
gmm_DT = Dict()
gmm_PD = Dict()
gmm_ED = Dict()

seed = 107
Random.seed!(seed)
for cp in components
    gmm_AT[cp] = GMM(cp, data_AT)
    gmm_DT[cp] = GMM(cp, data_DT)
    gmm_PD[cp] = GMM(cp, data_PD)
    gmm_ED[cp] = GMM(cp, data_ED)
end

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

# loop of GMM models
gmm_data = [data_AT, data_DT, data_PD, data_ED]
gmm_models = [gmm_AT, gmm_DT, gmm_PD, gmm_ED]
column_names = ["Arrival Time", "Departure Time", "Plug-in Duration", "Energy Demand"]

####################################################
# Error Metrics for GMM modeling

# 1. Mean Absolute Error (MAE)
# 2. Mean Squared Error (MSE)
# 3. Root Mean Squared Error (RMSE)
# 4. R-squared (R2)

function mae(y_true, y_pred)
    return mean(abs.(y_true - y_pred))
end

function mse(y_true, y_pred)
    return mean((y_true - y_pred).^2)
end

function rmse(y_true, y_pred)
    return sqrt(mse(y_true, y_pred))
end

function r2(y_true, y_pred)
    y_mean = mean(y_true)
    ss_tot = sum((y_true .- y_mean).^2)
    ss_res = sum((y_true .- y_pred).^2)
    return 1 - ss_res / ss_tot
end

# p_tmp = Vector{Plots.Plot}(undef, 4)
plots = []
error_metrics = Dict()
for i in 1:4
    data = gmm_data[i]
    gmm_dict = gmm_models[i]
    column_name = column_names[i]
    error_metrics[column_name] = Dict()
    x_range = range(minimum(data), stop=maximum(data), length=1000)

    # Plot histogram
    hist = histogram(data, bins=30, norm=:pdf, alpha=0.5, label="Original Data", xlabel="Value", ylabel="Density", title="Original and GMM Modeling: $column_name")
    # Compute the histogram using StatsBase
    hist_obj = fit(Histogram, data, nbins=30)

    # Extract bin edges and compute bin centers
    bin_edges = hist_obj.edges[1]
    bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

    # Extract histogram values (density)
    hist_values = hist_obj.weights
    bin_widths = diff(bin_edges)
    normalized_hist_values = hist_values ./ (sum(hist_values) * bin_widths)

    #= Compute the true density using kernel density estimation (Wrong for comparison between 2 models instead of model and truth)
    kde_data = kde(data)
    true_density = [pdf(kde_data, x) for x in x_range]
    plot!(hist, x_range, true_density, label="True Density", linewidth=1, alpha=1, color=:black)
    =#
    
    for cp in components
        # Plot GMM density
        gmm = gmm_dict[cp]
        y_density = [gmm_pdf(gmm, x) for x in x_range]
        gmm_bin = [gmm_pdf(gmm, x) for x in bin_centers]
        plot!(hist, x_range, y_density, label="GMM (k=$cp)", linewidth=1, alpha=1)
        plot!(hist, bin_centers, gmm_bin, seriestype = :scatter, label="", markersize=0.8, alpha=0.4)

        # Compute error metrics
        error_metrics[column_name][cp] = Dict()
        error_metrics[column_name][cp]["MAE"] = mae(normalized_hist_values, gmm_bin)
        error_metrics[column_name][cp]["MSE"] = mse(normalized_hist_values, gmm_bin)
        error_metrics[column_name][cp]["RMSE"] = rmse(normalized_hist_values, gmm_bin)
        error_metrics[column_name][cp]["R2"] = r2(normalized_hist_values, gmm_bin)
    end
    
    push!(plots, hist)
    # p_tmp[i] = plot(histogram(data, bins=30, normalize=:pdf, alpha=0.5, label="Original Data", xlabel="Value", ylabel="Density", title="Original and GMM Modeling: $column_name"), x_range, pdf_values, label="GMM Density", color=:red, xticks = round.(range(minimum(data),stop=maximum(data),length = 4),digits=2))
end

plot(plots..., layout=(2, 2), size=(1200, 800))
# plot(p_tmp..., layout=(2, 2), size=(1200, 800))
savefig("gmm_plots_$flag.pdf")


# Convert error metrics to DataFrame
error_df = DataFrame()
for (column_name, metrics) in error_metrics
    for (cp, metric_values) in metrics
        row = DataFrame(
            Column = column_name,
            Component = cp,
            MAE = metric_values["MAE"],
            MSE = metric_values["MSE"],
            RMSE = metric_values["RMSE"],
            R2 = metric_values["R2"]
        )
        append!(error_df, row)
    end
end

# For each column, we pick the model with the lowest average of normalized MAE, MSE, RMSE, and the highest R2
#= Normalize columns (Not appropriate for different types of error metrics)
function normalize_column(col)
    min_val = minimum(col)
    max_val = maximum(col)
    return (col .- min_val) ./ (max_val - min_val)
end
=#

# Clean the DataFrame
clean_index = findall(row -> !any(isnan, row), eachrow(error_df[:, 3:end]))
# filtered_df = filter(row -> !any(isnan, row), eachrow(error_df[:, 3:end]))
error_df = error_df[clean_index, :]

# Group by `Column` and rank by average of error metrics
error_df_rank = copy(error_df)
error_df_rank = groupby(error_df_rank, :Column)
error_df_rank = combine(error_df_rank) do group
    group.MAE_rank = sortperm(group.MAE)
    group.MSE_rank = sortperm(group.MSE)
    group.RMSE_rank = sortperm(group.RMSE)
    group.R2_rank = sortperm(group.R2, rev=true)
    return group
end

# Generate the 'grade' column by summing the ranks
error_df_rank.grade = error_df_rank.MAE_rank + error_df_rank.MSE_rank + error_df_rank.RMSE_rank + error_df_rank.R2_rank
error_df_rank = sort(error_df_rank, [:Column, :grade], rev=[false, false])
CSV.write("Error_metrics.csv", error_df_rank)

# Group by `Column` and pick the component with the lowest grade within each group
# error_df_rank = CSV.read("Error_metrics.csv", DataFrame)
best_cp = groupby(error_df_rank, :Column)
best_cp = combine(best_cp) do group
    best_idx = argmin(group.grade)
    return group[best_idx, :]
end

# ❗️Be aware of the sequence of the components!
best_gmm_AT = gmm_AT[best_cp.Component[1]]
best_gmm_DT = gmm_DT[best_cp.Component[2]]
best_gmm_ED = gmm_PD[best_cp.Component[3]]
best_gmm_PD = gmm_ED[best_cp.Component[4]]

# Save the best GMM models
# save(filename::String, name::String, gmm::GMM)
save("best_gmms.jld", "models", [best_gmm_AT, best_gmm_DT, best_gmm_ED, best_gmm_PD])
