"""
Script to run simulations with different initial conditions and plot the
resulting survival curves against the validation data.
"""

using DifferentialEquations
using JumpProcesses
using Plots
# using PlotlyJS
using Statistics
using Distributions
using Random
using JLD2

using Optimization
using OptimizationOptimJL
# using BlackBoxOptim

using CSV
using DataFrames
using DelimitedFiles

# include own module with functionalities, ATTENTION, this will include StatsPlots
include("functionalities.jl")

include("general_MNR_simulation_algorithm.jl")

# get exponential tumour growth
function expgrowth!(du, u, p, t)
    du[1] = p[1]* u[1]
end

function TumorPath(S0, p; endtime=30.0)
    prob = ODEProblem(expgrowth!, [S0], (0.0, endtime), [p.beta])
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sol
end

group = "pT4"

nsim = 200

npat = 2000
endtime = 30.0

#set parameter values
beta = 0.01

m_sigma = 6.0
m_order = 1.0

d_size = 0.011
d_metastasis = 0.31

# set initial tumour size
if group == "pT1"
    S0 = 10
elseif group == "pT2"
    S0 = 35
elseif group == "pT3"
    S0 = 75
elseif group =="pT4"
    S0 = 120
end

# set parameter dict
p = (beta = beta, m_sigma = m_sigma, m_order = m_order, d_size = d_size, d_metastasis = d_metastasis)


summary_dict = Dict()
os_fits = []
met_fits = []

Threads.@threads for n in 1:nsim
    data = simulate_many_MNR(p, TumorPath, npat=npat, S0=S0, metastatic_model="cell_division", endtime = endtime)
    os_fit, met_fit = plot_survival_curves(data, display=false)
    summary_dict["Run$n"] = Dict("data" => data, "os_fit" => os_fit, "met_fit" => met_fit)
    push!(os_fits, os_fit)
    push!(met_fits, met_fit)
end

# save summary dict with jld2
save("output/validation_model_$(group)_$(nsim)sim.jld2", "summary_dict", summary_dict)

# get validation data and plot mean 
survival_data = load("data/validation_data/MET_survival/survival_data_dict.jld2")["survival_data"];

common_times, mean_survival, ci_lower, ci_upper = mean_survival_curve(os_fits, survival_data, "OS", group)
os_survival_dict = Dict("times" => common_times, "mean" => mean_survival, "lower ci" => ci_lower, "upper ci" => ci_upper)
save("output/validation_model_$(group)_os_survival.jld2", "os_survival", os_survival_dict)


common_times, mean_survival, ci_lower, ci_upper = mean_survival_curve(met_fits, survival_data, "MET", group)
met_survival_dict = Dict("times" => common_times, "mean" => mean_survival, "lower ci" => ci_lower, "upper ci" => ci_upper)
save("output/validation_model_$(group)_met_survival.jld2", "met_survival", met_survival_dict)
