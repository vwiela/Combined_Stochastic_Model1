using DifferentialEquations
using JumpProcesses
using Plots
# using PlotlyJS
using Statistics
using Distributions
using Random
using JLD2
using HDF5

using Optimization
using OptimizationOptimJL
# using BlackBoxOptim

using CSV
using DataFrames
using DelimitedFiles

using BenchmarkTools
using LatinHypercubeSampling

# set seed for reproducibility
Random.seed!(123);

# set parameters
# set true parameter values from data generation
beta = 0.5
m_basal = 0.01
m_size = 0.003

m_sigma = 0.03
m_order = 1.0

d_size = 0.001
d_metastasis = 0.002
θ = [beta, m_basal, m_size, m_sigma, m_order, d_size, d_metastasis]

# frst the proportional Model
include("ProportionalLikelihoods.jl");
import .ProportionalLikelihoods as PL

# load data
npat = 500
data_path = joinpath(pwd(),"data/simplified_model/proportional_data_$(npat)_patients_$(θ).jld2")
proportional_data = load(data_path)["proportional_data"];

log_lb_x0 = [-0.75, -7, -7, -9, -9]
log_ub_x0 = [-0.65, -2, -2, -4, -4]

plan,_ = LHCoptim(100, length(log_lb_x0), 1000)
scaled_plan = scaleLHC(plan, [(log_lb_x0[i], log_ub_x0[i]) for i in eachindex(log_lb_x0)]);

time_benchmark_df = DataFrame(x =[], Analytic_Value=[], Analytic_Time = [], Numeric_Value=[], Numeric_Time=[], FG_Value=[], FG_Time=[])

for i in 1:100
    global x = exp.(scaled_plan[i, :])
    ana_val = PL.NegLogLikelihood(x, proportional_data, S0 = 0.065)
    ana_time = median(@benchmark PL.NegLogLikelihood(x, proportional_data, S0 = 0.065))
    num_val = PL.NumericNegLogLikelihood(x, proportional_data, S0 = 0.065)
    fg_val = PL.FGNegLogLikelihood(x, proportional_data, S0 = 0.065)
    num_time = median(@benchmark PL.NumericNegLogLikelihood(x, proportional_data, S0 = 0.065))
    fg_time = median(@benchmark PL.FGNegLogLikelihood(x, proportional_data, S0 = 0.065))
    push!(time_benchmark_df, [x, ana_val, ana_time, num_val, num_time, fg_val, fg_time])
end


# save time_benchmark_df
CSV.write("output/results/benchmark_proportional_likelihoods_$(npat)_patients_$(θ).csv", time_benchmark_df)

# next the cell division model
include("CellDivisionLikelihoods.jl")
import .CellDivisionLikelihoods as CDL

# load data
npat = 500
data_path = joinpath(pwd(),"data/simplified_model/cell_division_data_$(npat)_patients_$(θ).jld2")
cell_division_data = load(data_path)["cell_division_data"];


log_lb_x0 = [-0.75, -4, 0.0, -9, -9]
log_ub_x0 = [-0.65, -2, 0.0, -4, -4]

plan,_ = LHCoptim(100, length(log_lb_x0), 1000)
scaled_plan = scaleLHC(plan, [(log_lb_x0[i], log_ub_x0[i]) for i in eachindex(log_lb_x0)]);

time_benchmark_df = DataFrame(x =[], Analytic_Value=[], Analytic_Time = [], Numeric_Value=[], Numeric_Time=[], FG_Value=[], FG_Time=[])

for i in 1:100
    global x = exp.(scaled_plan[i, :])
    ana_val = CDL.NegLogLikelihood(x, cell_division_data, S0 = 0.065)
    ana_time = median(@benchmark CDL.NegLogLikelihood(x, cell_division_data, S0 = 0.065))
    num_val = CDL.NumericNegLogLikelihood(x, cell_division_data, S0 = 0.065)
    fg_val = CDL.FGNegLogLikelihood(x, cell_division_data, S0 = 0.065)
    num_time = median(@benchmark CDL.NumericNegLogLikelihood(x, cell_division_data, S0 = 0.065))
    fg_time = median(@benchmark CDL.FGNegLogLikelihood(x, cell_division_data, S0 = 0.065))
    push!(time_benchmark_df, [x, ana_val, ana_time, num_val, num_time, fg_val, fg_time])
end

# save time_benchmark_df
CSV.write("output/results/benchmark_cell_division_likelihoods_$(npat)_patients_$(θ).csv", time_benchmark_df)