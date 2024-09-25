using DifferentialEquations
using JumpProcesses
using Plots
using LatinHypercubeSampling
using Statistics
using Distributions
using Random
using JLD2
using JSON

using Optimization
using OptimizationOptimJL
# using BlackBoxOptim

using CSV
using DataFrames
using DelimitedFiles

using HDF5
using MCMCChains
using StatsPlots

# set main directory name
dir = dirname(pwd());

# set seed for reproducibility
Random.seed!(123);

# set model specifications for optimization
MODEL = "treated"
LLH_TYPE = "Analytic"
optimizer = "LBFGS"	
OPTIM_TYPE = "joined"
DATA_TYPE= "regular"

# set initial tumour size
S0 = 0.065

# set true parameter values from data generation
beta = 0.5
m_basal = 0.01
m_size = 0.003

m_sigma = 0.03
m_order = 1.0

d_size = 0.001
d_metastasis = 0.002
θ = [beta, m_basal, m_size, m_sigma, m_order, d_size, d_metastasis]

# set parameter names and bounds for startpoint sampling
if MODEL == "proportional"
    par_names = ["beta", "m_basal", "m_size", "d_size", "d_metastasis"]
    lb_x0 = [0.48, 1e-3, 1e-3, 1e-4, 1e-4]
    ub_x0 = [0.52, 0.1, 0.1, 0.01, 0.01]
    log_lb_x0 = [-0.75, -7, -7, -9, -9]
    log_ub_x0 = [-0.65, -2, -2, -4, -4]
    include(joinpath(dir, "src/ProportionalLikelihoods.jl"))
    import .ProportionalLikelihoods as Llh
elseif MODEL == "cell_division"
    par_names = ["beta", "m_sigma", "d_size", "d_metastasis"]
    lb_x0 = [0.48, 0.01, 1e-4, 1e-4]
    ub_x0 = [0.52, 0.1, 0.01, 0.01]
    log_lb_x0 = [-0.75, -4, -9, -9]
    log_ub_x0 = [-0.65, -2, -4, -4]
    include(joinpath(dir, "src/CellDivisionLikelihoods.jl"))
    import .CellDivisionLikelihoods as Llh
elseif MODEL =="gompertz"
    K = 150000
    α = 0.2
    m_size = 0.002
    d_size = 0.0003
    d_metastasis = 0.003
    θ = [K, α, m_basal, m_size, d_size, d_metastasis]
    par_names = ["K", "alpha", "m_basal", "m_size", "d_size", "d_metastasis"]
    lb_x0 = [130000, 0.1, 1e-3, 1e-4, 1e-4, 1e-4]
    ub_x0 = [170000, 0.3, 0.1, 0.01, 0.001, 0.01]
    log_lb_x0 = [11.8, -2, -7, -9, -9, -9]
    log_ub_x0 = [12, -1, -2, -4, -6, -4]
    include(joinpath(dir, "src/GompertzLikelihoods.jl"))
    import .GompertzLikelihoods as Llh
elseif MODEL =="treated"
    par_names = ["beta0", "rho", "delta", "m_basal", "m_size", "d_size", "d_metastasis"]
    beta0 = 0.5
    rho = 0.45
    delta = 0.2
    m_basal = 0.01
    m_size = 0.003

    m_sigma = 0.03
    m_order = 1.0

    d_size = 0.001
    d_metastasis = 0.002
    θ = [beta0, rho, delta, m_basal, m_size, d_size, d_metastasis]
    lb_x0 = [0.48, 0.3, 0.15, 1e-4, 1e-4, 0.01]
    ub_x0 = [0.52, 0.5, 0.25, 0.01, 0.01, 0.1]
    log_lb_x0 = [-0.75, -0.85, -2.3, -9, -9, -8, -8]
    log_ub_x0 = [-0.65, -0.75, -1.2, -4, -4, -5, -5]
    
    # load correct likelihood script
    include(joinpath(dir, "src/TherapyEffectLikelihoods.jl"))
    import .TherapyEffectLikelihoods as Llh
else  
    error("Model not implemented.")
end



# load data
npat = 500
if DATA_TYPE== "sparse"
    if MODEL !="proportional"
        error("Sparse data only available for proportional model.")
    end
    data_path = joinpath(dir,"data/$(MODEL)_sparse_data_$(npat)_patients_$(θ).jld2")
    patient_data = load(data_path)["sparse_$(MODEL)_data"]
else
    npat = 500
    data_path = joinpath(dir,"data/$(MODEL)_data_$(npat)_patients_$(θ).jld2")
    patient_data = load(data_path)["$(MODEL)_data"]
end


# perform multi-start optimization 
n_starts = 100

# set up dictionary for results storing
results_list = Array{Any}(undef, n_starts)

# sample startpoints using LatinHypercubeSampling
plan,_ = LHCoptim(n_starts, length(log_lb_x0), 1000)
scaled_plan = scaleLHC(plan, [(log_lb_x0[i], log_ub_x0[i]) for i in eachindex(log_lb_x0)])

x0_vectors = load(joinpath(dir, "data/startpoints_$(MODEL)_likelihoods_100_starts_$(npat)_patients_$(θ).jld2"))["startpoints"];

# set up optimization problem using threaded for loops

Threads.@threads for i in 1:n_starts

    # sample startpoint using LatinHypercubeSampling
    x0 = x0_vectors[i]
    
    # run optimization
    if OPTIM_TYPE == "joined"
        res_dict = Llh.LogOptimization(patient_data, x0, lb=log_lb_x0, ub=log_ub_x0, optimizer=optimizer, llh_type= LLH_TYPE)
    elseif OPTIM_TYPE == "hierarchical"
        par = LogHierarchOptimization(patient_data, x0, lb=log_lb_x0, ub=log_ub_x0)
        # calcualte resulting nllh
        nllh = NegLogLikelihood(exp.(par), patient_data, S0=S0)
        # create dictionary for storing results
        res_dict = Dict(
            "nllh" => nllh,
            "parameter" => par
        )
    end
    results_list[i] = res_dict
end

result = Dict("optimization_results" => results_list, "par_names" => par_names)

# save results
if DATA_TYPE== "sparse"
    save_path = joinpath(dir, "output/$(MODEL)_$(OPTIM_TYPE)_optimization_$(n_starts)_starts_$(optimizer)_sparse_data_$(LLH_TYPE).jld2")
else
    save_path = joinpath(dir, "output/$(MODEL)_$(OPTIM_TYPE)_optimization_$(n_starts)_starts_$(optimizer)_$(LLH_TYPE).jld2")
end

save(save_path, "results", result)

