using Distributed


# instantiate and precompile environment in all processes
@everywhere begin
    using Pkg; Pkg.activate(joinpath(pwd()))
    Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin

    using DifferentialEquations
    using JumpProcesses
    using Plots
    using LatinHypercubeSampling
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

    # using HDF5
    # using MCMCChains
    # using StatsPlots

    # set seed for reproducibility
    Random.seed!(123);

    # set model specifications for optimization
    MODEL = "treated"
    LLH_TYPE = "Analytic"
    optimizer = "LBFGS"	
    OPTIM_TYPE = "joined"

    # set initial tumour size
    S0 = 0.065

    # set parameter names and bounds for startpoint sampling
    if MODEL == "proportional" || MODEL == "sparse_proportional"
        par_names = ["beta", "m_basal", "m_size", "d_size", "d_metastasis"]
        lb_x0 = [0.48, 1e-3, 1e-3, 1e-4, 1e-4]
        ub_x0 = [0.52, 0.1, 0.1, 0.01, 0.01]
        log_lb_x0 = [-0.75, -7, -7, -9, -9]
        log_ub_x0 = [-0.65, -2, -2, -4, -4]
        # load correct likelihood script
        include("ProportionalLikelihoods.jl")
        import .ProportionalLikelihoods as Llh
    elseif MODEL == "cell_division" || MODEL == "sparse_cell_division"
        par_names = ["beta", "m_sigma", "d_size", "d_metastasis"]
        lb_x0 = [0.48, 0.01, 1e-4, 1e-4]
        ub_x0 = [0.52, 0.1, 0.01, 0.01]
        log_lb_x0 = [-0.75, -7, -9, -9]
        log_ub_x0 = [-0.65, -2, -4, -4]
        # load correct likelihood script
        include("CellDivisionLikelihoods.jl")
        import .CellDivisionLikelihoods as Llh
    elseif MODEL =="gompertz" || MODEL == "sparse_gompertz"
        K = 150000
        α = 0.2
        m_size = 0.002
        d_size = 0.0003
        d_metastasis = 0.003
        θ = [K, α, m_basal, m_size, d_size, d_metastasis]
        par_names = ["K", "alpha", "m_basal", "m_size", "d_size", "d_metastasis"]
        lb_x0 = [140000, 0.18, 1e-3, 1e-4, 1e-4, 1e-4]
        ub_x0 = [160000, 0.22, 0.1, 0.01, 0.001, 0.01]
        log_lb_x0 = [11.849, -1.75, -7, -9, -9, -9]
        log_ub_x0 = [11.983, -1.5, -2, -4, -6, -4]
        # load correct likelihood script
        include("GompertzLikelihoods.jl")
        import .GompertzLikelihoods as Llh
    elseif MODEL =="treated"
        par_names = ["beta0", "rho", "delta", "m_basal", "m_size", "d_size", "d_metastasis"]
        beta0 = 0.5
        rho = 0.1
        delta = 2
        m_basal = 0.01
        m_size = 0.003

        m_sigma = 0.03
        m_order = 1.0

        d_size = 0.001
        d_metastasis = 0.002
        θ = [beta0, rho, delta, m_basal, m_size, d_size, d_metastasis]
        lb_x0 = [0.48, 0.05, 1.5, 1e-4, 1e-4, 0.01]
        ub_x0 = [0.52, 0.15, 2.5, 0.01, 0.01, 0.1]
        log_lb_x0 = [-0.75, -2.8, 0.6, -9, -9, -8, -8]
        log_ub_x0 = [-0.65, -2, 0.8, -4, -4, -5, -5]
        # load correct likelihood script
        include("TherapyEffectLikelihoods.jl")
        import .TherapyEffectLikelihoods as Llh
    else  
        error("Model not implemented. Choose between 'proportional', 'cell_division', 'gompertz' or 'treated'.")
    end

    # load data
    npat = 500
    data_path = joinpath(pwd(),"data/simplified_model/$(MODEL)_data_$(npat)_patients_$(θ).jld2")
    patient_data = load(data_path)["$(MODEL)_data"];

    # Slurm Job-array
    n_starts = 100 # here just for sampling sufficiently many startpoints, real number of start is determiend by number of workers

    task_id = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
    # task_id = parse(Int64, task_id_str)

    # load startpoints
    x0_vectors = load(joinpath(pwd(), "data/startpoints_$(MODEL)_likelihoods_100_starts_$(npat)_patients_$(θ).jld2"))["startpoints"];
end

    # stuff needed on workers only
@everywhere workers() begin

    function run_single_optimization(x0; optimizer="SAMIN")        
        res_dict = Llh.LogOptimization(patient_data, x0, lb=log_lb_x0, ub=log_ub_x0, optimizer=optimizer, llh_type = LLH_TYPE)
        return res_dict
    end
end

jobs = [@spawnat(i, @timed(run_single_optimization(x0_vectors[i], optimizer=optimizer))) for i in 1:nworkers()]

# fetch results
all_results = [fetch(job) for job in jobs]
typeof.(all_results)

if any(typeof.(all_results) .== RemoteException)
    println("Worker $(findfirst(typeof.(all_results) .== RemoteException)) failed.")
end

any(all_results .== RemoteException)

result = Dict("optimization_results" => all_results, "par_names" => par_names)

# get number of workers
n_workers = nworkers()

# save results
save_path = joinpath(pwd(), "output/optimization/$(MODEL)_$(OPTIM_TYPE)_optimization_$(n_workers)_starts_$(optimizer)_distributed_$(LLH_TYPE).jld2")
# save_path = joinpath(pwd(), "output/results/cell_division_optimization_proportional_data_$(n_workers)_starts_$(optimizer)_distributed.jld2")
save(save_path, result)






