using Distributed # package for distributed computing in julia


# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(dirname(pwd()))
  Pkg.instantiate(); Pkg.precompile()
end

# sutff needed on workers and mcmc_chain
@everywhere begin
    using DifferentialEquations
    using JumpProcesses
    using Plots
    using Statistics
    using Distributions

    using CSV
    using DataFrames
    using DelimitedFiles
    using HDF5
    using JLD2add

    using Optimization
    using OptimizationOptimJL

    using MCMCChains
    using MCMCChainsStorage
    using AdvancedMH
    using Turing
    using AbstractMCMC
    using StatsPlots


    # set model specifications
    MODEL = "treated"
    LLH_TYPE = "Analytic"
    optimizer = "LBFGS"	
    OPTIM_TYPE = "joined"
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
end

# stuff only needed on workers
@everywhere workers() begin

    # settings for the sampler
    global niter = 1000000

    # get MLE as starting point
    init_par = [0.5, 0.02, 0.02, 0.003, 0.003];

    using PyCall

    pypesto = pyimport("pypesto")

    # convert PyPesto result to MCMCChains.jl chain type
    function Chains_from_pypesto(result; kwargs...)
        trace_x = result.sample_result["trace_x"] # parameter values
        trace_neglogp = result.sample_result["trace_neglogpost"] # posterior values
        samples = Array{Float64}(undef, size(trace_x, 2), size(trace_x, 3) + 1, size(trace_x, 1))
        samples[:, begin:end-1, :] .= PermutedDimsArray(trace_x, (2, 3, 1))
        samples[:, end, :] = .-PermutedDimsArray(trace_neglogp, (2, 1))
        param_names = Symbol.(result.problem.x_names)
        chain = Chains(
            samples,
            vcat(param_names, :lp),
            (parameters = param_names, internals = [:lp]);
            kwargs...
        )
        return chain
    end

    # define negative loglikelihood
    neg_llh = p -> begin
        return Llh.NegLogLikelihood(p, patient_data, S0=S0)
    end

    # define objective
    objective = pypesto.Objective(fun=neg_llh)

    # define problem
    pypesto_problem = pypesto.Problem(
        objective,
        x_names=par_names,
        lb = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
        ub = [1.0, 1.0, 1.0, 1.0, 1.0],
        copy_objective = false,
    )

    # define sampler
    sampler = pypesto.sample.AdaptiveMetropolisSampler();

    # function for sampling
    function mcmc_chain()
        result = pypesto.sample.sample(
                    pypesto_problem,
                    n_samples=niter,
                    x0=Vector(init_par), # starting point
                    sampler=sampler,
                    )
    return  Chains_from_pypesto(result)
    end;
end

# initialize and run the jobs for the workers
jobs = [@spawnat(i, @timed(mcmc_chain())) for i in workers()]

all_chains = map(fetch, jobs)

chains = all_chains[1].value.value.data

# get the chains
for j in 2:nworkers()
    global chains
    chains = cat(chains, all_chains[j].value.value.data, dims=(3,3))
end


chs = MCMCChains.Chains(chains, [:beta, :m_basal, :m_size, :d_size, :d_metastasis, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:beta, :m_basal, :m_size, :d_size, :d_metastasis], :internals => [:lp]))


h5open("output/sampling_$(MODEL)_"*string(nworkers())*"chs_"*string(niter)*"it_$(θ).h5", "w") do f
  write(f, complete_chain)
end



