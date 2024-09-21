using Distributed # package for distributed computing in julia


# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(dirname(pwd()))
  Pkg.instantiate(); Pkg.precompile()
end

# sutff needed on workers and mcmc_chain
@everywhere begin
    using DataFrames
    using DifferentialEquations
    using DifferentialEquations.EnsembleAnalysis
    using JumpProcesses
    using Plots
    using Statistics
    using Distributions

    using CSV
    using DataFrames
    using DelimitedFiles
    using HDF5

    using Optimization
    using OptimizationOptimJL
    # using OptimizationBBO
    using OptimizationMultistartOptimization
    using OptimizationCMAEvolutionStrategy

    using MCMCChains
    using MCMCChainsStorage
    using AdvancedMH
    using Turing
    using AbstractMCMC
    using StatsPlots

    using Particles
    using ParticlesDE
    using StaticDistributions
    
    # true underlying parameters
    beta = 0.5
    m_basal = 0.01
    m_size = 0.003
    m_sigma = 0.03
    m_order = 1.0
    d_size = 0.001
    d_metastasis = 0.002
    θ = [beta, m_basal, m_size, m_sigma, m_order, d_size, d_metastasis]
end

# stuff only needed on workers
@everywhere workers() begin
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

    # load model
    include("simplified_model.jl");


    # true underlying parameters
    beta = 0.5
    m_basal = 0.01
    m_size = 0.003
    m_sigma = 0.03
    m_order = 1.0
    d_size = 0.001
    d_metastasis = 0.002
    θ = [beta, m_basal, m_size, m_sigma, m_order, d_size, d_metastasis]


    # initial conditions
    S0 = 0.065;

    # number of patients to use 
    npat = 500;

    
    # load data
    data_path = joinpath(dirname(pwd()), "second_model/data/simplified_model/proportional_data_$(npat)_patients_$(θ)_promise2.jld2")
    patient_data = load(data_path, "proportional_data");


    # settings for the sampler
    niter = 100000
    init_par = [0.5, 0.02, 0.02, 0.003, 0.003];

    # define negative loglikelihood
    neg_llh = p -> begin
            return NegLogLikelihood(p, patient_data, S0=0.065)
        end

    # define objective
    objective = pypesto.Objective(fun=neg_llh)

    # define problem
    pypesto_problem = pypesto.Problem(
        objective,
        x_names=["beta", "m_basal", "m_size", "d_size", "d_metastasis"],
        lb = [0.0, 0.0, 0.0, 0.0, 0.0],
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


niter = 100000

chs = MCMCChains.Chains(chains, [:beta, :m_basal, :m_size, :d_size, :d_metastasis, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:beta, :m_basal, :m_size, :d_size, :d_metastasis], :internals => [:lp]))


h5open("output/simplified_model/sampling_proportional_data_"*string(nworkers())*"chs_"*string(niter)*"it_$(θ).h5", "w") do f
  write(f, complete_chain)
end



