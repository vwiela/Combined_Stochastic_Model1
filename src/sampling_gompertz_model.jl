using Distributed # package for distributed computing in julia

# set main directory
dir = dirname(pwd());

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

    using MCMCChains
    using MCMCChainsStorage
    using AdvancedMH
    using Turing
    using AbstractMCMC
    using StatsPlots

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
    include(joinpath(dir, "src/GompertzLikelihoods.jl"));
    import .GompertzLikelihoods as Llh


    # true underlying parameters
    K = 150000
    alpha = 0.2
    m_basal = 0.01
    m_size = 0.001
    d_size = 0.0003
    d_metastasis = 0.003

    θ = [K, alpha, m_basal, m_size, d_size, d_metastasis];

    # initial conditions
    S0 = 0.05;

    # number of patients to use 
    npat = 500;

    # load data
    load_path = joinpath(dir, "data/gompertz_data_$(npat)_patients_$(θ).jld2")
    patient_data = load(load_path, "gompertz_dataf");


    # settings for the sampler
    niter = 100000
   # best vector from optimization
    init_par = [150000, 0.2, 0.01, 0.001, 0.0003, 0.004]

    # define negative loglikelihood
    neg_llh = p -> begin
            return Llh.NegLogLikelihood(p, patient_data, S0=0.05)
        end

    # define objective
    objective = pypesto.Objective(fun=neg_llh)

    # define problem
    pypesto_problem = pypesto.Problem(
        objective,
        x_names=["K", "alpha", "m_basal", "m_size", "d_size", "d_metastasis"],
        lb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ub = [300000, 1.0, 1.0, 1.0, 1.0, 1.0],
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


chs = MCMCChains.Chains(chains, [:K, :alpha, :m_basal, :m_size, :d_size, :d_metastasis, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:K, :alpha, :m_basal, :m_size, :d_size, :d_metastasis], :internals => [:lp]))

niter = 100000

# true underlying parameters
K = 150000
alpha = 0.2
m_basal = 0.01
m_size = 0.001
d_size = 0.0003
d_metastasis = 0.003

θ = [K, alpha, m_basal, m_size, d_size, d_metastasis];

h5open(joinpath(dir, "output/sampling_gompertz_"*string(nworkers())*"chs_"*string(niter)*"it_$(θ).h5"), "w") do f
  write(f, complete_chain)
end


