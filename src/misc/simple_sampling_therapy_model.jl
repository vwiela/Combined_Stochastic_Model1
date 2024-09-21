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
using JLD2

using Optimization
using OptimizationOptimJL
# using OptimizationBBO

using MCMCChains
using MCMCChainsStorage
using AdvancedMH
using AbstractMCMC


# true underlying parameters
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

# load model
include("TherapyLikelihoods.jl");
import .TherapyLikelihoods as Llh


# true underlying parameters
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

# initial conditions
S0 = 0.065;

# number of patients to use 
npat = 1000;

# load data
data_path = joinpath(dirname(pwd()), "second_model/data/simplified_model/additive_treated_data_$(npat)_patients_$(θ).jld2")
patient_data = load(data_path, "additive_treated_data");


# settings for the sampler
niter = 1000000
nchains = 4
init_par = [0.5, 0.45, 0.2, 0.01, 0.003, 0.001, 0.002];

# define negative loglikelihood
llh = p -> begin
    return -Llh.NegLogLikelihood(p, patient_data, S0=0.065)
end

# define bounds
log_lb_x0 = [-0.75, -0.85, -1.9, -9, -9, -8, -8]
log_ub_x0 = [-0.65, -0.75, -1.4, -4, -4, -5, -5]

insupport(θ) = all(θ .>= exp.(log_lb_x0)) && all(θ .<= exp.(log_ub_x0))

density(θ) = insupport(θ) ? llh(θ) : -Inf

model = DensityModel(density)

spl = RWMH(MvNormal(zeros(7), [0.01, 0.01, 0.01, 0.002, 0.0005, 0.0005, 0.0005]))

# Sample from the posterior.
jlchain = sample(model, spl, MCMCThreads(), niter, nchains; param_names=["beta0" ,"rho", "delta", "m_basal", "m_size", "d_size", "d_metastasis"], initial_params=[θ, θ, θ, θ], chain_type=Chains)

complete_chain = set_section(jlchain, Dict(:parameters => [:beta0, :rho, :delta, :m_basal, :m_size, :d_size, :d_metastasis], :internals => [:lp]))


h5open(joinpath(dirname(pwd()), "output/simplified_model/simple_sampling_treated_data_"*string(nworkers())*"chs_"*string(niter)*"it_$(θ).h5"), "w") do f
  write(f, complete_chain)
end



