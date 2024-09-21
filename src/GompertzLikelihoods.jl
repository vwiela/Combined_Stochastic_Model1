module GompertzLikelihoods

using DifferentialEquations
using JumpProcesses
using Statistics
using Distributions
using Random
using DataFrames
using JLD2
using ForwardDiff
using Optimization
using OptimizationOptimJL

using Cubature
using FastGaussQuadrature, LinearAlgebra
# using Trapz
using Roots
using SpecialFunctions


# export functions needed
export ObservationProbability, NumericSurvivalProbability, NumericDeathProbability, AnalyticSurvivalProbability, AnalyticDeathProbability
export NegLogLikelihood, TumorNegLogLikelihood, MetastasisNegLogLikelihood, DeathNegLogLikelihood, PatientLogLikelihood
export NumericSurvivalProbability1, NumericSurvivalProbability2, NumericSurvivalProbability3, NumericSurvivalProbability4, NumericSurvivalProbability5, NumericSurvivalProbability6
export lambdaN, LambdaN, lambdaD, LambdaD, Phi


# set seed for reproducibility
Random.seed!(123)

# solution and ODE for the gompertz groth process
function GompertzSolution(t, S0, K, α)
    return K * exp(log(S0/K) * exp(-α * t))
end

function GompertzODE!(du, u, p, t)
    du[1] = p[2] * u[1] * log(p[1]/u[1])
end

function TumorPath(S0, p; endtime=30.0)
    prob = ODEProblem(GompertzODE!, [S0], (0.0, endtime), [p.K, p.α])
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sol
end

# set model parameter values
K = 100
α = 0.1
m_basal = 0.04
m_size = 0.04
d_size = 0.01
d_metastasis = 0.01

p = (K=100, α=0.1, m_basal = 0.04, m_size = 0.04, d_size = 0.01, d_metastasis = 0.01)

gt_par = Dict(
    "K" => log10(K),
    "α" => log10(α),
    "m_basal" => log10(m_basal), 
    "m_size" => log10(m_size), 
    "d_size" => log10(d_size), 
    "d_metastasis" => log10(d_metastasis)
    )

# set initial condition 
S0 = 0.05
u0 = [S0, 0, 0]

# set time interval
endtime = 30.0
timepoints = 0.0:1.0:30.0
tspan = (0.0, endtime)

# set chosen observation noise
sigma = 0.1;

# define the jump-rate functions
metastasis_rate(u, p, t) = p.m_basal+p.m_size*sqrt(u[1])
death_rate(u, p, t) = p.d_size*sqrt(u[1])+p.d_metastasis*u[2]

# define the jump functions
function metastasis_affect!(integrator)
    if (integrator.u[3] == 0) 
        integrator.u[2] += 1
    end
    nothing
end

function death_affect!(integrator)
    if (integrator.u[3] == 0) 
        integrator.u[3] += 1
    end
    terminate!(integrator) # could stop the simulation after death, but for visualization reasons one might not want to.
    nothing
end

# # define the ODE problem
# prob = ODEProblem(GompertzODE!, u0, tspan, full_p)

# # define the jump problem
# metastasis_jump = VariableRateJump(metastasis_rate, metastasis_affect!)
# death_jump =  VariableRateJump(death_rate, death_affect!)
# jump_problem = JumpProblem(prob, Direct(), metastasis_jump, death_jump)

# # solve and plot the solution
# sol = solve(jump_problem, Tsit5(), tstops=0.0:0.1:endtime)



#-----------------------------------------------------------------------------------------------------------------------------------------------
# define all the likelihood functions


# helper functions

function lambdaN(
    t, 
    K::Real,
    α::Real,
    m_basal::Real, 
    m_size::Real, 
    S0::Real, 
    )

    S = GompertzSolution(t, S0, K, α)
    return m_basal + m_size * sqrt(S)
end

function lambdaD(
    t, 
    K::Real,
    α::Real,
    d_size::Real, 
    d_metas::Real, 
    S0::Real, 
    Nt
    )

    S = GompertzSolution(t, S0, K, α)
    return d_size * sqrt(S) + d_metas * Nt
end

function LambdaN(
    t1, 
    t2, 
    K::Real, 
    α::Real,
    m_basal::Real, 
    m_size::Real, 
    S0::Real
    )

    dt = (t2-t1)

    return dt*m_basal + (m_size*((sqrt(K*(S0/K)^exp(-(α*t1)))expinti(log(S0/K)/(2*exp(α*t1))))/(S0/K)^(exp(-(α*t1))/2) - 
    (sqrt(K*(S0/K)^exp(-(α*t2)))expinti(log(S0/K)/(2*exp(α*t2))))/(S0/K)^(exp(-(α*t2))/2)))/α
end

function LambdaD(
    t1,
    t2,
    K::Real, 
    α::Real,
    d_size::Real,
    d_metas::Real,
    S0::Real,
    n::Real,
)
    dt = t2-t1

    return dt*(n*d_metas) + (d_size*((sqrt(K*(S0/K)^exp(-(α*t1)))expinti(log(S0/K)/(2*exp(α*t1))))/(S0/K)^(exp(-(α*t1))/2) - 
    (sqrt(K*(S0/K)^exp(-(α*t2)))expinti(log(S0/K)/(2*exp(α*t2))))/(S0/K)^(exp(-(α*t2))/2)))/α
end

function Phi(
    t1,
    t2,
    K::Real, 
    α::Real,
    d_size::Real,
    d_metas::Real,
    S0::Real,
    n::Real,
)
    dt = t2-t1

    return exp(-(dt*(n*d_metas) + (d_size*((sqrt(K*(S0/K)^exp(-(α*t1)))expinti(log(S0/K)/(2*exp(α*t1))))/(S0/K)^(exp(-(α*t1))/2) - 
    (sqrt(K*(S0/K)^exp(-(α*t2)))expinti(log(S0/K)/(2*exp(α*t2))))/(S0/K)^(exp(-(α*t2))/2)))/α))
end

# numerical integrals for death probability

function NumericSurvivalProbability1(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    integral = (1/(LambdaN(t1, t2, K, α, m_basal, m_size, S0))) *
    hquadrature(u1 ->lambdaN(u1, K, α, m_basal, m_size, S0)*
        Phi(t1, u1, K, α, d_size, d_metas, S0, n)*
        Phi(u1, t2, K, α, d_size, d_metas, S0, n+1),
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability2(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    integral = (2/(LambdaN(t1, t2, K, α, m_basal, m_size, S0)^2)) *
    hquadrature(u1 ->lambdaN(u1, K, α, m_basal, m_size, S0)*
        Phi(t1, u1, K, α, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, K, α, m_basal, m_size, S0)*
            Phi(u1, u2, K, α, d_size, d_metas, S0, n+1) *
            Phi(u2, t2,  K, α, d_size, d_metas, S0, n+2),
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end


function NumericSurvivalProbability3(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    integral = (6/(LambdaN(t1, t2, K, α, m_basal, m_size, S0)^3)) *
    hquadrature(u1 ->lambdaN(u1, K, α, m_basal, m_size, S0)*
        Phi(t1, u1, K, α, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, K, α, m_basal, m_size, S0)*
            Phi(u1, u2, K, α, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, K, α, m_basal, m_size, S0) *
                Phi(u2, u3, K, α, d_size, d_metas, S0, n+2) *
                Phi(u3, t2, K, α, d_size, d_metas, S0, n+3),
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability4(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    integral = (24/(LambdaN(t1, t2, K, α, m_basal, m_size, S0)^4)) *
    hquadrature(u1 ->lambdaN(u1, K, α, m_basal, m_size, S0)*
        Phi(t1, u1, K, α, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, K, α, m_basal, m_size, S0)*
            Phi(u1, u2, K, α, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, K, α, m_basal, m_size, S0) *
                Phi(u2, u3, K, α, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, K, α, m_basal, m_size, S0) *
                            Phi(u3, u4, K, α, d_size, d_metas, S0, n+3) *
                            Phi(u4, t2, K, α, d_size, d_metas, S0, n+4),
                u3, t2
                )[1],
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability5(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    integral = 120/((LambdaN(t1, t2, K, α, m_basal, m_size, S0))^5) *
    hquadrature(u1 ->lambdaN(u1, K, α, m_basal, m_size, S0)*
        Phi(t1, u1, K, α, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, K, α, m_basal, m_size, S0)*
            Phi(u1, u2, K, α, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, K, α, m_basal, m_size, S0) *
                Phi(u2, u3, K, α, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, K, α, m_basal, m_size, S0) *
                    Phi(u3, u4, K, α, d_size, d_metas, S0, n+3) *
                    hquadrature(u5 -> lambdaN(u5, K, α, m_basal, m_size, S0) *
                        Phi(u4, u5, K, α, d_size, d_metas, S0, n+4) *
                        Phi(u5, t2, K, α, d_size, d_metas, S0, n+5),
                    u4, t2
                    )[1],
                u3, t2
                )[1],
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability6(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    integral = 720/((LambdaN(t1, t2, K, α, m_basal, m_size, S0))^6) *
    hquadrature(u1 ->lambdaN(u1, K, α, m_basal, m_size, S0)*
        Phi(t1, u1, K, α, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, K, α, m_basal, m_size, S0)*
            Phi(u1, u2, K, α, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, K, α, m_basal, m_size, S0) *
                Phi(u2, u3, K, α, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, K, α, m_basal, m_size, S0) *
                    Phi(u3, u4, K, α, d_size, d_metas, S0, n+3) *
                    hquadrature(u5 -> lambdaN(u5, K, α, m_basal, m_size, S0) *
                        Phi(u4, u5, K, α, d_size, d_metas, S0, n+4) *
                        hquadrature(u6 -> lambdaN(u6, K, α, m_basal, m_size, S0) *
                            Phi(u5, u6, K, α, d_size, d_metas, S0, n+5) *
                            Phi(u6, t2, K, α, d_size, d_metas, S0, n+6),
                        u5, t2
                        )[1],
                    u4, t2
                    )[1],
                u3, t2
                )[1],
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end

# Numeric integration with FastGaussQuadrature for AD compatibility

function FGIntegral1(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, K, α, m_basal, m_size, S0) .* 
        Phi.(t1, ttrans, K, α, d_size, d_metas, S0, n) .* 
        Phi.(ttrans, t2, K, α, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability1(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)

    return (1/(LambdaN(t1, t2, K, α, m_basal, m_size, S0))) * 
    FGIntegral1(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
end

function FGIntegral2(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, K, α, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, K, α, d_size, d_metas, S0, n) .* 
        FGIntegral1.(ttrans, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability2(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)

    return (2/(LambdaN(t1, t2, K, α, m_basal, m_size, S0))^2) * 
    FGIntegral2(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
end


function FGIntegral3(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, K, α, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, K, α, d_size, d_metas, S0, n) .* 
        FGIntegral2.(ttrans, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability3(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)

    return (6/(LambdaN(t1, t2, K, α, m_basal, m_size, S0))^3) * 
    FGIntegral3(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
end


function FGIntegral4(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, K, α, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, K, α, d_size, d_metas, S0, n) .* 
        FGIntegral3.(ttrans, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability4(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)

    return (24/(LambdaN(t1, t2, K, α, m_basal, m_size, S0))^4) * 
    FGIntegral4(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
end

function FGIntegral5(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, K, α, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, K, α, d_size, d_metas, S0, n) .* 
        FGIntegral4.(ttrans, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability5(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)

    return (120/(LambdaN(t1, t2, K, α, m_basal, m_size, S0))^5) * 
    FGIntegral5(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
end

function FGIntegral6(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, K, α, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, K, α, d_size, d_metas, S0, n) .* 
        FGIntegral5.(ttrans, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability6(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)

    return (720/(LambdaN(t1, t2, K, α, m_basal, m_size, S0))^6) * 
    FGIntegral6(t1, t2, K, α, m_basal, m_size, d_size, d_metas, S0, n)
end

# important probability functions

function ObservationProbability(
    Xt::Vector{<:Real},
    Yt::Vector{<:Real}; 
    sigma::Real=0.1 # For now with fixed noise level (can be changed to estimated noise later)
    )::Real

    # unpack 
    # unpack data
    St, Nt, Dt = Xt
    Bt = Yt[1]

    # Compute the likelihood of the tumor size observation
    obs_prob = pdf(Normal(St, sigma*St), Bt)

    return obs_prob
end

function NumericSurvivalProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    # Unpack parameters
    K, α, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        surv_prob = Phi(t⁻, t, K, α, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        surv_prob = NumericSurvivalProbability1(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        surv_prob = NumericSurvivalProbability2(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        surv_prob = NumericSurvivalProbability3(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        surv_prob = NumericSurvivalProbability4(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        surv_prob = NumericSurvivalProbability5(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        surv_prob = NumericSurvivalProbability6(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    else
        println("more than 6 metastasis in one interval, $(Nt-Nt⁻)")
        surv_prob = 1.0
    end
    # did not observe more than 6 jumps in one time interval.

  
    if (surv_prob < 0.0)
        println("surv_prob is negative for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_size, d_metas])
        surv_prob=1e-50
    end
    # if (surv_prob > 1.0)
    #     println("surv_prob is bigger 1 for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_size, d_metas])
    # end

    return surv_prob
end


function FGSurvivalProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    # Unpack parameters
    beta, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        surv_prob = Phi(t⁻, t, K, α , d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        surv_prob = FGSurvivalProbability1(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        surv_prob = FGSurvivalProbability2(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        surv_prob = FGSurvivalProbability3(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        surv_prob = FGSurvivalProbability4(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        surv_prob = FGSurvivalProbability5(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        surv_prob = FGSurvivalProbability6(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    else
        println("more than 6 metastasis in one interval, $(Nt-Nt⁻)")
        surv_prob = 1.0
    end
    # did not observe more than 6 jumps in one time interval.

  
    if (surv_prob < 0.0)
        println("surv_prob is negative for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_size, d_metas])
        surv_prob=FGSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0)
    end
    # if (surv_prob > 1.0)
    #     println("surv_prob is bigger 1 for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_size, d_metas])
    # end

    return surv_prob
end


function NumericDeathProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    # Unpack parameters
    K, α, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*Phi(t⁻, t, K, α, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*NumericSurvivalProbability1(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0,  Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*NumericSurvivalProbability2(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*NumericSurvivalProbability3(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*NumericSurvivalProbability4(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*NumericSurvivalProbability5(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*NumericSurvivalProbability6(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    end

  
    if (death_prob < 0.0)
        println("death_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ", death_prob, "\n", θ)
        death_prob=1e-50
    end
    # if (death_prob > 1.0)
    #     println("death_prob is bigger 1 for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
    # end

    return death_prob
end


function FGDeathProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    # Unpack parameters
    beta, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*Phi(t⁻, t, K, α, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*FGSurvivalProbability1(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*FGSurvivalProbability2(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*FGSurvivalProbability3(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*FGSurvivalProbability4(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*FGSurvivalProbability5(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        death_prob = lambdaD(t, K, α, d_size, d_metas, S0, Nt)*FGSurvivalProbability6(t⁻, t, K, α, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    end

  
    if (death_prob < 0.0)
        println("death_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ", death_prob, "\n", θ)
        death_prob=1e-50
    end
    # if (death_prob > 1.0)
    #     println("death_prob is bigger 1 for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
    # end

    return death_prob
end


function MetastasisProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real
    # Unpack parameters
    K, α, m_basal, m_size, = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    dn = Int(Nt - Nt⁻)
    Lt = LambdaN(t⁻, t, K, α, m_basal, m_size, S0)
    met_prob = (Lt^dn)/(factorial(dn))*exp(-Lt)

    return met_prob
end

# likelihood functions

function TimepointLikelihood(
    t⁻,
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real}, 
    Yt::Vector{<:Real},
    S0::Real,
    )::Real

    # get observation probability (rather rate since it is not normalized)
    obs_prob = ObservationProbability(Xt, Yt)

    if (Xt[3] == 0.0) # no death

        # get process probability
        process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0) * (NumericSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    else # death
        process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0) * (NumericDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    end
    if (obs_prob < 0.0)
        println("obs_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt, Yt = $Yt\n It is ",obs_prob)
    end
    if (process_prob < 0.0)
        println("process_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt, Yt = $Yt\n 
        and θ=$θ. It is ",process_prob,
        "\n With metastasisProbability = ", MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0), "\n",
        "and deathProbability = ", NumericDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    end
    L = obs_prob * process_prob
    return L
end

function PatientLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
    )::Real

    # Unpack parameters
    K, α, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    timepoints, Bt, Nt, Dt = data

    # get X based on parameters
    St = GompertzSolution.(timepoints, S0, K, α)

    # Initialize loglikelihood
    l = 0.0

    # Loop over data points
    for i in eachindex(timepoints)

        if (i == 1)
            t = timepoints[i]
            t⁻ = 0.0
            Xt = [St[i], Nt[i], Dt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
            Yt = [Bt[i], Nt[i], Dt[i]]
            Xt⁻ = [S0, 0, 0]
        else
            # Set t, t⁻1
            t = timepoints[i]
            t⁻ = timepoints[i-1]

            # Set Xt, Yt, Xt⁻1
            Xt = [St[i], Nt[i], Dt[i]]
            Yt = [Bt[i], Nt[i], Dt[i]]
            Xt⁻ = [St[i-1], Nt[i-1], Dt[i-1]]
        end

        l += log(TimepointLikelihood(t⁻, t, θ, Xt⁻, Xt, Yt, S0))
    end

    return l
end

function OdeLogLikelihood(
    θ::Vector{<:Real},
    data;
    S0::Real=0.05
    )::Real

    n_patients = data.patient_id[end]
    ll = 0.0
    for i in 1:n_patients
        patient_data = data[data.patient_id .== i, :]
        ll += PatientLogLikelihood(
            θ, 
            [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis, 
            patient_data.death
            ],
            S0=S0)
    end
    return ll
end

function NegLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
    )::Real

    return -OdeLogLikelihood(θ, data, S0=S0)
end

#------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

# hierarchical optimization
"""
    This is to avoid the above problems of observing death probabilities being insanely high because of to high ODE 
    parameters and small intervals.
"""

# single probability functions
function OnlyMetastasisProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    # Unpack parameters
    K, α, m_basal, m_size, = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    dn = Int(Nt - Nt⁻)
    Lt = LambdaN(t⁻, t, K, α, m_basal, m_size, S0)
    met_prob = (Lt^dn)/(factorial(dn))*exp(-Lt)

    return met_prob
end

function OnlyDeathProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    if (Xt[3] == 0.0) # no death

        # get process probability
        process_prob = (NumericSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    else # death
        process_prob = (NumericDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    end
    return process_prob
end

function FGOnlyDeathProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    if (Xt[3] == 0.0) # no death

        # get process probability
        process_prob = (FGSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    else # death
        process_prob = (FGDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    end
    return process_prob
end


# likelihood functions

function TumorNegLogLikelihood(
    K,
    α,
    data;
    S0::Real=0.05
    )::Real

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Nt, Dt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis,
            patient_data.death
        ]

        # get X based on parameters
        St = GompertzSolution.(timepoints, S0, K, α)

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)

            if (i == 1)
                t = timepoints[i]
                t⁻ = 0.0
                Xt = [St[i], Nt[i], Dt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
                Yt = [Bt[i], Nt[i], Dt[i]]
                Xt⁻ = [S0, 0, 0]
            else
                # Set t, t⁻1
                t = timepoints[i]
                t⁻ = timepoints[i-1]

                # Set Xt, Yt, Xt⁻1
                Xt = [St[i], Nt[i], Dt[i]]
                Yt = [Bt[i], Nt[i], Dt[i]]
                Xt⁻ = [St[i-1], Nt[i-1], Dt[i-1]]
            end

            # likelihood is basically just the observation probability here.
            l += log(ObservationProbability(Xt, Yt))
        end
        ll += l
    end
    return -ll
end

function MetastasisNegLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
)
    # Unpack parameters
    K, α, m_basal, m_size = θ

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Nt, Dt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis,
            patient_data.death
        ]

        # get X based on parameters
        St = GompertzSolution.(timepoints, S0, K, α)

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)

            if (i == 1)
                t = timepoints[i]
                t⁻ = 0.0
                Xt = [St[i], Nt[i], Dt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
                Yt = [Bt[i], Nt[i], Dt[i]]
                Xt⁻ = [S0, 0, 0]
            else
                # Set t, t⁻1
                t = timepoints[i]
                t⁻ = timepoints[i-1]

                # Set Xt, Yt, Xt⁻1
                Xt = [St[i], Nt[i], Dt[i]]
                Yt = [Bt[i], Nt[i], Dt[i]]
                Xt⁻ = [St[i-1], Nt[i-1], Dt[i-1]]
            end

            # likelihood is basically just the observation probability here.
            l += log(OnlyMetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0))
        end
        ll += l
    end
    return -ll
end

function DeathNegLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
)
    # Unpack parameters
    K, α, m_basal, m_size = θ

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Nt, Dt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis,
            patient_data.death
        ]

        # get X based on parameters
        St = GompertzSolution.(timepoints, S0, K, α)

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)

            if (i == 1)
                t = timepoints[i]
                t⁻ = 0.0
                Xt = [St[i], Nt[i], Dt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
                Yt = [Bt[i], Nt[i], Dt[i]]
                Xt⁻ = [S0, 0, 0]
            else
                # Set t, t⁻1
                t = timepoints[i]
                t⁻ = timepoints[i-1]

                # Set Xt, Yt, Xt⁻1
                Xt = [St[i], Nt[i], Dt[i]]
                Yt = [Bt[i], Nt[i], Dt[i]]
                Xt⁻ = [St[i-1], Nt[i-1], Dt[i-1]]
            end

            # likelihood is basically just the observation probability here.
            l += log(OnlyDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
        end
        ll += l
    end
    return -ll
end

function FGDeathNegLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
)
    # Unpack parameters
    K, α, m_basal, m_size = θ

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Nt, Dt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis,
            patient_data.death
        ]

        # get X based on parameters
        St = GompertzSolution.(timepoints, S0, K, α)

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)

            if (i == 1)
                t = timepoints[i]
                t⁻ = 0.0
                Xt = [St[i], Nt[i], Dt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
                Yt = [Bt[i], Nt[i], Dt[i]]
                Xt⁻ = [S0, 0, 0]
            else
                # Set t, t⁻1
                t = timepoints[i]
                t⁻ = timepoints[i-1]

                # Set Xt, Yt, Xt⁻1
                Xt = [St[i], Nt[i], Dt[i]]
                Yt = [Bt[i], Nt[i], Dt[i]]
                Xt⁻ = [St[i-1], Nt[i-1], Dt[i-1]]
            end

            # likelihood is basically just the observation probability here.
            l += log(FGOnlyDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
        end
        ll += l
    end
    return -ll
end

function hierarchOptimization(data, x0; S0=0.065)

    # get the true parameters from the data
    #beta, m_basal, m_size, d_basal, d_size, d_metas = data.parameters
    #true_par = (beta = beta, m_basal = m_basal, m_size = m_size, d_basal = d_basal, d_size = d_size, d_metas = d_metas);


    # first we want to estimate only beta
    gompertzOptim(x, p) = TumorNegLogLikelihood(x[1], x[2], data, S0=S0)
    gompertzOptimFunc = OptimizationFunction(gompertzOptim, Optimization.AutoForwardDiff())
    gompertzOptimProblem = OptimizationProblem(gompertzOptimFunc, x0[1:2], lb=zeros(2).+1e-4, ub=[300000, 1])
    gompertzEst = solve(gompertzOptimProblem, LBFGS(), maxiters=10^5)
    println("K:", gompertzEst[1], "\n", "α:", gompertzEst[2], "\n")

    # next we want to optimize the metastasis parameters
    metOptim(x, p) = MetastasisNegLogLikelihood([gompertzEst[1], gompertzEst[2], x[1], x[2]], data, S0=S0)
    metOptimFunc = OptimizationFunction(metOptim, Optimization.AutoForwardDiff())
    metOptimProblem = OptimizationProblem(metOptimFunc, x0[3:4], lb=zeros(2).+1e-4, ub=ones(2))
    metEst = solve(metOptimProblem, LBFGS(), maxiters=10^5)
    println("Metastasis:", metEst, "\n")

    # next we want to optimize the death parameters
    deathOptim(x, p) = DeathNegLogLikelihood([gompertzEst[1], gompertzEst[2], metEst[1], metEst[2], x[1], x[2]], data, S0=S0)
    deathOptimFunc = OptimizationFunction(deathOptim, Optimization.AutoForwardDiff())
    deathOptimProblem = OptimizationProblem(deathOptimFunc, x0[5:6], lb=zeros(2).+1e-4, ub=ones(2))
    deathEst = solve(deathOptimProblem, SAMIN(), maxiters=10^6)
    est_par = [gompertzEst[1], gompertzEst[2], metEst[1], metEst[2], deathEst[1], deathEst[2]]

    # print solutions
    #println("True parameter is \n", true_par)
    println("Estimated parameter is \n", est_par)

    #return true_par, est_par
    return est_par
end

function LogHierarchOptimization(data, x0; S0=0.065, lb=log.(zeros(6).+1e-4), ub=log.([300000, ones(5)...]), optimizer="SAMIN", llh_type = "Numeric")

    # get the true parameters from the data
    #beta, m_basal, m_size, d_basal, d_size, d_metas = data.parameters
    #true_par = (beta = beta, m_basal = m_basal, m_size = m_size, d_basal = d_basal, d_size = d_size, d_metas = d_metas);

    if optimizer != "SAMIN"
        error("Please choose SAMIN as optimizer for this function.")
    end

    # first we want to estimate only beta
    gompertzOptim(x, p) = TumorNegLogLikelihood(exp(x[1]), exp(x[2]), data, S0=S0)
    gompertzOptimFunc = OptimizationFunction(gompertzOptim, Optimization.AutoForwardDiff())
    gompertzOptimProblem = OptimizationProblem(gompertzOptimFunc, x0[1:2], lb=lb[1:2], ub=ub[1:2])
    gompertzEst = solve(gompertzOptimProblem, LBFGS(), maxiters=10^6)
    println("K:", gompertzEst[1], "\n", "α:", gompertzEst[2], "\n")

    # next we want to optimize the metastasis parameters
    metOptim(x, p) = MetastasisNegLogLikelihood(exp.([gompertzEst[1], gompertzEst[2], x[1], x[2]]), data, S0=S0)
    metOptimFunc = OptimizationFunction(metOptim, Optimization.AutoForwardDiff())
    metOptimProblem = OptimizationProblem(metOptimFunc, x0[3:4], lb=lb[3:4], ub=ub[3:4])
    metEst = solve(metOptimProblem, LBFGS(), maxiters=10^6)
    println("Metastasis:", metEst, "\n")

    # next we want to optimize the death parameters
    deathOptim(x, p) = DeathNegLogLikelihood(exp.([gompertzEst[1], gompertzEst[2], metEst[1], metEst[2], x[1], x[2]]), data, S0=S0)
    deathOptimFunc = OptimizationFunction(deathOptim, Optimization.AutoForwardDiff())
    deathOptimProblem = OptimizationProblem(deathOptimFunc, x0[5:6], lb=lb[5:6], ub=ub[5:6])
    deathEst = solve(deathOptimProblem, SAMIN(), maxiters=10^8)
    est_par = [gompertzEst[1], gompertzEst[2], metEst[1], metEst[2], deathEst[1], deathEst[2]]

    # print solutions
    #println("True parameter is \n", true_par)
    println("Estimated parameter is \n", est_par)

    #return true_par, est_par
    return est_par
end

function LogOptimization(data, x0; S0=0.065, lb=log.(zeros(6).+1e-4), ub=log.([300000, ones(5)...]), optimizer="SAMIN", llh_type = "Numeric")

    if optimizer != "SAMIN"
        error("Please choose SAMIN as optimizer for this function.")
    end

    obj_values = []
    times = []
    function callback(p, l)
        push!(obj_values, l)
        push!(times, time_ns())
        return false
    end

    # next we want to optimize the death parameters
    OptimLlh(x, p) = NegLogLikelihood(exp.(x), data, S0=S0)
    OptimFunc = OptimizationFunction(OptimLlh, Optimization.AutoForwardDiff())
    OptimProblem = OptimizationProblem(OptimFunc, x0, lb=lb, ub=ub)
    joined_est = solve(OptimProblem, SAMIN(), maxiters=10^8, callback=callback)
    est_par = joined_est.u

    # print solutions
    println("Estimated parameter is \n", est_par)
    nllh = NegLogLikelihood(exp.(joined_est.u), data, S0=S0)
    times_sec = (Int.(times) .- minimum(Int.(times)))/1e9
    res_dict = Dict(
            "nllh" => nllh,
            "parameter" => joined_est.u,
            "result_object" => joined_est,
            "obj_val_trace" => obj_values,
            "time_trace" => times_sec
        )
    return res_dict
end


# function to carry out simulation and simulation_and_estimation

function simulation_and_estimation(
    par_dict::Dict;
    npat = 500,
    sim_algorithm = "sciml",
    log_par = nothing,
    rng = MersenneTwister(123)
)

    if sim_algorithm == "sciml"
        patient_data = second_model_data_simulator(par_dict, npat=npat, log_par=log_par)
    elseif sim_algorithm == "alternative"
        patient_data = second_model_alternative_simulator(par_dict, npat=npat, log_par=log_par)
    elseif sim_algorithm == "mnr"
        patient_data = second_model_mnr_simulator(par_dict, npat=npat, log_par=log_par, rng=rng)
    else
        error("Please choose a valid simulation algorithm from 'sciml', 'alternative' and 'mnr'. ")
        return
    end


    println("Data created \n")

    if isnothing(log_par)
        par_tuple = (beta = par_dict["beta"],
            m_basal = par_dict["m_basal"], 
            m_size = par_dict["m_size"], 
            d_size = par_dict["d_size"], 
            d_metastasis = par_dict["d_metastasis"])
    else
        par_tuple = (beta = 10.0^par_dict["beta"],
            m_basal = 10.0^par_dict["m_basal"], 
            m_size = 10.0^par_dict["m_size"], 
            d_size = 10.0^par_dict["d_size"], 
            d_metastasis = 10.0^par_dict["d_metastasis"])
    end

    θ = [p for p in values(par_tuple)]
    lb = zeros(length(θ)).+1e-6
    ub = ones(length(θ));

    # define negative log likelihood function
    function NegLlh(p, x)
        return NegLogLikelihood(p, patient_data, S0=0.05)
    end

    println("Start optimization! \n")
    # define optimization problem
    optf = OptimizationFunction(NegLlh, Optimization.AutoForwardDiff());
    opt_problem = OptimizationProblem(optf, θ, lb=lb, ub=ub);
    
    sol = solve(opt_problem, LBFGS())

    return patient_data, sol
end

end;