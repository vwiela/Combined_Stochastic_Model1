module GyllenbergWebbLikelihoods

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
using OptimizationCMAEvolutionStrategy
using OptimizationBBO

using Integrals
using Cubature
using LinearAlgebra
using Roots
using SpecialFunctions

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D


# export functions needed
# export ObservationProbability, NumericSurvivalProbability, NumericDeathProbability, AnalyticSurvivalProbability, AnalyticDeathProbability
# export NegLogLikelihood, TumorNegLogLikelihood, MetastasisNegLogLikelihood, DeathNegLogLikelihood, PatientLogLikelihood
# export NumericSurvivalProbability1, NumericSurvivalProbability2, NumericSurvivalProbability3, NumericSurvivalProbability4, NumericSurvivalProbability5, NumericSurvivalProbability6
# export lambdaN, LambdaN, lambdaD, LambdaD, Phi


# set seed for reproducibility
Random.seed!(123)

# solution and ODE for the Gyllenberg-Webb model
@parameters begin
    r = 1
    k = 2
    a = 1
    b = 1
    m = 2
    μ = 0.5
    d = 0.01
end

@variables begin
    P(t) = 0
    Q(t) = 0
    De(t) = 0
end

total_N(P,Q,De) = P+Q
r0(P,Q,De) = k*total_N(P,Q,De)/(a*total_N(P,Q,De)+1)
ri(P,Q,De) = r/(total_N(P,Q,De)+m)


# r0(P,Q) = k*total_N(P,Q)^2
# ri(P,Q) = r

eqs = [
    D(P) ~ (b-r0(P,Q, De))*P+ri(P,Q, De)*Q
    D(Q) ~ r0(P,Q, De)*P-(ri(P,Q, De)+μ)*Q
    D(De) ~ μ*Q-d*De
    ]

@named sys = ODESystem(eqs, t, [P,Q, De], [r,k,a,b,m,μ,d])
sys = structural_simplify(sys)

# set model parameter values
b = 1
μ = 0.05
m_basal = 0.04
m_size = 0.04
d_size = 0.01
d_metastasis = 0.01

p = (b = b, μ = μ, m_basal = 0.04, m_size = 0.04, d_size = 0.01, d_metastasis = 0.01)
θ = [b, μ, p.m_basal, p.m_size, p.d_size, p.d_metastasis]

gt_par = Dict(
    "K" => log10(b),
    "μ" => log10(μ),
    "m_basal" => log10(m_basal), 
    "m_size" => log10(m_size), 
    "d_size" => log10(d_size), 
    "d_metastasis" => log10(d_metastasis)
    )

# set initial condition 
P0 = 1.0

# set time interval
endtime = 30.0
timepoints = 0.0:1.0:30.0
tspan = (0.0, endtime)

# set chosen observation noise
sigma = 0.1;

function TotalTumorSize(t, P0, b, μ)
    parammap = [sys.μ => μ, b => p.b, sys.a => 0.03, sys.k => 0.05, sys.r => 0.1, sys.m => 2, sys.d => 0.01]
    initial_conditions = [sys.P => P0, sys.Q => 0.0, sys.De => 0.0]
    prob = ODEProblem(sys, parammap, (0.0, t), initial_conditions)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sum(sol(t))
end

#-----------------------------------------------------------------------------------------------------------------------------------------------
# define all the likelihood functions


# helper functions

function lambdaN(
    t, 
    m_basal::Real, 
    m_size::Real,
    sol::ODESolution,
    )

    S = sum(sol(t))
    return m_basal + m_size * sqrt(S)
end

function lambdaD(
    t, 
    d_size::Real, 
    d_metas::Real,
    sol::ODESolution,
    Nt
    )

    S = sum(sol(t))
    return d_size * sqrt(S) + d_metas * Nt
end

function LambdaN(
    t1,
    t2,
    m_basal::Real,
    m_size::Real,
    sol::ODESolution,
    )

    domain = (t1, t2)
    intprob=IntegralProblem((x,p) -> lambdaN(x, m_basal, m_size, sol), domain)
    intsol = solve(intprob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
    return intsol.u
end

function LambdaD(
    t1,
    t2,
    d_size::Real,
    d_metas::Real,
    sol::ODESolution,
    n::Real,
)
    domain = (t1, t2)
    intprob=IntegralProblem((x,p) -> lambdaD(x, d_size, d_metas, sol, n), domain)
    intsol = solve(intprob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
    return intsol.u
end

function Phi(
    t1,
    t2,
    d_size::Real,
    d_metas::Real,
    sol::ODESolution,
    n::Real,
)

    return exp(-LambdaD(t1, t2, d_size, d_metas, sol, n))
end


# Numeric integration with Integrals.jl package for AD compatibility

function SurvProbability1(t1, t2, m_basal, m_size, d_size, d_metas, sol, n)
    domain = (t1, t2)
    intprob = IntegralProblem((u1,p) ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n)*
        Phi(u1, t2, d_size, d_metas, sol, n+1),
    domain)
    integral = (1/(LambdaN(t1, t2, m_basal, m_size, sol))) * 
    solve(intprob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u
    return integral
end

function TestSurvProbability1(t1, t2, m_basal, m_size, d_size, d_metas, sol, n; integrator=CubatureJLp())
    domain = (t1, t2)
    intprob = IntegralProblem((u1,p) ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n)*
        Phi(u1, t2, d_size, d_metas, sol, n+1),
    domain)
    integral = (1/(LambdaN(t1, t2, m_basal, m_size, sol))) * 
    solve(intprob, integrator; reltol = 1e-3, abstol = 1e-3).u
    return integral
end

function SurvProbability2(t1, t2, m_basal, m_size, d_size, d_metas, sol, n)
    domain = (t1, t2)
    inner_integral(u1) = 
    solve(
    IntegralProblem((u2,p) ->lambdaN(u2, m_basal, m_size, sol)*
        Phi(u1, u2, d_size, d_metas, sol, n+1)*
        Phi(u2, t2, d_size, d_metas, sol, n+2), (u1, t2)),
    HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u

    outerintprob = 
    IntegralProblem((u1,p) ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n) *
        inner_integral(u1),
        domain)
    integral = (2/(LambdaN(t1, t2, m_basal, m_size, sol)^2)) * 
    solve(outerintprob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u
    return integral
end

function TestSurvProbability2(t1, t2, m_basal, m_size, d_size, d_metas, sol, n; integrator=CubatureJLp())
    domain = (t1, t2)
    inner_integral(u1) = 
    solve(
    IntegralProblem((u2,p) ->lambdaN(u2, m_basal, m_size, sol)*
        Phi(u1, u2, d_size, d_metas, sol, n+1)*
        Phi(u2, t2, d_size, d_metas, sol, n+2), (u1, t2)),
    integrator; reltol = 1e-3, abstol = 1e-3).u

    outerintprob = 
    IntegralProblem((u1,p) ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n) *
        inner_integral(u1),
        domain)
    integral = (2/(LambdaN(t1, t2, m_basal, m_size, sol)^2)) * 
    solve(outerintprob, integrator; reltol = 1e-3, abstol = 1e-3).u
    return integral
end


# numerical integrals for death probability using Cubature.jl

function NumericSurvivalProbability1(t1, t2, m_basal, m_size, d_size, d_metas, sol, n)
    integral = (1/(LambdaN(t1, t2, m_basal, m_size, sol))) *
    pquadrature(u1 ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n)*
        Phi(u1, t2, d_size, d_metas, sol, n+1),
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability2(t1, t2, m_basal, m_size, d_size, d_metas, sol, n)
    integral = (2/(LambdaN(t1, t2, m_basal, m_size, sol)^2)) *
    pquadrature(u1 ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n)*
        pquadrature(u2 -> lambdaN(u2, m_basal, m_size, sol)*
            Phi(u1, u2, d_size, d_metas, sol, n+1) *
            Phi(u2, t2,  d_size, d_metas, sol, n+2),
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end


function NumericSurvivalProbability3(t1, t2, m_basal, m_size, d_size, d_metas, sol, n)
    integral = (6/(LambdaN(t1, t2, m_basal, m_size, sol)^3)) *
    pquadrature(u1 ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n)*
        pquadrature(u2 -> lambdaN(u2, m_basal, m_size, sol)*
            Phi(u1, u2, d_size, d_metas, sol, n+1) *
            pquadrature(u3 -> lambdaN(u3, m_basal, m_size, sol) *
                Phi(u2, u3, d_size, d_metas, sol, n+2) *
                Phi(u3, t2, d_size, d_metas, sol, n+3),
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability4(t1, t2, m_basal, m_size, d_size, d_metas, sol, n)
    integral = (24/(LambdaN(t1, t2, m_basal, m_size, sol)^4)) *
    pquadrature(u1 ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n)*
        pquadrature(u2 -> lambdaN(u2, m_basal, m_size, sol)*
            Phi(u1, u2, d_size, d_metas, sol, n+1) *
            pquadrature(u3 -> lambdaN(u3, m_basal, m_size, sol) *
                Phi(u2, u3, d_size, d_metas, sol, n+2) *
                pquadrature(u4 -> lambdaN(u4, m_basal, m_size, sol) *
                            Phi(u3, u4, d_size, d_metas, sol, n+3) *
                            Phi(u4, t2, d_size, d_metas, sol, n+4),
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



function NumericSurvivalProbability5(t1, t2, m_basal, m_size, d_size, d_metas, sol, n)
    integral = 120/((LambdaN(t1, t2, m_basal, m_size, sol))^5) *
    pquadrature(u1 ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n)*
        pquadrature(u2 -> lambdaN(u2, m_basal, m_size, sol)*
            Phi(u1, u2, d_size, d_metas, sol, n+1) *
            pquadrature(u3 -> lambdaN(u3, m_basal, m_size, sol) *
                Phi(u2, u3, d_size, d_metas, sol, n+2) *
                pquadrature(u4 -> lambdaN(u4, m_basal, m_size, sol) *
                    Phi(u3, u4, d_size, d_metas, sol, n+3) *
                    pquadrature(u5 -> lambdaN(u5, m_basal, m_size, sol) *
                        Phi(u4, u5, d_size, d_metas, sol, n+4) *
                        Phi(u5, t2, d_size, d_metas, sol, n+5),
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

function NumericSurvivalProbability6(t1, t2, m_basal, m_size, d_size, d_metas, sol, n)
    integral = 720/((LambdaN(t1, t2, m_basal, m_size, sol))^6) *
    pquadrature(u1 ->lambdaN(u1, m_basal, m_size, sol)*
        Phi(t1, u1, d_size, d_metas, sol, n)*
        pquadrature(u2 -> lambdaN(u2, m_basal, m_size, sol)*
            Phi(u1, u2, d_size, d_metas, sol, n+1) *
            pquadrature(u3 -> lambdaN(u3, m_basal, m_size, sol) *
                Phi(u2, u3, d_size, d_metas, sol, n+2) *
                pquadrature(u4 -> lambdaN(u4, m_basal, m_size, sol) *
                    Phi(u3, u4, d_size, d_metas, sol, n+3) *
                    pquadrature(u5 -> lambdaN(u5, m_basal, m_size, sol) *
                        Phi(u4, u5, d_size, d_metas, sol, n+4) *
                        pquadrature(u6 -> lambdaN(u6, m_basal, m_size, sol) *
                            Phi(u5, u6, d_size, d_metas, sol, n+5) *
                            Phi(u6, t2, d_size, d_metas, sol, n+6),
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
    sol::ODESolution
    )::Real

    # Unpack parameters
    K, μ, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        surv_prob = Phi(t⁻, t, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        surv_prob = NumericSurvivalProbability1(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        surv_prob = NumericSurvivalProbability2(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        surv_prob = NumericSurvivalProbability3(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        surv_prob = NumericSurvivalProbability4(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        surv_prob = NumericSurvivalProbability5(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        surv_prob = NumericSurvivalProbability6(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
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


function NumericDeathProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    sol::ODESolution
    )::Real

    # Unpack parameters
    b, μ, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        death_prob = lambdaD(t, d_size, d_metas, sol, Nt)*Phi(t⁻, t, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, d_size, d_metas, sol, Nt)*NumericSurvivalProbability1(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, d_size, d_metas, sol, Nt)*NumericSurvivalProbability2(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, d_size, d_metas, sol, Nt)*NumericSurvivalProbability3(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, d_size, d_metas, sol, Nt)*NumericSurvivalProbability4(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, d_size, d_metas, sol, Nt)*NumericSurvivalProbability5(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        death_prob = lambdaD(t, d_size, d_metas, sol, Nt)*NumericSurvivalProbability6(t⁻, t, m_basal, m_size, d_size, d_metas, sol, Nt⁻)
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
    sol::ODESolution
    )::Real
    # Unpack parameters
    K, μ, m_basal, m_size, = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    dn = Int(Nt - Nt⁻)
    Lt = LambdaN(t⁻, t, m_basal, m_size, sol)
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
    sol::ODESolution,
    )::Real

    # get observation probability (rather rate since it is not normalized)
    obs_prob = ObservationProbability(Xt, Yt)

    if (Xt[3] == 0.0) # no death

        # get process probability
        process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, sol) * (NumericSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, sol))
    else # death
        process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, sol) * (NumericDeathProbability(t⁻, t, θ, Xt⁻, Xt, sol))
    end
    if (obs_prob < 0.0)
        println("obs_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt, Yt = $Yt\n It is ",obs_prob)
    end
    if (process_prob < 0.0)
        println("process_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt, Yt = $Yt\n 
        and θ=$θ. It is ",process_prob,
        "\n With metastasisProbability = ", MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, sol), "\n",
        "and deathProbability = ", NumericDeathProbability(t⁻, t, θ, Xt⁻, Xt, sol))
    end
    L = obs_prob * process_prob
    return L
end

function PatientLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=1.0,
    sol::ODESolution
    )::Real

    # Unpack data
    timepoints, Bt, Nt, Dt = data

    # get X based on parameters
    St = [sum(sol(t)) for t in timepoints]

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

        l += log(TimepointLikelihood(t⁻, t, θ, Xt⁻, Xt, Yt, sol))
    end

    return l
end

function OdeLogLikelihood(
    θ::Vector{<:Real},
    data;
    S0::Real=1.0,
    )::Real

    # unpack parameter
    b, μ, m_basal, m_size, d_size, d_metas = θ

    n_patients = data.patient_id[end]

    # ODEsolution to apss to likelihood function calculations
    parammap = [sys.μ => μ, sys.b => b, sys.a => 0.03, sys.k => 0.05, sys.r => 0.1, sys.m => 2, sys.d => 0.01]
    initial_conditions = [sys.P => S0, sys.Q => 0.0, sys.De => 0.0]
    prob = ODEProblem(sys, parammap, (0.0, 30.0), initial_conditions)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

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
            S0=S0,
            sol=sol)
    end
    return ll
end

function NegLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=1.0,
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
    sol::ODESolution
    )::Real

    # Unpack parameters
    b, μ, m_basal, m_size, = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    dn = Int(Nt - Nt⁻)
    Lt = LambdaN(t⁻, t, m_basal, m_size, sol)
    met_prob = (Lt^dn)/(factorial(dn))*exp(-Lt)

    return met_prob
end

function OnlyDeathProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    sol::ODESolution
    )::Real

    if (Xt[3] == 0.0) # no death

        # get process probability
        process_prob = (NumericSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, sol))
    else # death
        process_prob = (NumericDeathProbability(t⁻, t, θ, Xt⁻, Xt, sol))
    end
    return process_prob
end


# likelihood functions

function TumorNegLogLikelihood(
    b,
    μ,
    data;
    S0::Real=1.0,
    )::Real

    # ODEsolution to apss to likelihood function calculations
    parammap = [sys.μ => μ, sys.b => b, sys.a => 0.03, sys.k => 0.05, sys.r => 0.1, sys.m => 2, sys.d => 0.01]
    initial_conditions = [sys.P => S0, sys.Q => 0.0, sys.De => 0.0]
    prob = ODEProblem(sys, parammap, (0.0, 30.0), initial_conditions)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

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
        St = [sum(sol(t)) for t in timepoints]

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
    S0::Real=1.0,
)
    # Unpack parameters
    b, μ, m_basal, m_size = θ

    # ODEsolution to apss to likelihood function calculations
    parammap = [sys.μ => μ, sys.b => b, sys.a => 0.03, sys.k => 0.05, sys.r => 0.1, sys.m => 2, sys.d => 0.01]
    initial_conditions = [sys.P => S0, sys.Q => 0.0, sys.De => 0.0]
    prob = ODEProblem(sys, parammap, (0.0, 30.0), initial_conditions)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

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

        St = [sum(sol(t)) for t in timepoints]
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
            l += log(OnlyMetastasisProbability(t⁻, t, θ, Xt⁻, Xt, sol))
        end
        ll += l
    end
    return -ll
end

function DeathNegLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=1.0,
)
    # Unpack parameters
    b, μ, m_basal, m_size, d_size, d_metas = θ

    # ODEsolution to apss to likelihood function calculations
    parammap = [sys.μ => μ, sys.b => b, sys.a => 0.03, sys.k => 0.05, sys.r => 0.1, sys.m => 2, sys.d => 0.01]
    initial_conditions = [sys.P => S0, sys.Q => 0.0, sys.De => 0.0]
    prob = ODEProblem(sys, parammap, (0.0, 30.0), initial_conditions)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

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

        St = [sum(sol(t)) for t in timepoints]

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
            l += log(OnlyDeathProbability(t⁻, t, θ, Xt⁻, Xt, sol))
        end
        ll += l
    end
    return -ll
end


function LogHierarchOptimization(data, x0; S0=1.0, lb=[-1.0, -2.0, -7.0, -9.0, -9.0, -9.0], ub=[1.0, 0.0, -2.0, -4.0, -6.0, -4.0], optimizer="SAMIN", llh_type = "Numeric")

    # get the true parameters from the data
    #beta, m_basal, m_size, d_basal, d_size, d_metas = data.parameters
    #true_par = (beta = beta, m_basal = m_basal, m_size = m_size, d_basal = d_basal, d_size = d_size, d_metas = d_metas);

    if optimizer == "SAMIN"
        optim_alg = SAMIN()
    elseif optimizer =="CMAEvolution"
        optim_alg = Optim.CMAEvolutionStrategyOpt()
    else
        error("Please choose SAMIN or CMAEvolution as optimizer for this function.")
    end

    # first we want to estimate only beta
    gyllenbergOptim(x, p) = TumorNegLogLikelihood(exp(x[1]), exp(x[2]), data, S0=S0)
    gyllenbergOptimFunc = OptimizationFunction(gyllenbergOptim)
    gyllenbergOptimProblem = OptimizationProblem(gyllenbergOptimFunc, x0[1:2], lb=lb[1:2], ub=ub[1:2])
    gyllenbergEst = solve(gyllenbergOptimProblem, SAMIN(), maxiters=10^6)
    println("b:", gyllenbergEst[1], "\n", "μ:", gyllenbergEst[2], "\n")

    # next we want to optimize the metastasis parameters
    metOptim(x, p) = MetastasisNegLogLikelihood(exp.([gyllenbergEst[1], gyllenbergEst[2], x[1], x[2]]), data, S0=S0)
    metOptimFunc = OptimizationFunction(metOptim)
    metOptimProblem = OptimizationProblem(metOptimFunc, x0[3:4], lb=lb[3:4], ub=ub[3:4])
    metEst = solve(metOptimProblem, SAMIN(), maxiters=10^6)
    println("Metastasis:", metEst, "\n")

    # next we want to optimize the death parameters
    deathOptim(x, p) = DeathNegLogLikelihood(exp.([gyllenbergEst[1], gyllenbergEst[2], metEst[1], metEst[2], x[1], x[2]]), data, S0=S0)
    deathOptimFunc = OptimizationFunction(deathOptim)
    deathOptimProblem = OptimizationProblem(deathOptimFunc, x0[5:6], lb=lb[5:6], ub=ub[5:6])
    deathEst = solve(deathOptimProblem, SAMIN(), maxiters=10^8)
    est_par = [gyllenbergEst[1], gyllenbergEst[2], metEst[1], metEst[2], deathEst[1], deathEst[2]]

    # print solutions
    #println("True parameter is \n", true_par)
    println("Estimated parameter is \n", est_par)

    #return true_par, est_par
    return est_par
end

function LogOptimization(data, x0; S0=0.065, lb=[-1.0, -2.0, -7.0, -9.0, -9.0, -9.0], ub=[1.0, 0.0, -2.0, -4.0, -6.0, -4.0], optimizer="SAMIN", llh_type = "Numeric")

    # next we want to optimize
    OptimLlh(x, p) = NegLogLikelihood(exp.(x), data, S0=S0)
    OptimFunc = OptimizationFunction(OptimLlh)

    if optimizer == "SAMIN"
        OptimProblem = OptimizationProblem(OptimFunc, x0, lb=lb, ub=ub)
        optim_alg = SAMIN()
    elseif optimizer =="ParticleSwarm"
        OptimProblem = OptimizationProblem(OptimFunc, x0)
        optim_alg = Optim.ParticleSwarm()
    elseif optimizer =="SimulatedAnnealing"
        OptimProblem = OptimizationProblem(OptimFunc, x0)
        optim_alg = Optim.SimulatedAnnealing()
    elseif optimizer =="CMAEvolution"
        OptimProblem = OptimizationProblem(OptimFunc, x0, lb=lb, ub=ub)
        optim_alg = CMAEvolutionStrategyOpt()
    elseif optimizer =="BlackBox"
        OptimProblem = OptimizationProblem(OptimFunc, x0, lb=lb, ub=ub)
        optim_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
    end

    obj_values = []
    times = []
    function callback(p, l)
        push!(obj_values, l)
        push!(times, time_ns())
        return false
    end

    # run optimization
    joined_est = solve(OptimProblem, optim_alg, maxiters=10^8, callback=callback)
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

end;


#------------------------------------------------------------------------------------------------------------------------------
# # Test area 

# # set model parameter values
# b = 1
# μ = 0.05
# m_basal = 0.04
# m_size = 0.04
# d_size = 0.01
# d_metastasis = 0.01

# p = (b = b, μ = μ, m_basal = 0.04, m_size = 0.04, d_size = 0.01, d_metastasis = 0.01)
# θ = [b, μ, p.m_basal, p.m_size, p.d_size, p.d_metastasis]

# gt_par = Dict(
#     "b" => log10(b),
#     "μ" => log10(μ),
#     "m_basal" => log10(m_basal), 
#     "m_size" => log10(m_size), 
#     "d_size" => log10(d_size), 
#     "d_metastasis" => log10(d_metastasis)
#     )

# # set initial condition 
# P0 = 1.0

# # set time interval
# endtime = 30.0
# timepoints = 0.0:1.0:30.0
# tspan = (0.0, endtime)

# # set chosen observation noise
# sigma = 0.1;


# t1 = 4.0
# t2 = 5.0

# # ODEsolution to apss to likelihood function calculations
# parammap = [sys.μ => μ, sys.b => b, sys.a => 0.03, sys.k => 0.05, sys.r => 0.1, sys.m => 2, sys.d => 0.01]
# initial_conditions = [sys.P => P0, sys.Q => 0.0, sys.De => 0.0]
# prob = ODEProblem(sys, parammap, (0.0, 30.0), initial_conditions)
# sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

# using BenchmarkTools

# @btime NumericSurvivalProbability2(t1, t2, m_basal, m_size, d_size, d_metastasis, sol, 1)

# @btime NumericSurvivalProbability4(t1, t2, m_basal, m_size, d_size, d_metastasis, sol, 1)

# npat = 500
# gyllenberg_patient_df = load("data/simplified_model/gyllenberg_data_$(npat)_patients_$(θ).jld2")["gyllenberg_data"]

# @time TumorNegLogLikelihood(1.0, 0.05, gyllenberg_patient_df, S0=1.0)

# @time MetastasisNegLogLikelihood([1.0, 0.05, 0.04, 0.04], gyllenberg_patient_df, S0=1.0)

# @time DeathNegLogLikelihood([1.0, 0.05, 0.04, 0.04, 0.01, 0.01], gyllenberg_patient_df, S0=1.0)

# @time NegLogLikelihood([1.0, 0.05, 0.04, 0.04, 0.01, 0.01], gyllenberg_patient_df, S0=1.0)