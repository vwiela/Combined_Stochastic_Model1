module TherapyLikelihoodsTest

using DifferentialEquations
using JumpProcesses
using Statistics
using Distributions
using Random
using DifferentialEquations.EnsembleAnalysis
using DataFrames
using JLD2
using ForwardDiff
using Optimization
using OptimizationOptimJL
using NaNMath

using Cubature
using FastGaussQuadrature, LinearAlgebra

# export functions needed
export ObservationProbability, NumericSurvivalProbability, NumericDeathProbability, AnalyticSurvivalProbability, AnalyticDeathProbability
export NegLogLikelihood, TumorNegLogLikelihood, MetastasisNegLogLikelihood, DeathNegLogLikelihood, PatientLogLikelihood
export NumericSurvivalProbability1, NumericSurvivalProbability2, NumericSurvivalProbability3, NumericSurvivalProbability4, NumericSurvivalProbability5, NumericSurvivalProbability6
export lambdaN, LambdaN, lambdaD, LambdaD, Phi

#-----------------------------------------------------------------------------------------------------------------------------------------------
# define all the likelihood functions


# helper functions
function TumorGrowth(t, S0, beta0, rho, delta; BMI=20, treatment_time=nothing)
    if !isnothing(treatment_time)
        if t <= treatment_time
            return S0 * exp(beta0*t)
        else
            if BMI >= 30
                return S0 * exp(beta0*treatment_time + (beta0 - rho+delta)*(t-treatment_time))
            else
                return S0 * exp(beta0*treatment_time + (beta0 - rho)*(t-treatment_time))
            end
        end
    else
        return S0 * exp(beta0*t)
    end
end


function lambdaN(
    t, 
    beta0::Real,
    rho::Real,
    delta::Real,
    m_basal::Real, 
    m_size::Real, 
    S0::Real, 
    treatment_time,
    BMI::Real,
    )
    S = TumorGrowth(t, S0, beta0, rho, delta, treatment_time=treatment_time, BMI=BMI)
    return m_basal + m_size * sqrt(S)
end

function lambdaD(
    t, 
    beta0::Real,
    rho::Real,
    delta::Real,
    d_size::Real, 
    d_metas::Real, 
    S0::Real, 
    Nt,
    treatment_time,
    BMI::Real,
    )

    S = TumorGrowth(t, S0, beta0, rho, delta, treatment_time=treatment_time, BMI=BMI)
    return d_size * sqrt(S) + d_metas * Nt
end

function Phi(
    t1, 
    t2, 
    beta::Real, 
    d_size::Real, 
    d_metas::Real, 
    S0::Real, 
    n
    )

    return exp((2*d_size*(sqrt(exp(beta*t1)*S0) - sqrt(exp(beta*t2)*S0)))/beta + (d_metas*n)*(t1 - t2))
end

function LambdaN(
    t1, 
    t2, 
    beta::Real, 
    m_basal::Real, 
    m_size::Real, 
    S0::Real
    )

    dt = (t2-t1)
    return dt*m_basal+ (2*m_size*sqrt(S0))/beta*(sqrt(exp(beta*t2))-sqrt(exp(beta*t1)))
end

function LambdaD(
    t1,
    t2,
    beta::Real,
    d_size::Real,
    d_metas::Real,
    S0::Real,
    n::Real,
)
    dt = t2-t1

    return dt*(n*d_metas)+(2*d_size*sqrt(S0))/beta*(sqrt(exp(beta*t2))-sqrt(exp(beta*t1)))
end

# analytical integrals for death probability

function AnalyticSurvivalProbability1(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    return (beta*exp((2*d_size*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + d_metas*n*t1 - d_metas*(1 + 
    n)*t2)*(beta*(exp(d_metas*t1) - exp(d_metas*t2))*m_basal + 
    2*d_metas*exp(d_metas*t1)*(m_basal + m_size*sqrt(exp(beta*t1)*S0)) - 
    2*d_metas*exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))))/(d_metas*(beta + 
    2*d_metas)*(2*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0) + beta*m_basal*(t1 - t2)))
end

function AnalyticSurvivalProbability2(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    return (beta^2*exp((2*d_size*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + d_metas*n*t1 - d_metas*(2 + 
    n)*t2)*(beta^2*(exp(d_metas*t1) - exp(d_metas*t2))^2*m_basal^2 + 
    4*beta*d_metas*(exp(d_metas*t1) - 
    exp(d_metas*t2))*m_basal*(exp(d_metas*t1)*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) - exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))) + 4*d_metas^2*(exp((beta + 
    2*d_metas)*t1)*m_size^2*S0 + exp((beta + 2*d_metas)*t2)*m_size^2*S0 + 
    exp(2*d_metas*t1)*m_basal*(m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) -
    2*exp(d_metas*(t1 + t2))*(m_basal + m_size*sqrt(exp(beta*t1)*S0)) * 
    (m_basal + m_size*sqrt(exp(beta*t2)*S0)) + exp(2*d_metas*t2)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)))))/(d_metas^2*(beta + 
    2*d_metas)^2*(2*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0) + beta*m_basal*(t1 - t2))^2)
end

function AnalyticSurvivalProbability3(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    return (exp((2*d_size*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + d_metas*n*t1 - d_metas*(3 + 
    n)*t2)*(-(beta^3*(exp(d_metas*t1) - exp(d_metas*t2))^3*m_basal^3) - 
    6*beta^2*d_metas*(exp(d_metas*t1) - 
    exp(d_metas*t2))^2*m_basal^2*(exp(d_metas*t1)*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) - exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))) - 12*beta*d_metas^2*(exp(d_metas*t1) - 
    exp(d_metas*t2))*m_basal*(exp((beta + 2*d_metas)*t1)*m_size^2*S0 + 
    exp((beta + 2*d_metas)*t2)*m_size^2*S0 + 
    exp(2*d_metas*t1)*m_basal*(m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) 
    - 2*exp(d_metas*(t1 + t2))*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + exp(2*d_metas*t2)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0))) - 8*d_metas^3*(3*exp(d_metas*t1 + 
    beta*t2 + 2*d_metas*t2)*m_size^2*S0*(m_basal +
    m_size*sqrt(exp(beta*t1)*S0)) + exp((beta + 
    3*d_metas)*t1)*m_size^2*S0*(3*m_basal + m_size*sqrt(exp(beta*t1)*S0)) 
    + exp(3*d_metas*t1)*m_basal^2*(m_basal + 
    3*m_size*sqrt(exp(beta*t1)*S0)) - 3*exp(beta*t1 + 2*d_metas*t1 + 
    d_metas*t2)*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t2)*S0)) - 
    3*exp(d_metas*(2*t1 + t2))*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - exp((beta + 
    3*d_metas)*t2)*m_size^2*S0*(3*m_basal + m_size*sqrt(exp(beta*t2)*S0)) 
    + 3*exp(d_metas*(t1 + 2*t2))*m_basal*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 
    exp(3*d_metas*t2)*m_basal^2*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)))))/(d_metas^3*(beta + 
    2*d_metas)^3*((2*(-sqrt(exp(beta*t1)) + 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0))/beta + m_basal*(-t1 + t2))^3)
end

function AnalyticSurvivalProbability4(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    return (beta^4*exp((2*d_size*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + d_metas*n*t1 - d_metas*(4 + 
    n)*t2)*(beta^4*(exp(d_metas*t1) - exp(d_metas*t2))^4*m_basal^4 + 
    8*beta^3*d_metas*(exp(d_metas*t1) - exp(d_metas*t2))^3*m_basal^3*(exp(d_metas*t1)*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) - exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))) + 16*d_metas^4*(exp(2*(beta + 
    2*d_metas)*t1)*m_size^4*S0^2 + exp(2*(beta + 
    2*d_metas)*t2)*m_size^4*S0^2 + 6*exp((beta + 2*d_metas)*(t1 + 
    t2))*m_size^4*S0^2 - 12*exp(d_metas*t1 + beta*t2 + 
    3*d_metas*t2)*m_basal*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) - 4*exp(d_metas*(t1 + 
    3*t2))*m_size^3*(exp(beta*t2)*S0)^1.5*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 6*exp(beta*t2 + 2*d_metas*(t1 + 
    t2))*m_basal*m_size^2*S0*(m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) + 
    2*exp((beta + 4*d_metas)*t1)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0)) + exp(4*d_metas*t1)*m_basal^3*(m_basal + 
    4*m_size*sqrt(exp(beta*t1)*S0)) - 4*exp(beta*t1 + 3*d_metas*t1 + 
    d_metas*t2)*m_size^2*S0*(3*m_basal +  m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 4*exp(d_metas*(3*t1 + 
    t2))*m_basal^2*(m_basal + 3*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + 6*exp(beta*t1 + 2*d_metas*(t1 + 
    t2))*m_basal*m_size^2*S0*(m_basal + 2*m_size*sqrt(exp(beta*t2)*S0)) + 
    6*exp(2*d_metas*(t1 + t2))*m_basal^2*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) + 2*exp((beta + 
    4*d_metas)*t2)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 4*exp(d_metas*(t1 + 
    3*t2))*m_basal^2*(m_basal + m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)) + exp(4*d_metas*t2)*m_basal^3*(m_basal + 
    4*m_size*sqrt(exp(beta*t2)*S0))) + 
    24*beta^2*d_metas^2*m_basal^2*(exp((beta + 4*d_metas)*t1)*m_size^2*S0 + 
    exp((beta + 4*d_metas)*t2)*m_size^2*S0 - 2*exp(beta*t1 + 
    3*d_metas*t1 + d_metas*t2)*m_size^2*S0 - 2*exp(d_metas*t1 + beta*t2 + 
    3*d_metas*t2)*m_size^2*S0 + exp(beta*t1 + 2*d_metas*(t1 + 
    t2))*m_size^2*S0 + exp(beta*t2 + 2*d_metas*(t1 + t2))*m_size^2*S0 + 
    exp(4*d_metas*t1)*m_basal*(m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) 
    + exp(4*d_metas*t2)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) + exp(2*d_metas*(t1 + t2))*(6*m_basal^2 + 
    4*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    6*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    2*exp(d_metas*(3*t1 + t2))*(2*m_basal^2 + 
    m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    m_basal*m_size*(3*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    2*exp(d_metas*(t1 + 3*t2))*(2*m_basal^2 + 
    m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    m_basal*m_size*(sqrt(exp(beta*t1)*S0) + 3*sqrt(exp(beta*t2)*S0)))) + 
    32*beta*d_metas^3*m_basal*(-(exp(d_metas*(3*t1 + 
    t2))*m_size^3*(exp(beta*t1)*S0)^1.5) + 3*exp(beta*t2 + 2*d_metas*(t1 
    + t2))*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t1)*S0)) + 
    exp((beta + 4*d_metas)*t1)*m_size^2*S0*(3*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + exp(4*d_metas*t1)*m_basal^2*(m_basal 
    + 3*m_size*sqrt(exp(beta*t1)*S0)) + 3*exp(beta*t1 + 2*d_metas*(t1 + 
    t2))*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t2)*S0)) - 
    3*exp(beta*t1 + 3*d_metas*t1 + d_metas*t2)*m_size^2*S0*(2*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + exp((beta + 
    4*d_metas)*t2)*m_size^2*S0*(3*m_basal + m_size*sqrt(exp(beta*t2)*S0)) 
    + exp(4*d_metas*t2)*m_basal^2*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)) + 3*exp(2*d_metas*(t1 + 
    t2))*m_basal*(2*m_basal^2 + 4*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    3*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    exp(beta*t2 + d_metas*(t1 + 3*t2))*m_size^2*S0*(6*m_basal + 
    m_size*(3*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    exp(d_metas*(3*t1 + t2))*m_basal*(4*m_basal^2 + 
    6*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    3*m_basal*m_size*(3*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    exp(d_metas*(t1 + 3*t2))*m_basal*(4*m_basal^2 + 
    6*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    3*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + 
    3*sqrt(exp(beta*t2)*S0))))))/(d_metas^4*(beta + 
    2*d_metas)^4*(2*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0) + beta*m_basal*(t1 - t2))^4)
end

function AnalyticSurvivalProbability5(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    return -((beta^5*exp((2*d_size*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + d_metas*n*(t1 - t2) - 
    5*d_metas*t2)*(-(beta^5*(exp(d_metas*t1) - 
    exp(d_metas*t2))^5*m_basal^5) - 10*beta^4*d_metas*(exp(d_metas*t1) - 
    exp(d_metas*t2))^4*m_basal^4*(exp(d_metas*t1)*(m_basal +
    m_size*sqrt(exp(beta*t1)*S0)) - exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))) - 32*d_metas^5*(10*exp((beta + 
    5*d_metas)*t1)*m_basal^2*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 5*exp(d_metas*t1 + 2*beta*t2 + 
    4*d_metas*t2)*m_size^4*S0^2*(m_basal + m_size*sqrt(exp(beta*t1)*S0)) 
    + 10*exp(beta*t1 + 3*d_metas*t1 + beta*t2 + 2*d_metas*t2)*m_size^4*S0^2*(3*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + exp((2*beta + 5*d_metas)*t1)*m_size^4*S0^2*(5*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 10*exp(3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_basal^2*m_size^2*S0*(m_basal + 
    3*m_size*sqrt(exp(beta*t1)*S0)) + exp(5*d_metas*t1)*m_basal^4*(m_basal + 
    5*m_size*sqrt(exp(beta*t1)*S0)) - 10*exp((beta + 
    5*d_metas)*t2)*m_basal^2*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 5*exp(2*beta*t1 + 4*d_metas*t1 + 
    d_metas*t2)*m_size^4*S0^2*(m_basal + m_size*sqrt(exp(beta*t2)*S0)) - 
    10*exp(beta*t1 + d_metas*(4*t1 + t2))*m_basal*m_size^2*S0*(3*m_basal 
    + 2*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 5*exp(d_metas*(4*t1 + 
    t2))*m_basal^3*(m_basal + 4*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 10*exp(beta*t1 + 2*d_metas*t1 + 
    beta*t2 + 3*d_metas*t2)*m_size^4*S0^2*(3*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 10*exp(2*d_metas*t1 + beta*t2 + 
    3*d_metas*t2)*m_basal*m_size^2*S0*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0))*(3*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - exp((2*beta + 
    5*d_metas)*t2)*m_size^4*S0^2*(5*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + 10*exp(beta*t1 + 3*d_metas*t1 + 
    2*d_metas*t2)*m_basal*m_size^2*S0*(3*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) + 10*exp(3*d_metas*t1 + 
    2*d_metas*t2)*m_basal^3*(m_basal + 3*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) + 10*exp(beta*t2 + d_metas*(t1 + 
    4*t2))*m_basal*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t1)*S0))*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 10*exp(beta*t1 + 2*d_metas*t1 + 
    3*d_metas*t2)*m_basal^2*m_size^2*S0*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)) - 10*exp(2*d_metas*t1 + 
    3*d_metas*t2)*m_basal^3*(m_basal + 2*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)) + 5*exp(d_metas*(t1 + 
    4*t2))*m_basal^3*(m_basal + m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    4*m_size*sqrt(exp(beta*t2)*S0)) - exp(5*d_metas*t2)*m_basal^4*(m_basal + 
    5*m_size*sqrt(exp(beta*t2)*S0))) - 
    40*beta^3*d_metas^2*m_basal^3*(exp((beta + 5*d_metas)*t1)*m_size^2*S0 
    - exp((beta + 5*d_metas)*t2)*m_size^2*S0 - 3*exp(beta*t1 + 
    4*d_metas*t1 + d_metas*t2)*m_size^2*S0 + 3*exp(beta*t1 + 3*d_metas*t1 
    + 2*d_metas*t2)*m_size^2*S0 + exp(3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_size^2*S0 - exp(beta*t1 + 2*d_metas*t1 + 
    3*d_metas*t2)*m_size^2*S0 - 3*exp(2*d_metas*t1 + beta*t2 + 
    3*d_metas*t2)*m_size^2*S0 + 3*exp(beta*t2 + d_metas*(t1 +     
    4*t2))*m_size^2*S0 + exp(5*d_metas*t1)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0)) - exp(5*d_metas*t2)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) + 2*exp(3*d_metas*t1 + 
    2*d_metas*t2)*(5*m_basal^2 + 6*m_basal*m_size*sqrt(exp(beta*t1)*S0) + 
    4*m_basal*m_size*sqrt(exp(beta*t2)*S0) + 
    3*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0)) - 
    2*exp(2*d_metas*t1 + 3*d_metas*t2)*(5*m_basal^2 + 
    4*m_basal*m_size*sqrt(exp(beta*t1)*S0) + 
    6*m_basal*m_size*sqrt(exp(beta*t2)*S0) + 
    3*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0)) - 
    exp(d_metas*(4*t1 + t2))*(5*m_basal^2 + 
    2*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    2*m_basal*m_size*(4*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) + 
    exp(d_metas*(t1 + 4*t2))*(5*m_basal^2 + 
    2*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    2*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + 4*sqrt(exp(beta*t2)*S0)))) - 
    80*beta^2*d_metas^3*m_basal^2*(3*exp(3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t1)*S0)) + 
    exp((beta + 5*d_metas)*t1)*m_size^2*S0*(3*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + exp(5*d_metas*t1)*m_basal^2*(m_basal 
    + 3*m_size*sqrt(exp(beta*t1)*S0)) - 3*exp(beta*t1 + 2*d_metas*t1 + 
    3*d_metas*t2)*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t2)*S0)) - 
    exp((beta + 5*d_metas)*t2)*m_size^2*S0*(3*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + exp(beta*t2 + d_metas*(t1 + 
    4*t2))*m_size^2*S0*(9*m_basal + 3*m_size*sqrt(exp(beta*t1)*S0) + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - exp(5*d_metas*t2)*m_basal^2*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)) - exp(beta*t1 + d_metas*(4*t1 + 
    t2))*m_size^2*S0*(9*m_basal + 2*m_size*sqrt(exp(beta*t1)*S0) + 
    3*m_size*sqrt(exp(beta*t2)*S0)) + 2*exp(3*d_metas*t1 + 
    2*d_metas*t2)*m_basal*(5*m_basal^2 + 9*m_basal*m_size*sqrt(exp(beta*t1)*S0) + 
    6*m_basal*m_size*sqrt(exp(beta*t2)*S0) + 
    9*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0)) - 
    2*exp(2*d_metas*t1 + 3*d_metas*t2)*m_basal*(5*m_basal^2 + 
    6*m_basal*m_size*sqrt(exp(beta*t1)*S0) + 9*m_basal*m_size*sqrt(exp(beta*t2)*S0) + 
    9*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0)) - 
    exp(d_metas*(4*t1 + t2))*m_basal*(5*m_basal^2 + 
    6*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    3*m_basal*m_size*(4*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    exp(2*d_metas*t1 + beta*t2 + 3*d_metas*t2)*m_size^2*S0*(9*m_basal + 
    m_size*(6*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) + 
    exp(d_metas*(t1 + 4*t2))*m_basal*(5*m_basal^2 + 
    6*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    3*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + 4*sqrt(exp(beta*t2)*S0))) + 
    exp(beta*t1 + 3*d_metas*t1 + 2*d_metas*t2)*m_size^2*S0*(9*m_basal + 
    m_size*(sqrt(exp(beta*t1)*S0) + 6*sqrt(exp(beta*t2)*S0)))) - 
    80*beta*d_metas^4*m_basal*(exp((2*beta + 5*d_metas)*t1)*m_size^4*S0^2 - 
    exp((2*beta + 5*d_metas)*t2)*m_size^4*S0^2 - exp(2*beta*t1 + 
    4*d_metas*t1 + d_metas*t2)*m_size^4*S0^2 + 6*exp(beta*t1 + 
    3*d_metas*t1 + beta*t2 + 2*d_metas*t2)*m_size^4*S0^2 - 6*exp(beta*t1 +
    2*d_metas*t1 + beta*t2 + 3*d_metas*t2)*m_size^4*S0^2 + 
    exp(d_metas*t1 + 2*beta*t2 + 4*d_metas*t2)*m_size^4*S0^2 + 
    4*exp(d_metas*(t1 + 4*t2))*m_size^3*(exp(beta*t2)*S0)^1.5*(2*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 6*exp(3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_basal*m_size^2*S0*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0)) + 2*exp((beta + 
    5*d_metas)*t1)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0)) + 6*exp(beta*t2 + d_metas*(t1 + 
    4*t2))*m_basal*m_size^2*S0*(3*m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) + 
    exp(5*d_metas*t1)*m_basal^3*(m_basal + 
    4*m_size*sqrt(exp(beta*t1)*S0)) - 4*exp(d_metas*(4*t1 + 
    t2))*m_size^3*(exp(beta*t1)*S0)^1.5*(2*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 6*exp(beta*t1 + 2*d_metas*t1 + 
    3*d_metas*t2)*m_basal*m_size^2*S0*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 2*exp((beta + 
    5*d_metas)*t2)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 6*exp(beta*t1 + d_metas*(4*t1 + 
    t2))*m_basal*m_size^2*S0*(3*m_basal + 2*m_size*sqrt(exp(beta*t2)*S0)) 
    - exp(5*d_metas*t2)*m_basal^3*(m_basal + 
    4*m_size*sqrt(exp(beta*t2)*S0)) - exp(d_metas*(4*t1 + t2))*m_basal^2*(5*m_basal^2 +
    12*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    4*m_basal*m_size*(4*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    2*exp(2*d_metas*t1 + beta*t2 + 3*d_metas*t2)*m_size^2*S0*(9*m_basal^2 
    + 2*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    2*m_basal*m_size*(6*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) + 
    2*exp(3*d_metas*t1 + 2*d_metas*t2)*m_basal^2*(5*m_basal^2 + 
    18*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    4*m_basal*m_size*(3*sqrt(exp(beta*t1)*S0) + 2*sqrt(exp(beta*t2)*S0))) 
    - 2*exp(2*d_metas*t1 + 3*d_metas*t2)*m_basal^2*(5*m_basal^2 + 
    18*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    4*m_basal*m_size*(2*sqrt(exp(beta*t1)*S0) + 3*sqrt(exp(beta*t2)*S0))) 
    + exp(d_metas*(t1 + 4*t2))*m_basal^2*(5*m_basal^2 + 
    12*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    4*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + 4*sqrt(exp(beta*t2)*S0))) + 
    2*exp(beta*t1 + 3*d_metas*t1 + 2*d_metas*t2)*m_size^2*S0*(9*m_basal^2 
    + 2*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    2*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + 
    6*sqrt(exp(beta*t2)*S0))))))/(d_metas^5*(beta + 
    2*d_metas)^5*(2*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0) + beta*m_basal*(t1 - t2))^5))
end

function NumericSurvivalProbability1(t1, t2, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, n, BMI; treatment_time=nothing, reltol=1e-8)
    integral = (1/(LambdaN(t1, t2, beta, m_basal, m_size, S0))) *
    hquadrature(u1 ->lambdaN(u1, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        Phi(u1, t2, beta, d_size, d_metas, S0, n+1),
    t1, t2,
    reltol = reltol)[1]
    return integral
end

function NumericSurvivalProbability2(t1, t2, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, n, BMI; treatment_time=nothing, reltol=1e-8)
    integral = (2/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^2)) *
    hquadrature(u1 ->lambdaN(u1, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            Phi(u2, t2,  beta, d_size, d_metas, S0, n+2),
        u1, t2
        )[1],
    t1, t2,
    reltol = reltol)[1]
    return integral
end

function NumericSurvivalProbability3(t1, t2, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, n, BMI; treatment_time=nothing, reltol=1e-8)
    integral = (6/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^3)) *
    hquadrature(u1 ->lambdaN(u1, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                Phi(u2, u3, beta, d_size, d_metas, S0, n+2) *
                Phi(u3, t2, beta, d_size, d_metas, S0, n+3),
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2,
    reltol = reltol)[1]
    return integral
end


function NumericSurvivalProbability4(t1, t2, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, n, BMI; treatment_time=nothing, reltol=1e-8)
    integral = (24/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^4)) *
    hquadrature(u1 ->lambdaN(u1, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                Phi(u2, u3, beta, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                            Phi(u3, u4, beta, d_size, d_metas, S0, n+3) *
                            Phi(u4, t2, beta, d_size, d_metas, S0, n+4),
                u3, t2
                )[1],
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2,
    reltol = reltol)[1]
    return integral
end

function NumericSurvivalProbability5(t1, t2, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, n, BMI; treatment_time=nothing, reltol=1e-8)
    integral = 120/((LambdaN(t1, t2, beta, m_basal, m_size, S0))^5) *
    hquadrature(u1 ->lambdaN(u1, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                Phi(u2, u3, beta, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                    Phi(u3, u4, beta, d_size, d_metas, S0, n+3) *
                    hquadrature(u5 -> lambdaN(u5, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                        Phi(u4, u5, beta, d_size, d_metas, S0, n+4) *
                        Phi(u5, t2, beta, d_size, d_metas, S0, n+5),
                    u4, t2
                    )[1],
                u3, t2
                )[1],
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2,
    reltol = reltol)[1]
    return integral
end

function NumericSurvivalProbability6(t1, t2, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, n, BMI; treatment_time=nothing, reltol=1e-8)
    integral = 720/((LambdaN(t1, t2, beta, m_basal, m_size, S0))^6) *
    hquadrature(u1 ->lambdaN(u1, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                Phi(u2, u3, beta, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                    Phi(u3, u4, beta, d_size, d_metas, S0, n+3) *
                    hquadrature(u5 -> lambdaN(u5, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                        Phi(u4, u5, beta, d_size, d_metas, S0, n+4) *
                        hquadrature(u6 -> lambdaN(u6, beta0, rho, delta, m_basal, m_size, S0, treatment_time, BMI) *
                            Phi(u5, u6, beta, d_size, d_metas, S0, n+5) *
                            Phi(u6, t2, beta, d_size, d_metas, S0, n+6),
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
    t1, t2,
    reltol=reltol)[1]
    return integral
end

# important probability functions

function ObservationProbability(
    Xt::Vector{<:Union{Nothing, Real}},
    Yt; 
    sigma::Real=0.1 # For now with fixed noise level (can be changed to estimated noise later)
    )::Real

    # unpack data
    St, _ = Xt #Size, Metas, Death, Treatment, BMI
    Bt = Yt[1]

    # Compute the likelihood of the tumor size observation
    obs_prob = pdf(Normal(St, sigma*St), Bt)

    return obs_prob
end

function NumericSurvivalProbability(
    t⁻, 
    t, 
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}},
    S0::Real;
    treatment_time::Union{Nothing, Real}=nothing,
    reltol = 1e-8
    )::Real

    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt, Tt, BMIt = Xt #Size, Metas, Death, Treatment, BMI
    St⁻, Nt⁻, Dt⁻, Tt⁻, BMIt⁻ = Xt⁻ #Size, Metas, Death, Treatment, BMI

    # Check for the treament and BMI
    if !isnothing(treatment_time) && treatment_time <= t⁻
        if BMIt >= 30
            beta = beta0 - rho + delta
        else
            beta = beta0 - rho
        end
    else
        beta = beta0
    end

    if (Nt == Nt⁻)
        surv_prob = Phi(t⁻, t, beta, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        surv_prob = NumericSurvivalProbability1(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time=treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 2)
        surv_prob = NumericSurvivalProbability2(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time=treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 3)
        surv_prob = NumericSurvivalProbability3(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time=treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 4)
        surv_prob = NumericSurvivalProbability4(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time=treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 5)
        surv_prob = NumericSurvivalProbability5(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time=treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 6)
        surv_prob = NumericSurvivalProbability6(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time=treatment_time, reltol = reltol)
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


function SurvivalProbability(
    t⁻, 
    t, 
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}},
    S0::Real;
    treatment_time::Union{Nothing, Real}=nothing
    )::Real

    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt, Tt, BMIt = Xt #Size, Metas, Death, Treatment, BMI
    St⁻, Nt⁻, Dt⁻, Tt⁻, BMIt⁻ = Xt⁻ #Size, Metas, Death, Treatment, BMI
    
    # Check for the treament and BMI
    if !isnothing(treatment_time) && treatment_time <= t⁻
        if BMIt >= 30
            beta = beta0 - rho + delta
        else
            beta = beta0 - rho
        end
    else
        beta = beta0
    end

    if (Nt == Nt⁻)
        surv_prob = Phi(t⁻, t, beta, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        surv_prob = AnalyticSurvivalProbability1(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        surv_prob = AnalyticSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        surv_prob = AnalyticSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        surv_prob = AnalyticSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        surv_prob = AnalyticSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    else
        println("more than 5 metastasis in one interval, $(Nt-Nt⁻)")
        surv_prob = 1.0
    end
    # did not observe more than 6 jumps in one time interval.

  
    if (surv_prob < 0.0)
        println("surv_prob is negative for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_size, d_metas])
        surv_prob=NumericSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0, treatment_time=treatment_time)
    end
    # if (surv_prob > 1.0)
    #     println("surv_prob is bigger 1 for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_size, d_metas])
    # end

    return surv_prob
end

function NumericDeathProbability(
    t⁻, 
    t, 
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}},
    S0::Real;
    treatment_time::Union{Nothing, Real}=nothing,
    reltol = 1e-8
    )::Real

    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt, Tt, BMIt = Xt #Size, Metas, Death, Treatment, BMI
    St⁻, Nt⁻, Dt⁻, Tt⁻, BMIt⁻ = Xt⁻ #Size, Metas, Death, Treatment, BMI
    
    # Check for the treament and BMI
    if Tt⁻ == 1
        if BMIt >= 30
            beta = beta0 - rho + delta
        else
            beta = beta0 - rho
        end
    else
        beta = beta0
    end

    if (Nt == Nt⁻)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*Phi(t⁻, t, beta, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*NumericSurvivalProbability1(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0,  Nt⁻, BMIt, treatment_time = treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*NumericSurvivalProbability2(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time = treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*NumericSurvivalProbability3(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time = treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*NumericSurvivalProbability4(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time = treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*NumericSurvivalProbability5(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time = treatment_time, reltol = reltol)
    elseif (Nt == Nt⁻ + 6)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*NumericSurvivalProbability6(t⁻, t, beta, beta0, rho, delta, m_basal, m_size, d_size, d_metas, S0, Nt⁻, BMIt, treatment_time = treatment_time, reltol = reltol)
    end

  
    if (death_prob < 0.0)
        println("death_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
        death_prob=1e-50
    end
    # if (death_prob > 1.0)
    #     println("death_prob is bigger 1 for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
    # end

    return death_prob
end


function DeathProbability(
    t⁻, 
    t, 
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}},
    S0::Real;
    treatment_time::Union{Nothing, Real}=nothing
    )::Real

    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt, Tt, BMIt = Xt #Size, Metas, Death, Treatment, BMI
    St⁻, Nt⁻, Dt⁻, Tt⁻, BMIt⁻ = Xt⁻ #Size, Metas, Death, Treatment, BMI
    
    # Check for the treament and BMI
    if !isnothing(treatment_time) && treatment_time <= t⁻ 
        if BMIt >= 30
            beta = beta0 - rho + delta
        else
            beta = beta0 - rho
        end
    else
        beta = beta0
    end

    if (Nt == Nt⁻)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*Phi(t⁻, t, beta, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*AnalyticSurvivalProbability1(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0,  Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*AnalyticSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*AnalyticSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*AnalyticSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, beta0, rho, delta, d_size, d_metas, S0, Nt, treatment_time, BMIt)*AnalyticSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    else
        ValueError("More than 5 new metastasis observed.")
    end

  
    if (death_prob < 0.0)
        println("death_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
        death_prob=NumericDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0)
    end
    # if (death_prob > 1.0)
    #     println("death_prob is bigger 1 for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
    # end

    return death_prob
end


function MetastasisProbability(
    t⁻, 
    t, 
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}},
    S0::Real;
    treatment_time::Union{Nothing, Real}=nothing
    )::Real

    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, d_size, d_metas = θ
    
    # Unpack data
    St, Nt, Dt, Tt, BMIt = Xt #Size, Metas, Death, Treatment, BMI
    St⁻, Nt⁻, Dt⁻, Tt⁻, BMIt⁻ = Xt⁻ #Size, Metas, Death, Treatment, BMI
    
    # Check for the treament and BMI
    if !isnothing(treatment_time) && treatment_time <= t⁻ 
        if BMIt >= 30
            beta = beta0 - rho + delta
        else
            beta = beta0 - rho
        end
    else
        beta = beta0
    end

    dn = Int(Nt - Nt⁻)
    Lt = LambdaN(t⁻, t, beta, m_basal, m_size, S0)
    met_prob = (Lt^dn)/(factorial(dn))*exp(-Lt)

    return met_prob
end

# likelihood functions

function TimepointLikelihood(
    t⁻,
    t, 
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}}, 
    Yt,
    S0::Real;
    treatment_time::Union{Nothing, Real}=nothing
    )::Real

    # get observation probability (rather rate since it is not normalized)
    obs_prob = ObservationProbability(Xt, Yt)

    if (Xt[3] == 0.0) # no death

        # get process probability
        process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0, treatment_time=treatment_time) * (SurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0, treatment_time=treatment_time))
    else # death
        process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0, treatment_time=treatment_time) * (DeathProbability(t⁻, t, θ, Xt⁻, Xt, S0, treatment_time=treatment_time))
    end
    if (obs_prob < 0.0)
        println("obs_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt, Yt = $Yt\n It is ",obs_prob)
    end
    if (process_prob < 0.0)
        println("process_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt, Yt = $Yt\n 
        and θ=$θ. It is ",process_prob,
        "\n With metastasisProbability = ", MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0), "\n",
        "and deathProbability = ", DeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    end
    L = obs_prob * process_prob
    return L
end

function PatientLogLikelihood(
    θ, 
    data; 
    S0::Real=0.05
    )::Real

    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, d_size, d_metas = θ
    
    # Unpack data
    timepoints, Bt, Nt, Dt, Tt, BMIt = data

    if isnothing(findfirst(Tt .== 1))
        treatment_time = nothing
    else
        treatment_time = timepoints[Tt .== 1][1]
    end
    
    # Initialize loglikelihood
    l = 0.0

    # Loop over data points
    for i in eachindex(timepoints)
        t = timepoints[i]

        St = TumorGrowth(t, S0, beta0, rho, delta, treatment_time=treatment_time, BMI=BMIt[i])

        if (i == 1)
            t⁻ = 0.0
            Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
            Yt = [Bt[i], Nt[i], Dt[i], Tt[i], BMIt[i]]
            Xt⁻ = [S0, 0, 0, 0, BMIt[1]]
        else
            # Set t⁻1
            t⁻ = timepoints[i-1]

            St⁻ = TumorGrowth(t⁻, S0, beta0, rho, delta, treatment_time=treatment_time, BMI=BMIt[i])

            # Set Xt, Yt, Xt⁻1
            Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]]
            Yt = [Bt[i], Nt[i], Dt[i], Tt[i], BMIt[i]]
            Xt⁻ = [St⁻, Nt[i-1], Dt[i-1], Tt[i-1], BMIt[i-1]]
        end

        l += NaNMath.log(TimepointLikelihood(t⁻, t, θ, Xt⁻, Xt, Yt, S0, treatment_time=treatment_time))
    end

    return l
end

function OdeLogLikelihood(
    θ,
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
            patient_data.death,
            patient_data.treatment,
            patient_data.BMIt
            ],
            S0=S0)
    end
    return ll
end

function NegLogLikelihood(
    θ, 
    data; 
    S0::Real=0.065
    )::Real

    return -OdeLogLikelihood(θ, data, S0=S0)
end

#------------------------------------------------------------------------------------------------------------------------------------------------
# Numeric Likelihood functions for comparison of efficiency

function NumericNegLogLikelihood(
    θ, 
    data; 
    S0::Real=0.065,
    reltol = 1e-8
    )::Real

    n_patients = data.patient_id[end]

    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, d_size, d_metas = θ

    ll = 0.0

    for i in 1:n_patients
        patient_data = data[data.patient_id .== i, :]

        # Unpack data
        timepoints, Bt, Nt, Dt = [patient_data.time, 
                                  patient_data.tumor, 
                                  patient_data.metastasis, 
                                  patient_data.death,
                                  patient_data.treatment,
                                  patient_data.BMIt
        ]



        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)
            t = timepoints[i]

            # Check for the treament and BMI
            if Tt[i] == 1
                if BMIt[i] >= 30
                    beta = beta0 - rho + delta
                else
                    beta = beta0 - rho
                end
            else
                beta = beta0
            end

            St = S0*exp(beta*t)
            if (i == 1)
                t⁻ = 0.0
                Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
                Yt = [Bt[i], Nt[i], Dt[i], Tt[i], BMIt[i]]
                Xt⁻ = [S0, 0, 0]
            else
                # Set t, t⁻1
                t = timepoints[i]
                t⁻ = timepoints[i-1]

                # Set Xt, Yt, Xt⁻1
                Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]]
                Yt = [Bt, Nt[i], Dt[i]]
                Xt⁻ = [St[i-1], Nt[i-1], Dt[i-1], Tt[i-1], BMIt[i-1]]
            end

            # get observation probability (rather rate since it is not normalized)
            obs_prob = ObservationProbability(Xt, Yt)

            if (Xt[3] == 0.0) # no death

                # get process probability
                process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0) * (NumericSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0, reltol=reltol))
            else # death
                process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0) * (NumericDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0, reltol=reltol))
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
            l +=log(obs_prob * process_prob)          
        end
        ll += l
    end
    return -ll
end

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
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}},
    S0::Real
    )::Real

    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, = θ

    # Unpack data
    St, Nt, Dt, Tt, BMIt = Xt
    St⁻, Nt⁻, Dt⁻, Tt⁻, BMIt⁻ = Xt⁻

    # Check for the treament and BMI
    if Tt == 1
        if BMIt == 1
            beta = beta0 - rho + delta
        else
            beta = beta0 - rho
        end
    else
        beta = beta0
    end

    dn = Int(Nt - Nt⁻)
    Lt = LambdaN(t⁻, t, beta, m_basal, m_size, S0)
    met_prob = (Lt^dn)/(factorial(dn))*exp(-Lt)

    return met_prob
end

function OnlyDeathProbability(
    t⁻, 
    t, 
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}},
    S0::Real
    )::Real

    if (Xt[3] == 0.0) # no death

        # get process probability
        process_prob = (SurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    else # death
        process_prob = (DeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    end
    return process_prob
end

function NumericOnlyDeathProbability(
    t⁻, 
    t, 
    θ, 
    Xt⁻::Vector{<:Union{Nothing, Real}}, 
    Xt::Vector{<:Union{Nothing, Real}},
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


# likelihood functions

function TumorNegLogLikelihood(
    θ,
    data;   
    S0::Real=0.065
    )::Real

    beta0, rho, delta = θ

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Nt, Dt, Tt, BMIt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis,
            patient_data.death,
            patient_data.treatment,
            patient_data.BMIt
        ]

        if isnothing(findfirst(Tt .== 1))
            treatment_time = nothing
        else
            treatment_time = timepoints[Tt .== 1][1]
        end

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)
            t = timepoints[i]

            St = TumorGrowth(t, S0, beta0, rho, delta, treatment_time=treatment_time, BMI=BMIt[i])

            # Check for the treament and BMI
            if Tt[i] == 1
                if BMIt[i] >= 30
                    beta = beta0 - rho + delta
                else
                    beta = beta0 - rho
                end
            else
                beta = beta0
            end

            Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
            Yt = [Bt[i], Nt[i], Dt[i], Tt[i], BMIt[i]]


            # likelihood is basically just the observation probability here.
            l += log(ObservationProbability(Xt, Yt))
        end
        ll += l
    end
    return -ll
end

function MetastasisNegLogLikelihood(
    θ, 
    data; 
    S0::Real=0.05
)
    # Unpack parameters
    beta0, rho, delta, m_basal, m_size, = θ

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Nt, Dt, Tt, BMIt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis,
            patient_data.death,
            patient_data.treatment,
            patient_data.BMIt
        ]

        # get treatment_time
        if isnothing(findfirst(Tt .== 1))
            treatment_time = nothing
        else
            treatment_time = timepoints[Tt .== 1][1]
        end

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)
            t = timepoints[i]

            # Check for the treament and BMI
            if Tt[i] == 1
                if BMIt[i] >= 30
                    beta = beta0 - rho + delta
                else
                    beta = beta0 - rho
                end
            else
                beta = beta0
            end
            
            # get X based on parameters
            St = TumorGrowth(t, S0, beta0, rho, delta; BMI=BMIt[i], treatment_time=treatment_time)
            
            if (i == 1)
                t⁻ = 0.0
                Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
                Yt = [Bt[i], Nt[i], Dt[i], Tt[i], BMIt[i]]
                Xt⁻ = [S0, 0, 0, Tt[i], BMIt[i]]
            else
                # Set t, t⁻1
                t⁻ = timepoints[i-1]
                # Check for the treament and BMI
                if Tt[i] == 1
                    if BMIt[i] >= 30
                        beta⁻ = beta0 - rho + delta
                    else
                        beta⁻ = beta0 - rho
                    end
                else
                    beta⁻ = beta0
                end
                # get X based on parameters
                St⁻ = TumorGrowth(t, S0, beta0, rho, delta; BMI=BMIt[i], treatment_time=treatment_time)

                # Set Xt, Yt, Xt⁻1
                Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]]
                Yt = [Bt[i], Nt[i], Dt[i], Tt[i], BMIt[i]]
                Xt⁻ = [St⁻, Nt[i-1], Dt[i-1], Tt[i-1], BMIt[i-1]]
            end

            # likelihood is basically just the observation probability here.
            l += log(OnlyMetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0))
        end
        ll += l
    end
    return -ll
end

function DeathNegLogLikelihood(
    θ, 
    data; 
    S0::Real=0.05
)
    # Unpack parameters
    beta0, rho, delta, _ = θ

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Nt, Dt, Tt, BMIt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis,
            patient_data.death,
            patient_data.treatment,
            patient_data.BMIt
        ]

        # get treatment_time
        if isnothing(findfirst(Tt .== 1))
            treatment_time = nothing
        else
            treatment_time = timepoints[Tt .== 1][1]
        end

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)
            t = timepoints[i]

            # Check for the treament and BMI
            if Tt[i] == 1
                if BMIt[i] >= 30
                    beta = beta0 - rho + delta
                else
                    beta = beta0 - rho
                end
            else
                beta = beta0
            end

            # get X based on parameters
            St = TumorGrowth(t, S0, beta0, rho, delta; BMI=BMIt[i], treatment_time=treatment_time)

    
            if (i == 1)
                t⁻ = 0.0
                Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
                Yt = [Bt[i], Nt[i], Dt[i], Tt[i], BMIt[i]]
                Xt⁻ = [S0, 0, 0, Tt[i], BMIt[i]]
            else
                # Set t, t⁻1
                t⁻ = timepoints[i-1]
                # Check for the treament and BMI
                if Tt[i] == 1
                    if BMIt[i] >= 30
                        beta⁻ = beta0 - rho + delta
                    else
                        beta⁻ = beta0 - rho
                    end
                else
                    beta⁻ = beta0
                end
                # get X based on parameters
                St⁻ = TumorGrowth(t, S0, beta0, rho, delta; BMI=BMIt[i], treatment_time=treatment_time)
    
                # Set Xt, Yt, Xt⁻1
                Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]]
                Yt = [Bt, Nt[i], Dt[i]]
                Xt⁻ = [St⁻, Nt[i-1], Dt[i-1], Tt[i-1], BMIt[i-1]]
            end

            # likelihood is basically just the observation probability here.
            l += log(OnlyDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
        end
        ll += l
    end
    return -ll
end

function NumericDeathNegLogLikelihood(
    θ, 
    data; 
    S0::Real=0.05
)
    # Unpack parameters
    beta0, rho, delta, _ = θ

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Nt, Dt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.metastasis,
            patient_data.death,
            patient_data.treatment,
            patient_data.BMIt
        ]

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)
            t = timepoints[i]

            # Check for the treament and BMI
            if Tt[i] == 1
                if BMIt[i] >= 30
                    beta = beta0 - rho + delta
                else
                    beta = beta0 - rho
                end
            else
                beta = beta0
            end
    
            St = S0*exp(beta*t)
    
            if (i == 1)
                t⁻ = 0.0
                Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
                Yt = [Bt[i], Nt[i], Dt[i], Tt[i], BMIt[i]]
                Xt⁻ = [S0, 0, 0]
            else
                # Set t⁻1
                t⁻ = timepoints[i-1]
    
                # Set Xt, Yt, Xt⁻1
                Xt = [St, Nt[i], Dt[i], Tt[i], BMIt[i]]
                Yt = [Bt, Nt[i], Dt[i]]
                Xt⁻ = [St[i-1], Nt[i-1], Dt[i-1], Tt[i-1], BMIt[i-1]]
            end

            # likelihood is basically just the observation probability here.
            l += log(NumericOnlyDeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
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
    betaOptim(x, p) = TumorNegLogLikelihood(x[1:3], data, S0=S0)
    betaOptimFunc = OptimizationFunction(betaOptim, Optimization.AutoForwardDiff())
    betaOptimProblem = OptimizationProblem(betaOptimFunc, [x0[1], x0[2], x0[3]], lb=zeros(3).+1e-4, ub=ones(3))
    betaEst = solve(betaOptimProblem, LBFGS(), maxiters=10^5)
    println("Growth parameters:", betaEst, "\n")

    # next we want to optimize the metastasis parameters
    metOptim(x, p) = MetastasisNegLogLikelihood([betaEst..., x[1], x[2]], data, S0=S0)
    metOptimFunc = OptimizationFunction(metOptim, Optimization.AutoForwardDiff())
    metOptimProblem = OptimizationProblem(metOptimFunc, x0[4:5], lb=zeros(2).+1e-4, ub=ones(2))
    metEst = solve(metOptimProblem, LBFGS(), maxiters=10^5)
    println("Metastasis:", metEst, "\n")

    # next we want to optimize the death parameters
    deathOptim(x, p) = DeathNegLogLikelihood([betaEst..., metEst..., x[1], x[2]], data, S0=S0)
    deathOptimFunc = OptimizationFunction(deathOptim, Optimization.AutoForwardDiff())
    deathOptimProblem = OptimizationProblem(deathOptimFunc, x0[6:7], lb=zeros(2).+1e-4, ub=ones(2))
    deathEst = solve(deathOptimProblem, LBFGS(), maxiters=10^5)
    est_par = [betaEst; metEst; deathEst]

    # print solutions
    #println("True parameter is \n", true_par)
    println("Estimated parameter is \n", est_par)

    #return true_par, est_par
    return est_par
end

function LogHierarchOptimization(data, x0; S0=0.065, lb=zeros(7).+1e-4, ub=ones(7), optimizer="LBFGS", llh_type="Analytic")

    # first we want to estimate only beta
    betaOptim(x, p) = TumorNegLogLikelihood(exp.(x[1:3]), data, S0=S0)
    betaOptimFunc = OptimizationFunction(betaOptim, Optimization.AutoForwardDiff())
    betaOptimProblem = OptimizationProblem(betaOptimFunc, x0[1:3], lb=lb[1:3], ub=ub[1:3])
    betaEst = solve(betaOptimProblem, LBFGS(), maxiters=10^5)
    println("Beta:", betaEst, "\n")

    # next we want to optimize the metastasis parameters
    metOptim(x, p) = MetastasisNegLogLikelihood(exp.([betaEst..., x[1], x[2]]), data, S0=S0)
    metOptimFunc = OptimizationFunction(metOptim, Optimization.AutoForwardDiff())
    metOptimProblem = OptimizationProblem(metOptimFunc, x0[4:5], lb=lb[4:5], ub=ub[4:5])
    metEst = solve(metOptimProblem, LBFGS(), maxiters=10^5)
    println("Metastasis:", metEst, "\n")

    # next we want to optimize the death parameters
    if llh_type == "Analytic"
        deathOptim(x, p) = DeathNegLogLikelihood(exp.([betaEst..., metEst..., x[1], x[2]]), data, S0=S0)
        deathOptimFunc = OptimizationFunction(deathOptim, Optimization.AutoForwardDiff())
        deathOptimProblem = OptimizationProblem(deathOptimFunc, x0[6:7], lb=lb[6:7], ub=ub[6:7])
        if optimizer =="LBFGS"
            try
                global deathEst = solve(deathOptimProblem, LBFGS(), maxiters=10^8)
            catch
                global deathEst = solve(deathOptimProblem, SAMIN(), maxiters=10^8)
            end
            est_par = [betaEst; metEst; deathEst]
        elseif optimizer == "SAMIN"
            deathEst = solve(deathOptimProblem, SAMIN(), maxiters=10^8)
            est_par = [betaEst; metEst; deathEst]
        end

        # print solutions
        println("Estimated parameter is \n", est_par)

        #return  est_par
        return est_par
    elseif llh_type == "Numeric"
        NumericdeathOptim(x, p) = NumericDeathNegLogLikelihood(exp.([betaEst..., metEst..., x[1], x[2]]), data, S0=S0)
        NumericdeathOptimFunc = OptimizationFunction(NumericdeathOptim, Optimization.AutoForwardDiff())
        NumericdeathOptimProblem = OptimizationProblem(NumericdeathOptimFunc, x0[4:5], lb=lb[4:5], ub=ub[4:5])
        if optimizer =="LBFGS"
            error("Numeric likelihoods can only work with non-gradient based optimizers. Please, use SAMIN.")
        elseif optimizer == "SAMIN"
            NumericdeathEst = solve(NumericdeathOptimProblem, SAMIN(), maxiters=10^8)
            est_par = [betaEst; metEst; NumericdeathEst]
        end

        println("Estimated parameter is \n", est_par)

        #return est_par
        return est_par
    end
end

function LogOptimization(data, x0; S0=0.065, lb=zeros(7).+1e-4, ub=ones(7), optimizer="LBFGS", llh_type="Analytic")

    obj_values = []
    times = []
    function callback(p, l)
        push!(obj_values, l)
        push!(times, time_ns())
        return false
    end

    # next we want to optimize the death parameters
    if llh_type == "Analytic"
        Optim(x, p) = NegLogLikelihood(exp.(x), data, S0=S0)
        OptimFunc = OptimizationFunction(Optim, Optimization.AutoForwardDiff())
        OptimProblem = OptimizationProblem(OptimFunc, x0, lb=lb, ub=ub)
        if optimizer == "LBFGS"
            joined_est = solve(OptimProblem, LBFGS(), maxiters=10^8, callback=callback)
            est_par = joined_est.u
        elseif optimizer == "SAMIN"
            joined_est = solve(OptimProblem, SAMIN(), maxiters=10^8, callback=callback)
            est_par = joined_est.u
        end

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
    elseif llh_type == "Numeric"
        NumericOptim(x, p) = NumericNegLogLikelihood(exp.(x), data, S0=S0)
        NumericOptimFunc = OptimizationFunction(NumericOptim, Optimization.AutoForwardDiff())
        NumericOptimProblem = OptimizationProblem(NumericOptimFunc, x0, lb=lb, ub=ub)
        if optimizer =="LBFGS"
            error("Numeric likelihoods can only work with non-gradient based optimizers. Please, use SAMIN.")
        elseif optimizer == "SAMIN"
            joined_est = solve(NumericOptimProblem, SAMIN(), maxiters=10^8, callback=callback)
            est_par = joined_est.u
        end

        println("Estimated parameter is \n", est_par)
        nllh = NumericNegLogLikelihood(exp.(joined_est.u), data, S0=S0)
        times_sec = (Int.(times) .- minimum(Int.(times)))/1e9
        res_dict = Dict(
                "nllh" => nllh,
                "parameter" => joined_est.u,
                "obj_val_trace" => obj_values,
                "time_trace" => times_sec
            )
        return res_dict
    end
end

function OnlyTumorObservationProbability(
    Xt::Vector{<:Union{Nothing, Real}},
    Yt; 
    sigma::Real=0.5 # For now with fixed noise level (can be changed to estimated noise later)
    )::Real

    # unpack data
    St, _ = Xt #Size, Metas, Death, Treatment, BMI
    Bt = Yt[1]

    # Compute the likelihood of the tumor size observation
    obs_prob = pdf(Normal(St, sigma*St), Bt)

    return obs_prob
end

# only tumor data optimization
function OnlyTumorNegLogLikelihood(
    θ,
    data;   
    S0::Real=0.065
    )::Real

    beta0, rho, delta = θ

    n_patients = data.patient_id[end]
    ll = 0.0
    for p in 1:n_patients
        patient_data = data[data.patient_id .== p, :]
        # Unpack data
        timepoints, Bt, Tt, BMIt = [patient_data.time, 
            patient_data.tumor, 
            patient_data.treatment,
            patient_data.BMIt
        ]

        if isnothing(findfirst(Tt .== 1))
            treatment_time = nothing
        else
            treatment_time = timepoints[Tt .== 1][1]
        end

        # Initialize loglikelihood
        l = 0.0

        # Loop over data points
        for i in eachindex(timepoints)
            t = timepoints[i]

            St = TumorGrowth(t, S0, beta0, rho, delta, treatment_time=treatment_time, BMI=BMIt[i])

            Xt = [St, Tt[i], BMIt[i]] #(this creates Vector{<:Real} so we need Int for the factorial function later)
            Yt = [Bt[i], Tt[i], BMIt[i]]


            # likelihood is basically just the observation probability here.
            l += log(OnlyTumorObservationProbability(Xt, Yt))
        end
        ll += l
    end
    return -ll
end


end;