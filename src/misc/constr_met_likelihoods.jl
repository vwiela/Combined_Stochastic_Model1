using Random
using DifferentialEquations
using JumpProcesses
using Statistics
using Distributions
using ForwardDiff
using Plots

using Cubature


# set initial value for three-dimensional process
endtime = 30.0
u0 = [0.05, 0, 0]
tspan = (0.0, endtime)

# set parameter values
beta = 0.3
m_basal = 0.05
m_size = 0.1
d_basal = 0.01
d_size = 0.01
d_metastasis = 0.01
dt = 1/2^(8)

p = (beta = beta, m_basal = m_basal, m_size = m_size, d_basal = d_basal, d_size = d_size, d_metastasis = d_metastasis)

# define the drift function
function TumorODE!(du, u, p, t)
    if (u[3] == 0)
        du[1] = p.beta*u[1]
    else
        du[1] = 0
    end
end

# define the jump-rate functions
MetastasisRate(u, p, t) = p.m_basal+p.m_size*sqrt(u[1])
DeathRate(u, p, t) = p.d_basal+p.d_size*sqrt(u[1])+p.d_metastasis*u[2]

# define the jump functions
function MetastasisAffect!(integrator)
    if (integrator.u[3] == 0) # nlimited number of metastasis allowed
        integrator.u[2] += 1
    end
    nothing
end

function DeathAffect!(integrator)
    if (integrator.u[3] == 0) 
        integrator.u[3] += 1
    end
    terminate!(integrator)
    nothing
end

# define the ODE problem
prob = ODEProblem(TumorODE!, u0, tspan, p)

# define the jump problem
metastasis_jump = VariableRateJump(MetastasisRate, MetastasisAffect!)
death_jump =  VariableRateJump(DeathRate, DeathAffect!)
jump_problem = JumpProblem(prob, Direct(), metastasis_jump, death_jump)



# Next we define the analytical likelihood functions

# helper functions
function TumorGrowth(
    t, 
    S0::Real,
    beta::Real
    )
    return S0 * exp(beta * t)
end

function lambdaN(
    t, 
    beta::Real,
    m_basal::Real, 
    m_size::Real, 
    S0::Real, 
    )

    S = TumorGrowth(t, S0, beta)
    return m_basal + m_size * sqrt(S)
end

function lambdaD(
    t, 
    beta::Real,
    d_basal::Real, 
    d_size::Real, 
    d_metas::Real, 
    S0::Real, 
    Nt
    )

    S = TumorGrowth(t, S0, beta)
    return d_basal + d_size * sqrt(S) + d_metas * Nt
end

function Phi(
    t1, 
    t2, 
    beta::Real, 
    d_basal::Real, 
    d_size::Real, 
    d_metas::Real, 
    S0::Real, 
    n
    )

    return exp((2*d_size*(sqrt(exp(beta*t1)*S0) - sqrt(exp(beta*t2)*S0)))/beta + (d_basal + d_metas*n)*(t1 - t2))
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
    d_basal::Real,
    d_size::Real,
    d_metas::Real,
    S0::Real,
    n::Real,
)
    dt = t2-t1

    return dt*(d_basal+n*d_metas)+(2*d_size*sqrt(S0))/beta*(sqrt(exp(beta*t2))-sqrt(exp(beta*t1)))
end

# analytical integrals for death probability

function AnalyticSurvivalProbability1(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    return (beta*exp((2*d_size*(sqrt(exp(beta*t1)) -
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + (d_basal + d_metas*n)*t1 -
    (d_basal + d_metas + d_metas*n)*t2)*(beta*(exp(d_metas*t1) -
    exp(d_metas*t2))*m_basal + 2*d_metas*exp(d_metas*t1)*(m_basal +
    m_size*sqrt(exp(beta*t1)*S0)) - 2*d_metas*exp(d_metas*t2)*(m_basal +
    m_size*sqrt(exp(beta*t2)*S0))))/(d_metas*(beta +
    2*d_metas)*(2*(sqrt(exp(beta*t1)) -
    sqrt(exp(beta*t2)))*m_size*sqrt(S0) + beta*m_basal*(t1 - t2)))
end

function AnalyticSurvivalProbability2(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    return (beta^2*exp((2*d_size*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + (d_basal + d_metas*n)*t1 - 
    (d_basal + d_metas*(2 + n))*t2)*(beta^2*(exp(d_metas*t1) - 
    exp(d_metas*t2))^2*m_basal^2 + 4*beta*d_metas*(exp(d_metas*t1) - 
    exp(d_metas*t2))*m_basal*(exp(d_metas*t1)*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) - exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))) + 4*d_metas^2*(exp((beta + 
    2*d_metas)*t1)*m_size^2*S0 + exp((beta + 2*d_metas)*t2)*m_size^2*S0 + 
    exp(2*d_metas*t1)*m_basal*(m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) - 
    2*exp(d_metas*(t1 + t2))*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + exp(2*d_metas*t2)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)))))/(d_metas^2*(beta + 
    2*d_metas)^2*(2*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0) + beta*m_basal*(t1 - t2))^2)
end

function AnalyticSurvivalProbability3(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    return (exp((2*d_size*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + (d_basal + d_metas*n)*t1 - 
    (d_basal + d_metas*(3 + n))*t2)*(-(beta^3*(exp(d_metas*t1) - 
    exp(d_metas*t2))^3*m_basal^3) - 6*beta^2*d_metas*(exp(d_metas*t1) - 
    exp(d_metas*t2))^2*m_basal^2*(exp(d_metas*t1)*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) - exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))) - 12*beta*d_metas^2*(exp(d_metas*t1) - 
    exp(d_metas*t2))*m_basal*(exp((beta + 2*d_metas)*t1)*m_size^2*S0 + 
    exp((beta + 2*d_metas)*t2)*m_size^2*S0 + 
    exp(2*d_metas*t1)*m_basal*(m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) - 
    2*exp(d_metas*(t1 + t2))*(m_basal + m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + exp(2*d_metas*t2)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0))) - 8*d_metas^3*(3*exp(d_metas*t1 + 
    beta*t2 + 2*d_metas*t2)*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + exp((beta + 
    3*d_metas)*t1)*m_size^2*S0*(3*m_basal + m_size*sqrt(exp(beta*t1)*S0)) +
    exp(3*d_metas*t1)*m_basal^2*(m_basal + 3*m_size*sqrt(exp(beta*t1)*S0)) - 
    3*exp(beta*t1 + 2*d_metas*t1 + d_metas*t2)*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t2)*S0)) - 
    3*exp(d_metas*(2*t1 + t2))*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - exp((beta + 
    3*d_metas)*t2)*m_size^2*S0*(3*m_basal + m_size*sqrt(exp(beta*t2)*S0)) + 
    3*exp(d_metas*(t1 + 2*t2))*m_basal*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 
    exp(3*d_metas*t2)*m_basal^2*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)))))/(d_metas^3*(beta + 
    2*d_metas)^3*((2*(-sqrt(exp(beta*t1)) + 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0))/beta + m_basal*(-t1 + t2))^3)
end

function AnalyticSurvivalProbability4(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    return (beta^4*exp((2*d_size*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + (d_basal + d_metas*n)*t1 - 
    (d_basal + d_metas*(4 + n))*t2)*(beta^4*(exp(d_metas*t1) - 
    exp(d_metas*t2))^4*m_basal^4 + 8*beta^3*d_metas*(exp(d_metas*t1) - 
    exp(d_metas*t2))^3*m_basal^3*(exp(d_metas*t1)*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) - exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))) + 16*d_metas^4*(exp(2*(beta + 
    2*d_metas)*t1)*m_size^4*S0^2 + exp(2*(beta + 
    2*d_metas)*t2)*m_size^4*S0^2 + 6*exp((beta + 2*d_metas)*(t1 + 
    t2))*m_size^4*S0^2 + 6*exp(beta*t2 + 2*d_metas*(t1 + 
    t2))*m_basal*m_size^2*S0*(m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) + 
    2*exp((beta + 4*d_metas)*t1)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0)) + 
    exp(4*d_metas*t1)*m_basal^3*(m_basal + 
    4*m_size*sqrt(exp(beta*t1)*S0)) - 4*exp(beta*t1 + d_metas*(3*t1 + 
    t2))*m_size^2*S0*(3*m_basal + m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 4*exp(d_metas*(3*t1 + 
    t2))*m_basal^2*(m_basal + 3*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 4*exp(beta*t2 + d_metas*(t1 + 
    3*t2))*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0))*(3*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + 6*exp(beta*t1 + 2*d_metas*(t1 + 
    t2))*m_basal*m_size^2*S0*(m_basal + 2*m_size*sqrt(exp(beta*t2)*S0)) + 
    6*exp(2*d_metas*(t1 + t2))*m_basal^2*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) + 2*exp((beta + 
    4*d_metas)*t2)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 4*exp(d_metas*(t1 + 3*t2))*m_basal^2*
    (m_basal + m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 3*m_size*sqrt(exp(beta*t2)*S0)) + 
    exp(4*d_metas*t2)*m_basal^3*(m_basal + 4*m_size*sqrt(exp(beta*t2)*S0))) + 
    24*beta^2*d_metas^2*m_basal^2*(exp((beta + 4*d_metas)*t1)*m_size^2*S0 + 
    exp((beta + 4*d_metas)*t2)*m_size^2*S0 - 2*exp(d_metas*t1 + beta*t2 + 
    3*d_metas*t2)*m_size^2*S0 + exp(beta*t1 + 2*d_metas*(t1 + t2))*m_size^2*S0 + 
    exp(beta*t2 + 2*d_metas*(t1 + t2))*m_size^2*S0 - 2*exp(beta*t1 + 
    d_metas*(3*t1 + t2))*m_size^2*S0 + exp(4*d_metas*t1)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0)) + exp(4*d_metas*t2)*m_basal*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) + exp(2*d_metas*(t1 + t2))*(6*m_basal^2 + 
    4*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    6*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    2*exp(d_metas*(3*t1 + t2))*(2*m_basal^2 + 
    m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    m_basal*m_size*(3*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    2*exp(d_metas*(t1 + 3*t2))*(2*m_basal^2 + 
    m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    m_basal*m_size*(sqrt(exp(beta*t1)*S0) + 3*sqrt(exp(beta*t2)*S0)))) + 
    32*beta*d_metas^3*m_basal*(3*exp(beta*t2 + 2*d_metas*(t1 + 
    t2))*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t1)*S0)) + exp((beta + 
    4*d_metas)*t1)*m_size^2*S0*(3*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + exp(4*d_metas*t1)*m_basal^2*(m_basal + 
    3*m_size*sqrt(exp(beta*t1)*S0)) + 3*exp(beta*t1 + 2*d_metas*(t1 + 
    t2))*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t2)*S0)) + exp((beta + 
    4*d_metas)*t2)*m_size^2*S0*(3*m_basal + m_size*sqrt(exp(beta*t2)*S0)) + 
    exp(4*d_metas*t2)*m_basal^2*(m_basal + 3*m_size*sqrt(exp(beta*t2)*S0)) +
    3*exp(2*d_metas*(t1 + t2))*m_basal*(2*m_basal^2 + 
    4*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    3*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    exp(beta*t2 + d_metas*(t1 + 3*t2))*m_size^2*S0*(6*m_basal + 
    m_size*(3*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    exp(d_metas*(3*t1 + t2))*m_basal*(4*m_basal^2 + 
    6*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    3*m_basal*m_size*(3*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    exp(beta*t1 + d_metas*(3*t1 + t2))*m_size^2*S0*(6*m_basal + 
    m_size*(sqrt(exp(beta*t1)*S0) + 3*sqrt(exp(beta*t2)*S0))) - 
    exp(d_metas*(t1 + 3*t2))*m_basal*(4*m_basal^2 + 
    6*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    3*m_basal*m_size*(sqrt(exp(beta*t1)*S0) + 
    3*sqrt(exp(beta*t2)*S0))))))/(d_metas^4*(beta + 
    2*d_metas)^4*(2*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0) + beta*m_basal*(t1 - t2))^4)

end

function AnalyticSurvivalProbability5(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    return -((beta^5*exp((2*d_size*(sqrt(exp(beta*t1)) -
    sqrt(exp(beta*t2)))*sqrt(S0))/beta + d_basal*(t1 - t2) + 
    d_metas*n*(t1 - t2) - 5*d_metas*t2)*(-(beta^5*(exp(d_metas*t1) - 
    exp(d_metas*t2))^5*m_basal^5) - 10*beta^4*d_metas*(exp(d_metas*t1) - 
    exp(d_metas*t2))^4*m_basal^4*(exp(d_metas*t1)*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) - exp(d_metas*t2)*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0))) - 32*d_metas^5*(5*exp(2*beta*t1 + 
    5*d_metas*t1)*m_basal*m_size^4*S0^2 - 5*exp(2*beta*t2 + 
    5*d_metas*t2)*m_basal*m_size^4*S0^2 + exp((beta + 
    5*d_metas)*t1)*m_size^5*S0*(exp(beta*t1)*S0)^1.5 - exp((beta + 
    5*d_metas)*t2)*m_size^5*S0*(exp(beta*t2)*S0)^1.5 + 10*exp((beta + 
    5*d_metas)*t1)*m_basal^2*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 30*exp(d_metas*t1 + beta*t2 + 
    4*d_metas*t2)*m_basal^2*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 5*exp(d_metas*t1 + 2*beta*t2 + 
    4*d_metas*t2)*m_size^4*S0^2*(m_basal + m_size*sqrt(exp(beta*t1)*S0))+ 
    20*exp(d_metas*(t1 + 4*t2))*m_basal*m_size^3*(exp(beta*t2)*S0)^1.5*(m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 10*exp(beta*t1 + 3*d_metas*t1 + 
    beta*t2 + 2*d_metas*t2)*m_size^4*S0^2*(3*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 10*exp(3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_basal^2*m_size^2*S0*(m_basal + 3*m_size*sqrt(exp(beta*t1)*S0)) + 
    exp(5*d_metas*t1)*m_basal^4*(m_basal + 5*m_size*sqrt(exp(beta*t1)*S0)) - 10*exp((beta + 
    5*d_metas)*t2)*m_basal^2*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 30*exp(beta*t1 + 4*d_metas*t1 + 
    d_metas*t2)*m_basal^2*m_size^2*S0*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 5*exp(2*beta*t1 + 4*d_metas*t1 + 
    d_metas*t2)*m_size^4*S0^2*(m_basal + m_size*sqrt(exp(beta*t2)*S0)) - 
    20*exp(d_metas*(4*t1 + t2))*m_basal*m_size^3*(exp(beta*t1)*S0)^1.5*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 5*exp(d_metas*(4*t1 + 
    t2))*m_basal^3*(m_basal + 4*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 10*exp(beta*t1 + 2*d_metas*t1 + 
    beta*t2 + 3*d_metas*t2)*m_size^4*S0^2*(3*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 10*exp(2*d_metas*t1 + beta*t2 + 
    3*d_metas*t2)*m_basal*m_size^2*S0*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0))*(3*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) + 10*exp(beta*t1 + 3*d_metas*t1 + 
    2*d_metas*t2)*m_basal*m_size^2*S0*(3*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) + 10*exp(3*d_metas*t1 + 
    2*d_metas*t2)*m_basal^3*(m_basal + 
    3*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 10*exp(beta*t1 + 2*d_metas*t1 + 
    3*d_metas*t2)*m_basal^2*m_size^2*S0*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)) - 10*exp(2*d_metas*t1 + 
    3*d_metas*t2)*m_basal^3*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)) + 5*exp(d_metas*(t1 + 
    4*t2))*m_basal^3*(m_basal + m_size*sqrt(exp(beta*t1)*S0))*(m_basal + 
    4*m_size*sqrt(exp(beta*t2)*S0)) - 
    exp(5*d_metas*t2)*m_basal^4*(m_basal + 
    5*m_size*sqrt(exp(beta*t2)*S0))) - 
    40*beta^3*d_metas^2*m_basal^3*(exp((beta + 5*d_metas)*t1)*m_size^2*S0 - 
    exp((beta + 5*d_metas)*t2)*m_size^2*S0 - 3*exp(beta*t1 + 
    4*d_metas*t1 + d_metas*t2)*m_size^2*S0 + 3*exp(beta*t1 + 3*d_metas*t1 
    + 2*d_metas*t2)*m_size^2*S0 + exp(3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_size^2*S0 - exp(beta*t1 + 2*d_metas*t1 + 
    3*d_metas*t2)*m_size^2*S0 - 3*exp(2*d_metas*t1 + beta*t2 + 
    3*d_metas*t2)*m_size^2*S0 + 3*exp(d_metas*t1 + beta*t2 + 
    4*d_metas*t2)*m_size^2*S0 + exp(5*d_metas*t1)*m_basal*(m_basal + 
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
    80*beta^2*d_metas^3*m_basal^2*(2*exp(d_metas*(t1 + 
    4*t2))*m_size^3*(exp(beta*t2)*S0)^1.5 + 3*exp(3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t1)*S0)) + 
    exp((beta + 5*d_metas)*t1)*m_size^2*S0*(3*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 3*exp(d_metas*t1 + beta*t2 + 
    4*d_metas*t2)*m_size^2*S0*(3*m_basal + m_size*sqrt(exp(beta*t1)*S0)) + 
    exp(5*d_metas*t1)*m_basal^2*(m_basal + 
    3*m_size*sqrt(exp(beta*t1)*S0)) - 3*exp(beta*t1 + 2*d_metas*t1 + 
    3*d_metas*t2)*m_size^2*S0*(m_basal + m_size*sqrt(exp(beta*t2)*S0)) - 
    exp((beta + 5*d_metas)*t2)*m_size^2*S0*(3*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - exp(5*d_metas*t2)*m_basal^2*(m_basal + 
    3*m_size*sqrt(exp(beta*t2)*S0)) - exp(beta*t1 + 4*d_metas*t1 + 
    d_metas*t2)*m_size^2*S0*(9*m_basal + 2*m_size*sqrt(exp(beta*t1)*S0) + 
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
    80*beta*d_metas^4*m_basal*(exp(2*beta*t1 + 
    5*d_metas*t1)*m_size^4*S0^2 - exp(2*beta*t1 + 4*d_metas*t1 + 
    d_metas*t2)*m_size^4*S0^2 + 6*exp(beta*t1 + 3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_size^4*S0^2 - 6*exp(beta*t1 + 2*d_metas*t1 + beta*t2 + 
    3*d_metas*t2)*m_size^4*S0^2 + exp(d_metas*t1 + 2*beta*t2 + 
    4*d_metas*t2)*m_size^4*S0^2 - exp(2*beta*t2 + 
    5*d_metas*t2)*m_size^4*S0^2 + 4*exp(d_metas*(t1 + 
    4*t2))*m_size^3*(exp(beta*t2)*S0)^1.5*(2*m_basal + 
    m_size*sqrt(exp(beta*t1)*S0)) + 6*exp(3*d_metas*t1 + beta*t2 + 
    2*d_metas*t2)*m_basal*m_size^2*S0*(m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0)) + 2*exp((beta + 
    5*d_metas)*t1)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t1)*S0)) + 6*exp(d_metas*t1 + beta*t2 + 
    4*d_metas*t2)*m_basal*m_size^2*S0*(3*m_basal + 2*m_size*sqrt(exp(beta*t1)*S0)) + 
    exp(5*d_metas*t1)*m_basal^3*(m_basal + 
    4*m_size*sqrt(exp(beta*t1)*S0)) - 4*exp(d_metas*(4*t1 + 
    t2))*m_size^3*(exp(beta*t1)*S0)^1.5*(2*m_basal + 
    m_size*sqrt(exp(beta*t2)*S0)) - 6*exp(beta*t1 + 2*d_metas*t1 + 
    3*d_metas*t2)*m_basal*m_size^2*S0*(m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 2*exp((beta + 
    5*d_metas)*t2)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - 6*exp(beta*t1 + 4*d_metas*t1 + 
    d_metas*t2)*m_basal*m_size^2*S0*(3*m_basal + 
    2*m_size*sqrt(exp(beta*t2)*S0)) - exp(5*d_metas*t2)*m_basal^3*(m_basal + 
    4*m_size*sqrt(exp(beta*t2)*S0)) - exp(d_metas*(4*t1 + t2))*m_basal^2*(5*m_basal^2 + 
    12*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
    4*m_basal*m_size*(4*sqrt(exp(beta*t1)*S0) + sqrt(exp(beta*t2)*S0))) - 
    2*exp(2*d_metas*t1 + beta*t2 + 3*d_metas*t2)*m_size^2*S0*(9*m_basal^2 + 
    2*m_size^2*sqrt(exp(beta*t1)*S0)*sqrt(exp(beta*t2)*S0) + 
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
    6*sqrt(exp(beta*t2)*S0))))))/(d_metas^5*(beta + 2*d_metas)^5*(2*(sqrt(exp(beta*t1)) - 
    sqrt(exp(beta*t2)))*m_size*sqrt(S0) + beta*m_basal*(t1 - t2))^5))    
end

function NumericSurvivalProbability1(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    integral = (1/(LambdaN(t1, t2, beta, m_basal, m_size, S0))) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_basal, d_size, d_metas, S0, n)*
        Phi(u1, t2, beta, d_basal, d_size, d_metas, S0, n+1),
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability2(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    integral = (2/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^2)) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_basal, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_basal, d_size, d_metas, S0, n+1) *
            Phi(u2, t2,  beta, d_basal, d_size, d_metas, S0, n+2),
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability3(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    integral = (6/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^3)) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_basal, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_basal, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta, m_basal, m_size, S0) *
                Phi(u2, u3, beta, d_basal, d_size, d_metas, S0, n+2) *
                Phi(u3, t2, beta, d_basal, d_size, d_metas, S0, n+3),
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end


function NumericSurvivalProbability4(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    integral = (24/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^4)) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_basal, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_basal, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta, m_basal, m_size, S0) *
                Phi(u2, u3, beta, d_basal, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta, m_basal, m_size, S0) *
                            Phi(u3, u4, beta, d_basal, d_size, d_metas, S0, n+3) *
                            Phi(u4, t2, beta, d_basal, d_size, d_metas, S0, n+4),
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

function NumericSurvivalProbability5(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    integral = 120/((LambdaN(t1, t2, beta, m_basal, m_size, S0))^5) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_basal, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_basal, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta, m_basal, m_size, S0) *
                Phi(u2, u3, beta, d_basal, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta, m_basal, m_size, S0) *
                    Phi(u3, u4, beta, d_basal, d_size, d_metas, S0, n+3) *
                    hquadrature(u5 -> lambdaN(u5, beta, m_basal, m_size, S0) *
                        Phi(u4, u5, beta, d_basal, d_size, d_metas, S0, n+4) *
                        Phi(u5, t2, beta, d_basal, d_size, d_metas, S0, n+5),
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

function NumericSurvivalProbability6(t1, t2, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, n)
    integral = 720/((LambdaN(t1, t2, beta, m_basal, m_size, S0))^6) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_basal, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_basal, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta, m_basal, m_size, S0) *
                Phi(u2, u3, beta, d_basal, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta, m_basal, m_size, S0) *
                    Phi(u3, u4, beta, d_basal, d_size, d_metas, S0, n+3) *
                    hquadrature(u5 -> lambdaN(u5, beta, m_basal, m_size, S0) *
                        Phi(u4, u5, beta, d_basal, d_size, d_metas, S0, n+4) *
                        hquadrature(u6 -> lambdaN(u6, beta, m_basal, m_size, S0) *
                            Phi(u5, u6, beta, d_basal, d_size, d_metas, S0, n+5) *
                            Phi(u6, t2, beta, d_basal, d_size, d_metas, S0, n+6),
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

function SurvivalProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    # Unpack parameters
    beta, m_basal, m_size, d_basal, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        surv_prob = Phi(t⁻, t, beta, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        surv_prob = AnalyticSurvivalProbability1(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        surv_prob = AnalyticSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        surv_prob = AnalyticSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        surv_prob = AnalyticSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        surv_prob = AnalyticSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        surv_prob = NumericSurvivalProbability6(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    else
        println("more than 6 metastasis in one interval, $(Nt-Nt⁻)")
        surv_prob = 1.0
    end
    # did not observe more than 6 jumps in one time interval.

  
    if (surv_prob < 0.0)
        println("surv_prob is negative for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_basal, d_size, d_metas])
        surv_prob=1e-50
    end
    # if (surv_prob > 1.0)
    #     println("surv_prob is bigger 1 for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_basal, d_size, d_metas])
    # end

    return surv_prob
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
    beta, m_basal, m_size, d_basal, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        surv_prob = Phi(t⁻, t, beta, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        surv_prob = NumericSurvivalProbability1(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        surv_prob = NumericSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        surv_prob = NumericSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        surv_prob = NumericSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        surv_prob = NumericSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        surv_prob = NumericSurvivalProbability6(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    else
        println("more than 6 metastasis in one interval, $(Nt-Nt⁻)")
        surv_prob = 1.0
    end
    # did not observe more than 6 jumps in one time interval.

  
    if (surv_prob < 0.0)
        println("surv_prob is negative for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_basal, d_size, d_metas])
        surv_prob=1e-50
    end
    if (surv_prob > 1.0)
        println("surv_prob is bigger 1 for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_basal, d_size, d_metas])
    end

    return surv_prob
end

function DeathProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    # Unpack parameters
    beta, m_basal, m_size, d_basal, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*Phi(t⁻, t, beta, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability1(t⁻, t, beta,m_basal, m_size, d_basal, d_size, d_metas, S0,  Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*NumericSurvivalProbability6(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    end

  
    if (death_prob < 0.0)
        println("death_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
        death_prob=1e-50
    end
    if (death_prob > 1.0)
        println("Hey, death_prob is bigger 1 for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
    end

    return death_prob
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
    beta, m_basal, m_size, d_basal, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*Phi(t⁻, t, beta, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*NumericSurvivalProbability1(t⁻, t, beta,m_basal, m_size, d_basal, d_size, d_metas, S0,  Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*NumericSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*NumericSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*NumericSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*NumericSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        death_prob = lambdaD(t, beta, d_basal, d_size, d_metas, S0, Nt)*NumericSurvivalProbability6(t⁻, t, beta, m_basal, m_size, d_basal, d_size, d_metas, S0, Nt⁻)
    end

  
    # if (death_prob < 0.0)
    #     println("death_prob is negative for t⁻ = $t⁻, t = $t, Xt⁻ = $Xt⁻, Xt = $Xt\n It is ",death_prob, "\n", θ)
    #     death_prob=1e-50
    # end
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
    beta, m_basal, m_size, = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    dn = Int(Nt - Nt⁻)
    Lt = LambdaN(t⁻, t, beta, m_basal, m_size, S0)
    met_prob = (Lt^dn)/(factorial(dn))*exp(-Lt)

    return met_prob
end


function OnlyMetastasisProbability(
    t⁻, 
    t, 
    θ::Vector{<:Real}, 
    Xt⁻::Vector{<:Real}, 
    Xt::Vector{<:Real},
    S0::Real
    )::Real

    # Unpack parameters
    beta, m_basal, m_size, = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    dn = Int(Nt - Nt⁻)
    Lt = LambdaN(t⁻, t, beta, m_basal, m_size, S0)
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
        process_prob = (SurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    else # death
        process_prob = (DeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    end
    return process_prob
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
        process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0) * (SurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0))
    else # death
        process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0) * (DeathProbability(t⁻, t, θ, Xt⁻, Xt, S0))
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
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
    )::Real

    # Unpack parameters
    beta, m_basal, m_size, d_basal, d_size, d_metas = θ

    # Unpack data
    timepoints, Bt, Nt, Dt = data

    # get X based on parameters
    St = TumorGrowth.(timepoints, S0, beta)

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


# define the likelihoods for hierarchical optimization

function TumorNegLogLikelihood(
    beta,
    data,   
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
        St = TumorGrowth.(timepoints, S0, beta)

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
    beta, m_basal, m_size = θ

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
        St = TumorGrowth.(timepoints, S0, beta)

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
    beta, m_basal, m_size = θ

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
        St = TumorGrowth.(timepoints, S0, beta)

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

function FullNegLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
    )::Real

    return -OdeLogLikelihood(θ, data, S0=S0)
end

# define likelihoods for just considering survival

function SurvivalTimepointLikelihood(
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

    # we consider just surival until that time and treat everything as censored somehow 
    # so we don't care for the death timepoint and just consider the survival probability

    # get process probability
    process_prob = MetastasisProbability(t⁻, t, θ, Xt⁻, Xt, S0) * (SurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0))

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

function SurvivalPatientLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
    )::Real

    # Unpack parameters
    beta, m_basal, m_size, d_basal, d_size, d_metas = θ

    # Unpack data
    timepoints, Bt, Nt, Dt = data

    # get X based on parameters
    St = TumorGrowth.(timepoints, S0, beta)

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

        l += log(SurvivalTimepointLikelihood(t⁻, t, θ, Xt⁻, Xt, Yt, S0))
    end

    return l
end

function SurvivalOdeLogLikelihood(
    θ::Vector{<:Real},
    data;
    S0::Real=0.05
    )::Real

    n_patients = data.patient_id[end]
    ll = 0.0
    for i in 1:n_patients
        patient_data = data[data.patient_id .== i, :]
        ll += SurvivalPatientLogLikelihood(
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

function SurvivalNegLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
    )::Real

    return -SurvivalOdeLogLikelihood(θ, data, S0=S0)
end

#------------------------------------------------------------------------------------------------------------------------------------------------
## Test functions

# set initial value for three-dimensional process
# endtime = 30.0
# S0 = 0.05
# u0 = [S0, 0, 0]
# tspan = (0.0, endtime)

# set parameter values
beta = 0.3
m_basal = 0.05
m_size = 0.1
d_basal = 0.01
d_size = 0.01
d_metas = 0.01
t1 = 25.0
t2 = 26.0
S0 = 0.05
n = 1 # number of metastasis at t1
true_θ = [beta, m_basal, m_size, d_basal, d_size, d_metas]
true_p = (beta = beta, 
          m_basal = m_basal, 
          m_size = m_size, 
          d_basal = d_basal, 
          d_size = d_size, 
          d_metas = d_metas,
          t1 = t1,
          t2 = t2,
          n = n,
          S0 = S0,
          )

# # set example parameter values
ex_beta = 0.3003129605733999
ex_m_basal = 0.05153573335115184
ex_m_size = 0.05994777733109157
ex_d_basal = 0.00044956392348983326
ex_d_size = 0.00022083360110801718
ex_d_metastasis = 1.9473690940604096e-6
ex_θ = [ex_beta, ex_m_basal, ex_m_size, ex_d_basal, ex_d_size, ex_d_metastasis]
S0Example = 0.05
t1Example = 29.0
t2Example = 30.0
nExample = 4



ex_p = (beta = ex_beta, 
        m_basal = ex_m_basal, 
        m_size = ex_m_size, 
        d_basal = ex_d_basal, 
        d_size = ex_d_size,
        d_metastasis = ex_d_metastasis,
        t1 = t1Example,
        t2 = t2Example,
        n = nExample,
        S0 = S0Example
        )

SurvivalProbability(t1Example ,t2Example, ex_θ, [0.0,nExample,0.0], [0.0,nExample+1,0.0], S0Example)

SurvivalProbability(t1Example, t2Example, ex_θ, [0.0,nExample,0.0], [0.0,nExample+2,0.0], S0Example)

SurvivalProbability(t1Example, t2Example, ex_θ, [0.0,nExample,0.0], [0.0,nExample+3,0.0], S0Example)

SurvivalProbability(t1Example, t2Example, ex_θ, [0.0,nExample,0.0], [0.0,nExample+4,0.0], S0Example)

SurvivalProbability(t1Example, t2Example, ex_θ, [0.0,nExample,0.0], [0.0,nExample+5,0.0], S0Example)

SurvivalProbability(t1Example, t2Example, ex_θ, [0.0,nExample,0.0], [0.0,nExample+6,0.0], S0Example)

AnalyticSurvivalProbability4(t1Example, t2Example, ex_beta, ex_m_basal, ex_m_size, ex_d_basal, ex_d_size, ex_d_metastasis, S0Example, nExample)

AnalyticSurvivalProbability5(t1Example, t2Example, ex_beta, ex_m_basal, ex_m_size, ex_d_basal, ex_d_size, ex_d_metastasis, S0Example, nExample)

NumericSurvivalProbability4(t1Example, t2Example, ex_beta, ex_m_basal, ex_m_size, ex_d_basal, ex_d_size, ex_d_metastasis, S0Example, nExample)
    
NumericSurvivalProbability5(t1Example, t2Example, ex_beta, ex_m_basal, ex_m_size, ex_d_basal, ex_d_size, ex_d_metastasis, S0Example, nExample)

    
# lambdaD(25, beta, d_basal, d_size, d_metas, S0, 5)

# LambdaD(25, 26, beta, d_basal, d_size, d_metas, S0, 5)

t⁻ = 23.0
t = 24.0
Nt⁻ = 2.0
Nt = 6.0

par = [0.30031298234265574, 0.05153362562313967, 0.05994831657254766, 0.00044747028141182716, 0.0002209105965031559, 9.998736343226622e-10]

AnalyticSurvivalProbability3(t⁻, t, par[1], par[2], par[3], par[4], par[5], par[6], S0Example, Nt⁻)

NumericSurvivalProbability3(t⁻, t, par[1], par[2], par[3], par[4], par[5], par[6], S0Example, Nt⁻)

AnalyticSurvivalProbability4(t⁻, t, par[1], par[2], par[3], par[4], par[5], par[6], S0Example, Nt⁻)

NumericSurvivalProbability4(t⁻, t, par[1], par[2], par[3], par[4], par[5], par[6], S0Example, Nt⁻)

AnalyticSurvivalProbability5(t⁻, t, par[1], par[2], par[3], par[4], par[5], par[6], S0Example, Nt⁻)
    
NumericSurvivalProbability5(t⁻, t, par[1], par[2], par[3], par[4], par[5], par[6], S0Example, Nt⁻)


## just with numeric integrals
# likelihood functions

function NumericTimepointLikelihood(
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

function NumericPatientLogLikelihood(
    θ::Vector{<:Real}, 
    data; 
    S0::Real=0.05
    )::Real

    # Unpack parameters
    beta, m_basal, m_size, d_basal, d_size, d_metas = θ

    # Unpack data
    timepoints, Bt, Nt, Dt = data

    # get X based on parameters
    St = TumorGrowth.(timepoints, S0, beta)

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

        l += log(NumericTimepointLikelihood(t⁻, t, θ, Xt⁻, Xt, Yt, S0))
    end

    return l
end

function NumericOdeLogLikelihood(
    θ::Vector{<:Real},
    data;
    S0::Real=0.05
    )::Real

    n_patients = data.patient_id[end]
    ll = 0.0
    for i in 1:n_patients
        patient_data = data[data.patient_id .== i, :]
        ll += NumericPatientLogLikelihood(
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


#----------------------------------------------------------------------------------------------------------------------------------------------

# Plot Survivalprobability over time for some metastasis trajectory

function MetProcess(
    t
    )
    return maximum([0, round((t-2.5)/5, RoundUp)])
end

deathtime = 28.3

function SurvProbTrajectory(t)
    t⁻ = Int(floor(t))
    if t⁻ == t
        t⁻ = t⁻ -1
    end
    n⁻ = MetProcess(t⁻)
    n = MetProcess(t)

    return SurvivalProbability(t⁻, t, true_θ, [0.0,n⁻,0.0], [0.0,n,0.0], S0Example)
end

plot(SurvProbTrajectory, 1, 30, label="Survival Probability")

bar(SurvProbTrajectory.(1:30), label="Survival Probability")

plot(5.0:0.01:6.0, t -> NumericSurvivalProbability(5.0, t, ex_θ, [0.0,1,0.0], [0.0,2,0.0], S0Example), label="Survival Probability")

plot(25.0:0.01:26.0, t -> SurvivalProbability(25.0, t, ex_θ, [0.0,1,0.0], [0.0,1,0.0], S0Example), label="Survival Probability")

plot(25.0:0.01:26.0, t -> DeathProbability(25.0, t, ex_θ, [0.0,1,0.0], [0.0,1,0.0], S0Example), label="Survival Probability")


plot(5.0:0.01:6.0, t -> Phi(5.0, t, true_θ[1], true_θ[4], true_θ[5], true_θ[6], S0Example, 1), label="Survival Probability")

NumericSurvivalProbability(5.0, 5.2, ex_θ, [0.0,1,0.0], [0.0,2,0.0], S0Example)

SurvProbTrajectory(5)
SurvProbTrajectory(20)
SurvivalProbability(19, 20, true_θ, [0.0,3,0.0], [0.0,3,0.0], S0Example)

# Look at death probabilities

plot(5.0:0.01:6.0, t -> NumericDeathProbability(5.0, t, true_θ, [0.0,1,0.0], [0.0,2,0.0], S0Example), label="Death Probability")

strue = SurvivalProbability(0.0, 28.0, true_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)
dtrue =DeathProbability(28.0, 28.3, true_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)

strue * dtrue

DeathProbability(0.0, 28.3, true_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)

SurvivalProbability(0.0, 28.0, ex_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)

DeathProbability(0.0, 28.3, ex_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)

DeathProbability(28.0, 28.3, ex_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)



strue = SurvivalProbability(0.0, 30.0, true_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)

sex = SurvivalProbability(0.0, 30.0, ex_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)

DeathProbability(0.0, 28.3, ex_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)

DeathProbability(28.0, 28.3, ex_θ, [0.0,0,0.0], [0.0,0,0.0], S0Example)