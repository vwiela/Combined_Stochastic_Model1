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

using Cubature
using FastGaussQuadrature, LinearAlgebra
include("MNR_simulation_algorithm.jl")


# set seed for reproducibility
Random.seed!(123)


# set model parameter values
beta = 0.3
m_basal = 0.04
m_size = 0.04
d_size = 0.01
d_metastasis = 0.01


full_p = (beta = beta, m_basal = m_basal, m_size = m_size, d_size = d_size, d_metastasis = d_metastasis)
gt_par = Dict(
    "beta" => log10(beta),
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

# set seed for reproducibility
Random.seed!(123);

# define the drift function
function TumorODE!(du, u, p, t)
    if (u[3] == 0)
        du[1] = p.beta*u[1]
    else
        du[1] = 0
    end
end

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

# define the ODE problem
prob = ODEProblem(TumorODE!, u0, tspan, full_p)

# define the jump problem
metastasis_jump = VariableRateJump(metastasis_rate, metastasis_affect!)
death_jump =  VariableRateJump(death_rate, death_affect!)
jump_problem = JumpProblem(prob, Direct(), metastasis_jump, death_jump)

# solve and plot the solution
sol = solve(jump_problem, Tsit5(), tstops=0.0:0.1:endtime)


# functions for data creation

function EnsembleToData(
    ensemble_solution::EnsembleSolution;
    timepoints = 0:1:30,
)
    npat = length(ensemble_solution)
    patient_data = DataFrame(patient_id=Int64[], time=Real[], tumor=Real[], metastasis=Real[], death=Real[])

    for i in 1:npat
        single_sol = ensemble_solution[i]
        death_values = Int.([single_sol.u[i][3] for i in 1:length(single_sol.u)])
        death_index = findfirst(death_values .== 1)
        if (death_index !== nothing) 
            new_timepoints = timepoints[findall(timepoints .< single_sol.t[death_index])] # get all observation times before death
            patient_data = vcat(
            patient_data, 
            DataFrame(
                patient_id=i, 
                time=new_timepoints, 
                tumor=rand(MvNormal(hcat(ensemble_solution[i](new_timepoints).u...)[1, :], sigma.*hcat(ensemble_solution[i](new_timepoints).u...)[1, :])), 
                metastasis = Int.(hcat(ensemble_solution[i](new_timepoints).u...)[2, :]),
                death=Int.(hcat(ensemble_solution[i](new_timepoints).u...)[3, :])
            )
            )
            # add death observation withe exact death time
            death_timepoint = [single_sol.t[death_index]]
            patient_data = vcat(
            patient_data, 
            DataFrame(
                patient_id=i, 
                time=death_timepoint, 
                tumor=rand(Normal(single_sol.u[death_index][1], sigma*single_sol.u[death_index][1])), 
                metastasis = Int.(single_sol.u[death_index][2]),
                death=Int.(single_sol.u[death_index][3])
            )
            )
        else
            patient_data = vcat(
                patient_data, 
                DataFrame(
                    patient_id=i, 
                    time=timepoints, 
                    tumor=rand(MvNormal(hcat(ensemble_solution[i](timepoints).u...)[1, :], sigma.*hcat(ensemble_solution[i](timepoints).u...)[1, :])),  
                    metastasis = Int.(hcat(ensemble_solution[i](timepoints).u...)[2, :]),
                    death=Int.(hcat(ensemble_solution[i](timepoints).u...)[3, :])
                )
            )
        end
    end

    return patient_data
end

# give summary statistics of data

function data_summary(patient_data)
    if isempty(patient_data.time[patient_data.death .== 1.0])
        mean_survival_time = 0.0
        low_quantile_survival_time = 0.0
        high_quantile_survival_time = 0.0
    else
        mean_survival_time = mean(patient_data.time[patient_data.death .== 1.0])
        low_quantile_survival_time = quantile(patient_data.time[patient_data.death .== 1.0], 0.25)
        high_quantile_survival_time = quantile(patient_data.time[patient_data.death .== 1.0], 0.75)
    end
    patients_died = length(unique(patient_data.patient_id[patient_data.death .== 1.0]))
    mean_metastasis_at_death = mean(patient_data.metastasis[patient_data.death .== 1.0])
    mean_metastasis_10_days = mean(patient_data.metastasis[patient_data.time .== 10.0])
    mean_metastasis_20_days = mean(patient_data.metastasis[patient_data.time .== 20.0])
    maximal_metastasis = maximum(patient_data.metastasis)
    maximal_new_metastasis = maximum(diff(patient_data.metastasis))
    tumor_size_10_days = mean(patient_data.tumor[patient_data.time .== 10.0])

    summary_dict = Dict(
        "Mean survival time:" => mean_survival_time,
        "Low Quantile survival time:" => low_quantile_survival_time,
        "High Quantile survival time:" => high_quantile_survival_time,
        "Patients died:" => patients_died,
        "Maximal metastasis number:" => maximal_metastasis,
        "Maximum metastasis increase:" => maximal_new_metastasis,
        "Mean metastasis at death:" => mean_metastasis_at_death,
        "Mean metastasis at day 10:" => mean_metastasis_10_days,
        "Mean metastasis at day 20:" => mean_metastasis_20_days,
        "Mean tumor size at day 10:" => tumor_size_10_days,
    )
    return summary_dict
end


function second_model_data_simulator(
    par;
    npat = 200,
    log_par = nothing,
)
    """
        par - Dictionary of parameters on log 10 scale
    """
    endtime = 30.0
    
    if isnothing(log_par)
        p = (beta = par["beta"],
            m_basal = par["m_basal"], 
            m_size = par["m_size"], 
            d_size = par["d_size"], 
            d_metastasis = par["d_metastasis"])
    else
        p = (beta = 10.0^par["beta"],
            m_basal = 10.0^par["m_basal"], 
            m_size = 10.0^par["m_size"], 
            d_size = 10.0^par["d_size"], 
            d_metastasis = 10.0^par["d_metastasis"])
    end
    jump_prob = remake(jump_problem, p=p)

    # set up ensemble problem and solution
    ensembleprob = EnsembleProblem(jump_prob)
    ensemble_sol = solve(ensembleprob, Tsit5(), EnsembleThreads(); trajectories = npat, tstops=0.0:1.0:endtime)

    patient_data = EnsembleToData(ensemble_sol)
    
    while data_summary(patient_data)["Maximum metastasis increase:"] >5
        ensemble_sol = solve(ensembleprob, Tsit5(), EnsembleThreads(); trajectories = npat, tstops=0.0:1.0:endtime)
        patient_data = EnsembleToData(ensemble_sol)
    end

    return patient_data
end

# alternative simulation algorithm

rng = MersenneTwister(123)

function growth(t, S0, beta)
    return S0 * exp(beta * t)
end

function sample_single(patient_id_, beta_, m_basal_, m_size_, d_size_, d_metas_, S0_, timepoints_)
    TIME_STEP = 0.0001

    t_end = timepoints_[end]
    times = [Float64(timepoints_[1])]
    tumour_sizes = [S0_ + 0.1 * S0_ * randn(rng)]
    metastases = [0]
    death_status = [0]
    DEAD_FLAG = false

    for time in (0.0 + TIME_STEP):TIME_STEP:t_end
        push!(times, time)
        

        if TIME_STEP * (d_size_ * sqrt(growth(time-TIME_STEP, S0_, beta_)) + d_metas_ * metastases[end]) > rand(rng) || DEAD_FLAG
            push!(death_status, 1)
            DEAD_FLAG = true
        else
            push!(death_status, 0)
        end

        if (TIME_STEP * (m_basal_ + m_size_ * sqrt(growth(time-TIME_STEP, S0_, beta_))) > rand(rng))
            push!(metastases, metastases[end] + 1)
        else
            push!(metastases, metastases[end])
        end

        push!(tumour_sizes, growth(time, S0_, beta_))
    end

    # Subset to those values of times which are elements of timepoints_


    sub_idc = findall(x -> x in timepoints_, times)
    death_idc = findfirst(x -> x == 1, death_status)
    if !isnothing(death_idc)
        push!(sub_idc, death_idc)
        sort!(sub_idc)
    end

    times = times[sub_idc]
    tumour_sizes = tumour_sizes[sub_idc]
    metastases = metastases[sub_idc]
    death_status = death_status[sub_idc]

    return_df = DataFrame(
        patient_id = repeat([patient_id_], length(sub_idc)),
        time = times,
        tumor = rand(MvNormal(tumour_sizes, 0.1 .* tumour_sizes)),
        metastasis = metastases,
        death = death_status
    )    

    dead_idc = findall(x -> x == 1, return_df.death)

    if length(dead_idc) > 0
        return_df = return_df[1:dead_idc[1], :]
    end
    return return_df
end

function sample_many(npat_, beta_, m_basal_, m_size_, d_size_, d_metas_, S0_, timepoints_)
    df = DataFrame()
    for i in 1:npat_
        df = vcat(df, sample_single(i, beta_, m_basal_, m_size_, d_size_, d_metas_, S0_, timepoints_))
    end

    
    return df
end

function second_model_alternative_simulator(
    par;
    npat = 200,
    log_par = nothing,
)
    """
        par - Dictionary of parameters on log 10 scale
    """
    S0 = 0.05
    endtime = 30.0
    timepoints = 0.0:1.0:endtime
    
    if isnothing(log_par)
        p = (beta = par["beta"],
            m_basal = par["m_basal"], 
            m_size = par["m_size"], 
            d_size = par["d_size"], 
            d_metastasis = par["d_metastasis"])
    else
        p = (beta = 10.0^par["beta"],
            m_basal = 10.0^par["m_basal"], 
            m_size = 10.0^par["m_size"], 
            d_size = 10.0^par["d_size"], 
            d_metastasis = 10.0^par["d_metastasis"])
    end
    
    # sample data
    alternative_data = sample_many(npat, p.beta, p.m_basal, p.m_size, p.d_size, p.d_metastasis, S0, timepoints);
    data_statistics = data_summary(alternative_data)

    while data_statistics["Maximum metastasis increase:"] >5
        alternative_data = sample_many(npat, p.beta, p.m_basal, p.m_size, p.d_size, p.d_metastasis, S0, timepoints);
        data_statistics = data_summary(alternative_data)
    end

    return alternative_data
end

function second_model_mnr_simulator(
    par;
    npat = 200,
    log_par = nothing,
    rng = MersenneTwister(123),
)
    S0 = 0.05
    endtime = 30.0
    timepoints = 0.0:1.0:endtime
    
    if isnothing(log_par)
        p = (beta = par["beta"],
            m_basal = par["m_basal"], 
            m_size = par["m_size"], 
            d_size = par["d_size"], 
            d_metastasis = par["d_metastasis"])
    else
        p = (beta = 10.0^par["beta"],
            m_basal = 10.0^par["m_basal"], 
            m_size = 10.0^par["m_size"], 
            d_size = 10.0^par["d_size"], 
            d_metastasis = 10.0^par["d_metastasis"])
    end

    par_array = [p.beta, p.m_basal, p.m_size, p.d_size, p.d_metastasis]

    # sample data
    mnr_data = simulate_many_MNR(par_array, npat=npat, S0=S0, endtime=endtime, rng=rng);
    data_statistics = data_summary(mnr_data)

    while data_statistics["Maximum metastasis increase:"] >5
        mnr_data = simulate_many_MNR(par_array, npat=npat, S0=S0, endtime=endtime, rng=rng);
        data_statistics = data_summary(mnr_data)
    end
    return mnr_data
end




#-----------------------------------------------------------------------------------------------------------------------------------------------
# define all the likelihood functions


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
    d_size::Real, 
    d_metas::Real, 
    S0::Real, 
    Nt
    )

    S = TumorGrowth(t, S0, beta)
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

function NumericSurvivalProbability1(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    integral = (1/(LambdaN(t1, t2, beta, m_basal, m_size, S0))) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        Phi(u1, t2, beta, d_size, d_metas, S0, n+1),
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability2(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    integral = (2/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^2)) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            Phi(u2, t2,  beta, d_size, d_metas, S0, n+2),
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability3(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    integral = (6/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^3)) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta, m_basal, m_size, S0) *
                Phi(u2, u3, beta, d_size, d_metas, S0, n+2) *
                Phi(u3, t2, beta, d_size, d_metas, S0, n+3),
            u2, t2
            )[1],
        u1, t2
        )[1],
    t1, t2
    )[1]
    return integral
end


function NumericSurvivalProbability4(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    integral = (24/(LambdaN(t1, t2, beta, m_basal, m_size, S0)^4)) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta, m_basal, m_size, S0) *
                Phi(u2, u3, beta, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta, m_basal, m_size, S0) *
                            Phi(u3, u4, beta, d_size, d_metas, S0, n+3) *
                            Phi(u4, t2, beta, d_size, d_metas, S0, n+4),
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

function NumericSurvivalProbability5(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    integral = 120/((LambdaN(t1, t2, beta, m_basal, m_size, S0))^5) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta, m_basal, m_size, S0) *
                Phi(u2, u3, beta, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta, m_basal, m_size, S0) *
                    Phi(u3, u4, beta, d_size, d_metas, S0, n+3) *
                    hquadrature(u5 -> lambdaN(u5, beta, m_basal, m_size, S0) *
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
    t1, t2
    )[1]
    return integral
end

function NumericSurvivalProbability6(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    integral = 720/((LambdaN(t1, t2, beta, m_basal, m_size, S0))^6) *
    hquadrature(u1 ->lambdaN(u1, beta, m_basal, m_size, S0)*
        Phi(t1, u1, beta, d_size, d_metas, S0, n)*
        hquadrature(u2 -> lambdaN(u2, beta, m_basal, m_size, S0)*
            Phi(u1, u2, beta, d_size, d_metas, S0, n+1) *
            hquadrature(u3 -> lambdaN(u3, beta, m_basal, m_size, S0) *
                Phi(u2, u3, beta, d_size, d_metas, S0, n+2) *
                hquadrature(u4 -> lambdaN(u4, beta, m_basal, m_size, S0) *
                    Phi(u3, u4, beta, d_size, d_metas, S0, n+3) *
                    hquadrature(u5 -> lambdaN(u5, beta, m_basal, m_size, S0) *
                        Phi(u4, u5, beta, d_size, d_metas, S0, n+4) *
                        hquadrature(u6 -> lambdaN(u6, beta, m_basal, m_size, S0) *
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
    t1, t2
    )[1]
    return integral
end

# Numeric integration with FastGaussQuadrature for AD compatibility

function FGIntegral1(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, beta, m_basal, m_size, S0) .* 
        Phi.(t1, ttrans, beta, d_size, d_metas, S0, n) .* 
        Phi.(ttrans, t2, beta, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability1(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)

    return (1/(LambdaN(t1, t2, beta, m_basal, m_size, S0))) * 
    FGIntegral1(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
end

function FGIntegral2(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, beta, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, beta, d_size, d_metas, S0, n) .* 
        FGIntegral1.(ttrans, t2, beta, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability2(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)

    return (2/(LambdaN(t1, t2, beta, m_basal, m_size, S0))^2) * 
    FGIntegral2(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
end


function FGIntegral3(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, beta, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, beta, d_size, d_metas, S0, n) .* 
        FGIntegral2.(ttrans, t2, beta, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability3(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)

    return (6/(LambdaN(t1, t2, beta, m_basal, m_size, S0))^3) * 
    FGIntegral3(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
end


function FGIntegral4(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, beta, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, beta, d_size, d_metas, S0, n) .* 
        FGIntegral3.(ttrans, t2, beta, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability4(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)

    return (24/(LambdaN(t1, t2, beta, m_basal, m_size, S0))^4) * 
    FGIntegral4(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
end

function FGIntegral5(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, beta, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, beta, d_size, d_metas, S0, n) .* 
        FGIntegral4.(ttrans, t2, beta, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability5(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)

    return (120/(LambdaN(t1, t2, beta, m_basal, m_size, S0))^5) * 
    FGIntegral5(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
end

function FGIntegral6(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
    # transform t from [-1, 1] to [t1, t2] first
    x, w = gausslegendre(20)
    ttrans = t1.+((t2-t1)/2).*(x.+1)
    vals = lambdaN.(ttrans, beta, m_basal, m_size, S0) .*
        Phi.(t1, ttrans, beta, d_size, d_metas, S0, n) .* 
        FGIntegral5.(ttrans, t2, beta, m_basal, m_size, d_size, d_metas, S0, n+1) .*
        (t2-t1)/2 # derivative from change of integration limits.
    return dot(w, vals) 
end

function FGSurvivalProbability6(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)

    return (720/(LambdaN(t1, t2, beta, m_basal, m_size, S0))^6) * 
    FGIntegral6(t1, t2, beta, m_basal, m_size, d_size, d_metas, S0, n)
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
    beta, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        surv_prob = Phi(t⁻, t, beta, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        surv_prob = NumericSurvivalProbability1(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        surv_prob = NumericSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        surv_prob = NumericSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        surv_prob = NumericSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        surv_prob = NumericSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        surv_prob = NumericSurvivalProbability6(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
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
    elseif (Nt == Nt⁻ + 6)
        surv_prob = FGSurvivalProbability6(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    else
        println("more than 6 metastasis in one interval, $(Nt-Nt⁻)")
        surv_prob = 1.0
    end
    # did not observe more than 6 jumps in one time interval.

  
    if (surv_prob < 0.0)
        println("surv_prob is negative for t⁻ = $t⁻, t = $t, Nt⁻ = ", ForwardDiff.value(Nt⁻) ,"Nt = ", ForwardDiff.value(Nt), "\n It is ",ForwardDiff.value(surv_prob), "\n", "parameters: ", [beta, m_basal, m_size, d_size, d_metas])
        surv_prob=NumericSurvivalProbability(t⁻, t, θ, Xt⁻, Xt, S0)
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
    beta, m_basal, m_size, d_size, d_metas = θ

    # Unpack data
    St, Nt, Dt = Xt
    St⁻, Nt⁻, Dt⁻ = Xt⁻

    if (Nt == Nt⁻)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*Phi(t⁻, t, beta, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*NumericSurvivalProbability1(t⁻, t, beta,m_basal, m_size, d_size, d_metas, S0,  Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*NumericSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*NumericSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*NumericSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*NumericSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*NumericSurvivalProbability6(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
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
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*Phi(t⁻, t, beta, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 1)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability1(t⁻, t, beta,m_basal, m_size, d_size, d_metas, S0,  Nt⁻)
    elseif (Nt == Nt⁻ + 2)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability2(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 3)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability3(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 4)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability4(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 5)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*AnalyticSurvivalProbability5(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
    elseif (Nt == Nt⁻ + 6)
        death_prob = lambdaD(t, beta, d_size, d_metas, S0, Nt)*FGSurvivalProbability6(t⁻, t, beta, m_basal, m_size, d_size, d_metas, S0, Nt⁻)
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
    beta, m_basal, m_size, d_size, d_metas = θ

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

function TumorNegLogLikelihood(
    beta,
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

function hierarchOptimization(data; x0=[0.26, 0.1, 0.1, 0.1, 0.1, 0.1])

    # get the true parameters from the data
    #beta, m_basal, m_size, d_basal, d_size, d_metas = data.parameters
    #true_par = (beta = beta, m_basal = m_basal, m_size = m_size, d_basal = d_basal, d_size = d_size, d_metas = d_metas);


    # first we want to estimate only beta
    betaOptim(x, p) = TumorNegLogLikelihood(x[1], data)
    betaOptimFunc = OptimizationFunction(betaOptim, Optimization.AutoForwardDiff())
    betaOptimProblem = OptimizationProblem(betaOptimFunc, [x0[1]], lb=zeros(1).+1e-4, ub=ones(1))
    betaEst = solve(betaOptimProblem, LBFGS(), maxiters=10^5)
    println("Beta:", betaEst, "\n")

    # next we want to optimize the metastasis parameters
    metOptim(x, p) = MetastasisNegLogLikelihood([betaEst[1], x[1], x[2]], data)
    metOptimFunc = OptimizationFunction(metOptim, Optimization.AutoForwardDiff())
    metOptimProblem = OptimizationProblem(metOptimFunc, x0[2:3], lb=zeros(2).+1e-4, ub=ones(2))
    metEst = solve(metOptimProblem, LBFGS(), maxiters=10^5)
    println("Metastasis:", metEst, "\n")

    # next we want to optimize the death parameters
    deathOptim(x, p) = DeathNegLogLikelihood([betaEst[1], metEst[1], metEst[2], x[1], x[2]], data)
    deathOptimFunc = OptimizationFunction(deathOptim, Optimization.AutoForwardDiff())
    deathOptimProblem = OptimizationProblem(deathOptimFunc, x0[4:5], lb=zeros(2).+1e-4, ub=ones(2))
    deathEst = solve(deathOptimProblem, LBFGS(), maxiters=10^5)
    est_par = [betaEst[1], metEst[1], metEst[2], deathEst[1], deathEst[2]]

    # print solutions
    #println("True parameter is \n", true_par)
    println("Estimated parameter is \n", est_par)

    #return true_par, est_par
    return est_par
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

#------------------------------------------------------------------------------------------------------------------------------
# # Test area
# t1 = 10.0
# t2 = 11.0
# beta = 0.3
# m_basal = 0.03
# m_size = 0.03
# d_size = 0.01
# d_metas = 0.01
# S0 = 0.05
# n = 2

# using FastGaussQuadrature, LinearAlgebra

# Num1 = NumericSurvivalProbability1(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# An1 = AnalyticSurvivalProbability1(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# FG1 = FGSurvivalProbability1(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)

# Num2 = NumericSurvivalProbability2(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# An2 = AnalyticSurvivalProbability2(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# FG2 = FGSurvivalProbability2(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)

# Num3 = NumericSurvivalProbability3(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# An3 = AnalyticSurvivalProbability3(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# FG3 = FGSurvivalProbability3(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)

# Num4 = NumericSurvivalProbability4(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# An4 = AnalyticSurvivalProbability4(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# FG4 = FGSurvivalProbability4(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)

# Num5 = NumericSurvivalProbability5(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# An5 = AnalyticSurvivalProbability5(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)
# FG5 = FGSurvivalProbability5(t1, t2, beta, m_sigma, m_order, d_size, d_metas, S0, n)