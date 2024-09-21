# using NLsolve # Roots performed a bit more accurate
using Random
using ForwardDiff
using Roots
using JLD2
using DataFrames
using Distributions

include("functionalities.jl")

rng = MersenneTwister(123);

function LambdaN(
    t1, 
    t2, 
    beta::Real, 
    m_basal::Real, 
    m_size::Real, 
    S0::Real
    )

    dt = (t2-t1)
    return dt*m_basal+ (2*m_size*sqrt(S0))*(sqrt(exp(beta*t2))-sqrt(exp(beta*t1)))/beta
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

function death_t2(
    t1,
    beta,
    d_size,
    d_metastasis,
    S0,
    Nt,
    outcome,
)

    D(f) = x -> ForwardDiff.derivative(f, float(x))

    f(dt) = LambdaD(t1, t1+dt, beta, d_size, d_metastasis, S0, Nt)-outcome

    x0 = rand(rng)
    return find_zero((f, D(f)), x0)
end

function metastasis_t2(
    t1,
    beta,
    m_basal,
    m_size,
    S0,
    outcome,
)

    D(f) = x -> ForwardDiff.derivative(f, float(x))

    f(dt) = LambdaN(t1, t1+dt, beta, m_basal, m_size, S0)-outcome

    x0 = rand(rng)
    return find_zero((f, D(f)), x0)
end



function single_MNR_simulation(
    θ::Vector{Float64};
    S0=0.05,
    endtime=30.0,
    rng = MersenneTwister(123),
)

    # unpack parameter vectors
    beta, m_basal, m_size, d_size, d_metastasis = θ

    # initialize times and species
    times = zeros(1)
    X = zeros(1)
    X[1] = S0
    N = zeros(1)
    Death = zeros(1)

    # initialize firing times
    T_N = 0.0
    T_D = 0.0
    r_N = randexp(rng)
    P_N = r_N
    r_D = randexp(rng)
    P_D = r_D

    # start loop until death or endtime is reached
    i = 1
    while times[end] < endtime && Death[end] == 0
        t = times[i]
        Δt_D = death_t2(t, beta, d_size, d_metastasis, S0, N[end], P_D-T_D)
        Δt_N = metastasis_t2(t, beta, m_basal, m_size, S0, P_N-T_N)
        if Δt_D < Δt_N
            t = t+Δt_D
            push!(times, t)
            push!(X, S0*exp(beta*t))
            push!(N, N[i])
            push!(Death, 1)

            # update firing times
            T_D = T_D + LambdaD(times[end-1], t, beta, d_size, d_metastasis, S0, N[end-1])
            T_N = T_N + LambdaN(times[end-1], t, beta, m_basal, m_size, S0)

            r_D = randexp(rng) 
            P_D = P_D+r_D

            i += 1
        else
            t = t+Δt_N
            push!(times, t)
            push!(X, S0*exp(beta*t))
            push!(N, N[i]+1)
            push!(Death, 0)

            # update firing times
            T_D = T_D + LambdaD(times[end-1], t, beta, d_size, d_metastasis, S0, N[end-1])
            T_N = T_N + LambdaN(times[end-1], t, beta, m_basal, m_size, S0)

            r_N = randexp(rng)
            P_N = P_N+r_N

            i += 1
        end

    end

    return Dict("t" => times, "X" => X, "N" => N, "Death" => Death)
end


#--------------------------------------------------------------------------------------------------------------------------------------------
# Sample many and convert into the patient data frame format

function simulate_many_MNR(
    θ::Vector{Float64};
    npat = 200,
    S0=0.05,
    endtime=30.0,
    rng = MersenneTwister(123),
)
    timepoints = 0.0:1.0:endtime

    # initialize data frame
    df = DataFrame(patient_id=Int64[], time=Float64[], tumor=Float64[], metastasis=Int64[], death=Int64[])

    # loop over N simulations
    for i in 1:npat
        pat_id = i
        sim = single_MNR_simulation(θ, S0=S0, endtime=endtime, rng=rng)
        # get death time
        death_idc = findfirst(sim["Death"] .== 1)
        if !isnothing(death_idc)
            death_time = sim["t"][death_idc]
            if death_time > endtime
                death_time = endtime
            end
        else
            death_time = endtime
        end
        times = vcat(timepoints[timepoints .< death_time], death_time)
        for t in times
            t_idc = findall(sim["t"] .<= t)[end]
            tumor_size = rand(Normal(S0*exp(θ[1]*t), S0*exp(θ[1]*t)*0.1))
            push!(df, (pat_id, t, tumor_size, sim["N"][t_idc], sim["Death"][t_idc]))
        end
    end

    return df
end

#------------------------------------------------------------------------------------------
# # test_sim
# S0 = 0.05
# beta = 0.3
# m_basal = 0.04
# m_size = 0.04
# d_size = 0.01
# d_metastasis = 0.01
# θ = [beta, m_basal, m_size, d_size, d_metastasis]

# endtime=30.0


# sim = single_MNR_simulation(θ, S0=0.05, endtime=30.0, rng=MersenneTwister(123))

# timepoints = 0.0:1.0:endtime
# # initialize data frame
# df = DataFrame(patient_id=Int64[], time=Float64[], tumor=Float64[], metastasis=Int64[], death=Int64[])

# pat_id = 1
# death_idc = findfirst(sim["Death"] .== 1)
# if !isnothing(death_idc)
#     death_time = sim["t"][death_idc]
#     if death_time > endtime
#         death_time = endtime
#     end
# else
#     death_time = endtime
# end
# times = vcat(timepoints[timepoints .< death_time], death_time)
# for t in times
#     t_idc = findall(sim["t"] .<= t)[end]
#     tumor_size = rand(Normal(S0*exp(θ[1]*t), S0*exp(θ[1]*t)*0.1))
#     push!(df, (pat_id, t, tumor_size, sim["N"][t_idc], sim["Death"][t_idc]))
# end


# df

# # initialize times and species
# S0 = 0.05
# endtime = 30.0

# # simulate
# npat = 500

# df = simulate_many_MNR(θ, npat=npat, S0=0.05, endtime=30.0, rng=MersenneTwister(123))

# data_summary(df)


# # save data
# data_path = joinpath(pwd(),"second_model/data/simplified_model/MNR_data_$(npat)_patients.jld2")
# save(data_path, "mnr_data", df)

#------------------------------------------------------------------------------------------
# test single steps


# # initialize times and species
# times = zeros(1)
# X = zeros(1)
# X[1] = S0
# N = zeros(1)
# Death = zeros(1)

# # initialize firing times
# T_N = 0.0
# T_D = 0.0
# r_N = rand(rng)
# P_N = log(1/r_N)
# r_D = rand(rng)
# P_D = log(1/r_D)

# # start loop until death or endtime is reached
# i = 1

# t = times[i]

# Δt_D = death_t2(t, beta, d_size, d_metastasis, S0, N[end], P_D-T_D)
# Δt_N = metastasis_t2(t, beta, m_basal, m_size, S0, P_N-T_N)
# if Δt_D < Δt_N
#     t = t+Δt_D
#     push!(times, t)
#     push!(X, S0*exp(beta*t))
#     push!(N, N[i])
#     push!(Death, 1)

#     # update firing times
#     T_D = LambdaD(0, t, beta, d_size, d_metastasis, S0, N[end])
#     T_N = LambdaN(0, t, beta, m_basal, m_size, S0)

#     r_D = rand(rng) 
#     P_D = P_D+log(1/r_D)

#     i += 1
# else
#     t = t+Δt_N
#     push!(times, t)
#     push!(X, S0*exp(beta*t))
#     push!(N, N[i]+1)
#     push!(Death, 0)

#     # update firing times
#     T_D = LambdaD(0, t, beta, d_size, d_metastasis, S0, N[end])
#     T_N = LambdaN(0, t, beta, m_basal, m_size, S0)

#     r_N = rand(rng)
#     P_N = P_N+log(1/r_N)

#     i+=1
# end

# X
# N
# Death
# sim = Dict("t" => times, "X" => X, "N" => N, "Death" => Death)


# timepoints = 0.0:1.0:endtime
# death_idc = findfirst(sim["Death"] .== 1)
# if !isnothing(death_idc)
#     death_time = sim["t"][death_idc]
#     if death_time > endtime
#         death_time = endtime
#     end
# else
#     death_time = endtime
# end
# times = vcat(timepoints[timepoints .< death_time], death_time)

# # initialize data frame
# df = DataFrame(patient_id=Int64[], time=Float64[], tumor=Float64[], metastasis=Int64[], death=Int64[])
# pat_id = 1
# for t in times
#     t_idc = findall(sim["t"] .<= t)[end]
#     print(t_idc)
#     tumor_size = S0*exp(θ[1]*t)*(1+0.1*randn(rng))
#     push!(df, (pat_id, t, tumor_size, sim["N"][t_idc], sim["Death"][t_idc]))
# end

