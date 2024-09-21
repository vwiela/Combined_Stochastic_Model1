using Random
using Roots
using Trapz
using JLD2
using DataFrames
using Distributions
using DiffEqNoiseProcess
using DifferentialEquations
using NaNMath


rng = MersenneTwister(123);

# rate functions for the Poisson processes

# ToDo: Check whether that S really returns a vector always!!!

# metastasis rate
function lambdaN(t, p, S)
    return p.m_basal + p.m_size*NaNMath.sqrt(S(t)[1])
end

# integrate the above rate over time interval
function LambdaN(t1, t2, p, S)
    tspan = t1:0.001:t2
    return trapz(tspan, lambdaN.(tspan, Ref(p), Ref(S)))
end

function lambdaD(t, p, S, n)
    return p.d_size*NaNMath.sqrt(S(t)[1]) + p.d_metastasis*n
end

function LambdaD(t1, t2, p, S, n)
    tspan = t1:0.001:t2
    return trapz(tspan, lambdaD.(tspan, Ref(p), Ref(S), Ref(n)))
end


function dLambdaN(t1, t2, p, S)
    return lambdaN(t2, p, S)
end

function dLambdaD(t1, t2, p, S, n)
    return lambdaD(t2, p, S, n)
end

# metastasis rate based on cell division model from Gasparini and Humphreys
function DivisionlambdaN(t, p, S; S0=0.05)
    p.m_sigma*p.beta*(p.beta*t/log(2))^(p.m_order)
end

function DivisionLambdaN(t1, t2, p, S; S0=0.05)
    p.m_sigma/(2*log(2)^(p.m_order))*p.beta^(p.m_order+1)*(t2^(p.m_order+1)-t1^(p.m_order+1))
end

function dDivisionLambdaN(t1, t2, p, S; S0=S0)
    return DivisionlambdaN(t2, p, S; S0=S0)
end

function death_t2(
    t1,
    p,
    S,
    Nt,
    outcome,
)
    f(dt) = LambdaD(t1, t1+dt, p, S, Nt)-outcome

    Df(dt) = dLambdaD(t1, t1+dt, p, S, Nt)
    try
        return find_zero((f, Df), t1, Roots.Newton)
    catch
        # println("Newton failed on death. $(t1), $(outcome), $Nt")
        T = 1.0
        while f(T) < 0
            T += 1.0
        end
        # check for Nan, which sometimes happens but, then T is large anyways probably and we can just retunr the last non NaN T
        if isnan(f(T))
            return T-1
        end
        # return find_zero(f, (t1, T), Roots.Bisection())
        try 
            find_zero(f, (0.0,T))
        catch
            println("Bisection failed with T = $(T)")
            return fzero(f, T)
        end
    end
end

function metastasis_t2(
    t1,
    p,
    S,
    outcome;
    endtime = 30.0
)
    f(dt) = LambdaN(t1, t1+dt, p, S)-outcome

    Df(dt) = dLambdaN(t1, t1+dt, p, S)
    try
        # println("try statement")
        return find_zero((f, Df), t1, Roots.Newton)
    catch 
        T = 1.0
        # println("Newton failed on met. $(t1), $outcome")
        if f(0.0) > 0
            println("f(0) > 0, $(t1), $p, $outcome")
            ValueError("f(0) > 0")
        end
        while f(T) < 0 && T < endtime
            T += 1.0
        end
        if T == endtime
            return T+1
        end
        try 
            find_zero(f, (0.0,T))
        catch
            println("Bisection failed with T = $(T)")
            return fzero(f, T)
        end
    end
end

function division_metastasis_t2(
    t1,
    p,
    S,
    outcome;
    endtime= 30.0
)
    f(dt) = DivisionLambdaN(t1, t1+dt, p, S)-outcome

    Df(dt) = dDivisionLambdaN(t1, t1+dt, p, S)
    try
        return find_zero((f, Df), t1, Roots.Newton)
    catch 
        T = 1.0
        # println("Newton failed on met. $(t1), $outcome")
        while f(T) < 0 && T < endtime
            T += 1.0
        end
        if T == endtime
            return T+1
        end
        return find_zero(f, (0.0,T))
    end
end


function single_MNR_simulation_proportional_intensity(
    p::NamedTuple,
    ODESol;
    S0=0.05,
    endtime=30.0,
    rng = MersenneTwister(123),
)

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
        Δt_D = death_t2(t, p, ODESol, N[end], P_D-T_D)
        Δt_N = metastasis_t2(t, p, ODESol, P_N-T_N, endtime=endtime)
        if Δt_D < Δt_N
            t = t+Δt_D
            push!(times, t)
            push!(X, ODESol(t)[1])
            push!(N, N[i])
            push!(Death, 1)

            # update firing times
            T_D = T_D + LambdaD(times[end-1], t, p, ODESol, N[end-1])
            T_N = T_N + LambdaN(times[end-1], t, p, ODESol)

            r_D = randexp(rng) 
            P_D = P_D+r_D

            i += 1
        else
            t = t+Δt_N
            push!(times, t)
            push!(X, ODESol(t)[1])
            push!(N, N[i]+1)
            push!(Death, 0)

            # update firing times
            T_D = T_D + LambdaD(times[end-1], t, p, ODESol, N[end-1])
            T_N = T_N + LambdaN(times[end-1], t, p, ODESol)

            r_N = randexp(rng)
            P_N = P_N+r_N

            i += 1
        end

    end

    return Dict("t" => times, "X" => X, "N" => N, "Death" => Death)
end

function single_MNR_simulation_cell_division(
    p::NamedTuple,
    ODESol;
    S0=0.05,
    endtime=30.0,
    rng = MersenneTwister(123),
)

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
        Δt_D = death_t2(t, p, ODESol, N[end], P_D-T_D)
        Δt_N = division_metastasis_t2(t, p, ODESol, P_N-T_N, endtime=endtime)
        if Δt_D < Δt_N
            t = t+Δt_D
            push!(times, t)
            push!(X, ODESol(t)[1])
            push!(N, N[i])
            push!(Death, 1)

            # update firing times
            T_D = T_D + LambdaD(times[end-1], t, p, ODESol, N[end-1])
            T_N = T_N + DivisionLambdaN(times[end-1], t, p, ODESol)

            r_D = randexp(rng) 
            P_D = P_D+r_D

            i += 1
        else
            t = t+Δt_N
            push!(times, t)
            push!(X, ODESol(t)[1])
            push!(N, N[i]+1)
            push!(Death, 0)

            # update firing times
            T_D = T_D + LambdaD(times[end-1], t, p, ODESol, N[end-1])
            T_N = T_N + DivisionLambdaN(times[end-1], t, p, ODESol)

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
    p::NamedTuple,
    TumorPath;
    npat = 200,
    S0=0.05,
    endtime=30.0,
    sigma = 0.1,
    metastatic_model = "proportional_intensity",
    rng = MersenneTwister(123),
    )
    # check for metastatic model
    if metastatic_model == "proportional_intensity"
        single_MNR_simulation = single_MNR_simulation_proportional_intensity
    elseif metastatic_model == "cell_division"
        single_MNR_simulation = single_MNR_simulation_cell_division
    else
        error("Metastatic model not recognized. Please choose one of 'proportional_intensity' or 'cell_division'")
    end

    timepoints = 0.0:1.0:endtime

    # initialize data frame
    df = DataFrame(patient_id=Int64[], time=Float64[], tumor=Float64[], metastasis=Int64[], death=Int64[])

    # loop over N simulations
    for i in 1:npat
        # get function for Tumor Growth ODE
        ODESol = TumorPath(S0, p; endtime=endtime)
        pat_id = i
        try
            sim = single_MNR_simulation(p, ODESol, S0=S0, endtime=endtime, rng=rng)
        catch e
            println("Simulation failed for patient $i")
            print(e)
            return ODESol
        end
        sim = single_MNR_simulation(p, ODESol, S0=S0, endtime=endtime, rng=rng)
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
            tumor_size = rand(Normal(ODESol(t)[1], ODESol(t)[1]*sigma))
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

