
using MCMCChains
using StatsPlots
using MCMCDiagnosticTools
using Combinatorics
using CSV
using DataFrames

using DifferentialEquations
using JumpProcesses
using Plots
using Statistics
using Distributions
using Random

# include the simulation algorithm
include("general_MNR_simulation_algorithm.jl")

# survival analysis functions

# Kaplan-Meier estimator done by hand
function KM_Estimate_from_patient_data(
    patient_data;
    endtime=30.0
)
    npat = length(unique(patient_data.patient_id))
    death_times = Vector{Any}(undef, npat)
    nalive = 0

    for i in 1:npat
        try
            death_index = findfirst(patient_data[patient_data.patient_id .== i, :].death .== 1)

            death_times[i] = patient_data[patient_data.patient_id .== i, :][death_index, :]
        catch e
            nalive += 1
            death_times[i] = endtime
        end
    end

    # sort by survival
    sorted_survivals = sort(death_times)
    pushfirst!(sorted_survivals, 0.0)

    S = Vector{Any}(undef, npat+1)

    for id in eachindex(sorted_survivals)
        time = sorted_survivals[id]
        if time == 0.0
            S[id] = 1.0
        elseif time == endtime
            S[id] = S[id-1]
        else
            d = count(x -> x == time, death_times)
            S[id] = S[id-1]*(npat - d)/npat
        end
    end

    # plot
    print("$nalive patients survived.")
    plt = Plots.plot(sorted_survivals, S, xlim=(0.0, endtime), xlabel="Time", ylabel="Survival probability", label="KM-estimate")

    return plt
end

# Nelson_aalen estimator kind of thing done by hand
function SurvivalProb_from_patient_data(
    patient_data,
)
    npat = length(unique(patient_data.patient_id))
    death_lines = Vector{Any}(undef, npat)
    nalive = 0

    for i in 1:npat
        try
            death_index = findfirst(patient_data[patient_data.patient_id .== i, :].death .== 1)

            death_lines[i] = patient_data[patient_data.patient_id .== i, :][death_index, :]
        catch e
            nalive += 1
            death_lines[i] = patient_data[patient_data.patient_id .== i, :][end, :]
        end
    end

    survivals = Vector{Any}(undef, npat)

    for i in 1:npat
        survivals[i] = death_lines[i].time[1]
    end

    # Plot Kaplan-Meier curve for survivals

    # sort by survival
    sorted_survivals = sort(survivals)

    # get survival probabilities
    survival_probs = zeros(npat) 

    for i in 1:npat
        survival_probs[i] = (npat - i + 1) / npat
    end

    # plot
    println("$nalive patients survived.")
    plt = Plots.plot(sorted_survivals, survival_probs, xlabel="Time", ylabel="Survival probability", label="SciML")
    return plt
end


# get chains object from dictionary if using pypesto sampler
function dict_to_chain(
    result_dict,
    par_names,
    n_chains,
    n_iter
)
    chs = Array{Float64}(undef, n_iter+1, length(par_names), n_chains)
    for i in 1:n_chains
        ch = Array{Float64}(undef, n_iter+1, length(par_names))
        for id in eachindex(par_names)
            par = par_names[id]
            ch[:, id] = result_dict["parameters"][par][:,i]
        end
        chs[:, :, i] = ch
    end

    return Chains(chs, par_names)
end

# calculate geweke index for MCMCChains

# calculate burn-in using the Geweke diagnostic
function burn_in_from_geweke(chain::Chains, z_threshold::Float64=2.0)
    niter = size(chain)[1]
    npar = size(chain)[2] #-1 if the internal lp is present
    nchains = size(chain)[3]
    # number of fragments
    n = 10
    step = Int(floor(niter/n))
    fragments = 0:step:niter-20
    z = zeros(length(fragments), npar, nchains)
    burn_in_list = []
    for j in 1:nchains
        for (i, indices) in enumerate(fragments)
            z[i, :, j] = DataFrame(gewekediag(chain[indices+1:end,:, j]))[!,"zscore"][1]
        end
        max_z = maximum(abs.(z[:,:,j]), dims=2) #note that it returns a matrix with one column
        idxs = sortperm(max_z[:,1], rev=true) #sort descending
        alpha2 = z_threshold * ones(length(idxs))
        max_z = maximum(abs.(z[:,:,j]), dims=2)
        idxs = sortperm(max_z[:,1], rev=true)
        alpha2 = z_threshold * ones(length(idxs))
        for k in 1:length(max_z)
           alpha2[idxs[k]] = alpha2[idxs[k]]/(length(fragments)-findfirst(==(k), idxs) +1) 
        end
        if any(alpha2.>max_z)
            burn_in = findfirst((alpha2 .> max_z)[:,1]) * step
        else
            burn_in = niter
        end
        append!(burn_in_list, burn_in)
    end
    return Int64(maximum(burn_in_list)) #a conservative choice is the maximum of all chains; or median for a less conservative choice
end


# Visualize data with KM-curves and histogram of metastasis times

function data_visualization(
    patient_data,
    )
    # Get death lines per patient from patient_data
    npat = length(unique(patient_data.patient_id))
    death_lines = Vector{Any}(undef, npat)
    nalive = 0

    for i in 1:npat
        try
            death_index = findfirst(patient_data[patient_data.patient_id .== i, :].death .== 1)

            death_lines[i] = patient_data[patient_data.patient_id .== i, :][death_index, :]
        catch e
            nalive += 1
            death_lines[i] = patient_data[patient_data.patient_id .== i, :][end, :]
        end
    end

    # Plot histogram of metastasis numbers per line in death lines

    metastasis_numbers = zeros(npat)
    for i in 1:npat
        metastasis_numbers[i] = death_lines[i].metastasis[1]
    end

    survivals = Vector{Any}(undef, npat)

    for i in 1:npat
        survivals[i] = death_lines[i].time[1]
    end

    # Plot Kaplan-Meier curve for survivals

    # sort by survival
    sorted_survivals = sort(survivals)

    # get survival probabilities
    survival_probs = zeros(npat) 

    for i in 1:npat
        survival_probs[i] = (npat - i + 1) / npat
    end

    # plot
    print("$nalive patients survived.")
    display(plot(sorted_survivals, survival_probs, xlabel="Time", ylabel="Survival probability", label="Kaplan-Meier curve"))

    # Histogram of metastasis numbers at death
    max_met = Int(maximum(metastasis_numbers))
    display(histogram(metastasis_numbers, bins=max_met, legend=false, xlabel="Metastasis at death", ylabel="Number of patients", title="Histogram of metastasis numbers at death"))
end

# plot mean trajectories of the different processes
function visualize_data_trajectories(
    patient_data;
    timepoints = 0.0:1.0:30.0
    )
    npat = length(unique(patient_data.patient_id))
    tumor_mean = []
    tumor_low_ci = []
    tumor_high_ci = []
    metastasis_mean = []
    metastasis_low_ci = []
    metastasis_high_ci = []
    dead = []
        for t in unique(timepoints)
            push!(tumor_mean, mean(patient_data[patient_data.time .== t, :tumor]))
            push!(tumor_low_ci, quantile(patient_data[patient_data.time .== t, :tumor], 0.05))
            push!(tumor_high_ci, quantile(patient_data[patient_data.time .== t, :tumor], 0.95))
            push!(metastasis_mean, mean(patient_data[patient_data.time .== t, :metastasis]))
            push!(metastasis_low_ci, quantile(patient_data[patient_data.time .== t, :metastasis], 0.05))
            push!(metastasis_high_ci, quantile(patient_data[patient_data.time .== t, :metastasis], 0.95))
            push!(dead, npat - size(patient_data[patient_data.time .== t, :])[1])
        end
    display(plot(timepoints, tumor_mean, ribbon=((tumor_mean-tumor_low_ci),tumor_high_ci-tumor_mean), fillalpha=0.2, label="95% CI for Tumor growth"))
    display(plot(timepoints, metastasis_mean, ribbon=((metastasis_mean-metastasis_low_ci),metastasis_high_ci-metastasis_mean), fillalpha=0.2, label="95% CI for metastasis number"))
    display(plot(timepoints, dead, label="Number of deaths"))
end



function optimizer_trace_plots(results; legend=false, xlim=(0.0, 5.0))
    all_obj_tr = []
    all_t_tr = []
    add_id = 0
    for id in eachindex(results)
        t_tr, obj_tr = results[id]["time_trace"], results[id]["obj_val_trace"]
        min_obj = minimum(obj_tr)
        if isinf(min_obj)
            continue
            add_idx += 1
        end
        obj_tr_sorted = []
        sort_idx= []
        for i in eachindex(obj_tr)
            if i == 1
                push!(obj_tr_sorted, obj_tr[i]-min_obj+1)
                push!(sort_idx, i)
            elseif obj_tr[i]-min_obj+1 <= obj_tr_sorted[end]
                push!(obj_tr_sorted, obj_tr[i]-min_obj+1)
                push!(sort_idx, i)
            end
        end
        t_tr_sorted = t_tr[sort_idx]
        push!(all_obj_tr, obj_tr_sorted)
        push!(all_t_tr, t_tr_sorted)
    end

    nstarts = length(all_obj_tr)
    
    all_timepoints = sort(unique(vcat(all_t_tr...)))

    percentile_values = []
    # median_values = []
    for t in all_timepoints
        values = [all_obj_tr[i][findlast(all_t_tr[i].<=t)] for i in 1:nstarts]
        # push!(median_values, median(values))
        push!(percentile_values, quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]))
    end

    # Because of problems with ribbon and xscale=:log10, we manually log-transform the x-values.
    log_timepoints = log10.(all_timepoints)
    log_t_tr =  [log10.(all_t_tr[i]) for i in 1:nstarts]

    plt = plot()
    plot!(plt, xlabel="Time [s]", ylabel="Log of Neg. Llh. difference")
    plot!(plt, log_t_tr, all_obj_tr, label="", color=:gray, alpha=0.2)
    plot!(plt, log_timepoints, [x[3] for x in percentile_values], label="Median", linewidth=2, color=:blue)
    plot!(plt, log_timepoints, [x[3] for x in percentile_values], ribbon=([x[3]-x[2] for x in percentile_values], [x[4]-x[3] for x in percentile_values]),
            fillalpha=0.5, linewidth=0, color=:lightblue3, label="25-75% CI")
    plot!(plt, log_timepoints, [x[3] for x in percentile_values], ribbon=([x[3]-x[1] for x in percentile_values], [x[5]-x[3] for x in percentile_values]), 
            fillalpha=0.5, linewidth=0, color=:lightblue, label="10-90% CI")
    plot!(xticks=(0:4, ["10⁰", "10¹", "10²", "10³", "10⁴"]))
    plot!(plt, legendfontsize=12, tickfontsize=14, guidefontsize=14, legend=legend, yscale=:log10, xlimit=xlim)
    return plt
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

# Model fit function

function visualize_model_fit(
    patient_data,
    p;
    tumor_model = "exponential",
    metastatic_model = "proportional_intensity",
    rng =  MersenneTwister(123)
)
    nsim = patient_data.patient_id[end]
    endtime = maximum(patient_data.time)

    # check for tumor model
    if tumor_model == "exponential"
        function expgrowth!(du, u, p, t)
            du[1] = p[1]* u[1]
        end
        
        function TumorPath(S0, p; endtime=30.0)
            prob = ODEProblem(expgrowth!, [S0], (0.0, endtime), [p.beta])
            sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
            return sol
        end
    else
        error("Tumor model not recognized. Please choose one of 'exponential'.")
    end

    # check for metastatic model
    if metastatic_model == "proportional_intensity"
        sim_data = simulate_many_MNR(p, TumorPath, npat=nsim, endtime=endtime, metastatic_model="proportional_intensity", rng=rng)
    elseif metastatic_model == "cell_division"
        sim_data = simulate_many_MNR(p, TumorPath, npat=nsim, endtime=endtime, metastatic_model="cell_division", rng=rng)
    else
        error("Metastatic model not recognized. Please choose one of 'proportional_intensity' or 'cell_division'")
    end

    println("Original data \n\n", data_summary(patient_data))
    println("\n Simulated data\n\n", data_summary(sim_data))


    timepoints = 0.0:1.0:endtime
    npat = nsim

    patient_tumor_mean = []
    patient_tumor_low_ci = []
    patient_tumor_high_ci = []
    sim_tumor_mean = []
    sim_tumor_low_ci = []
    sim_tumor_high_ci = []
    patient_metastasis_mean = []
    patient_metastasis_low_ci = []
    patient_metastasis_high_ci = []
    sim_metastasis_mean = []
    sim_metastasis_low_ci = []
    sim_metastasis_high_ci = []

    for t in unique(timepoints)
        push!(patient_tumor_mean, mean(patient_data[patient_data.time .== t, :tumor]))
        push!(patient_tumor_low_ci, quantile(patient_data[patient_data.time .== t, :tumor], 0.05))
        push!(patient_tumor_high_ci, quantile(patient_data[patient_data.time .== t, :tumor], 0.95))
        push!(sim_tumor_mean, mean(sim_data[sim_data.time .== t, :tumor]))
        push!(sim_tumor_low_ci, quantile(sim_data[sim_data.time .== t, :tumor], 0.05))
        push!(sim_tumor_high_ci, quantile(sim_data[sim_data.time .== t, :tumor], 0.95))
        push!(patient_metastasis_mean, mean(patient_data[patient_data.time .== t, :metastasis]))
        push!(patient_metastasis_low_ci, quantile(patient_data[patient_data.time .== t, :metastasis], 0.05))
        push!(patient_metastasis_high_ci, quantile(patient_data[patient_data.time .== t, :metastasis], 0.95))
        push!(sim_metastasis_mean, mean(sim_data[sim_data.time .== t, :metastasis]))
        push!(sim_metastasis_low_ci, quantile(sim_data[sim_data.time .== t, :metastasis], 0.05))
        push!(sim_metastasis_high_ci, quantile(sim_data[sim_data.time .== t, :metastasis], 0.95))
    end
    # plot data trajectories against each other

    plt1 = plot(timepoints, patient_tumor_mean, ribbon=((patient_tumor_mean-patient_tumor_low_ci),patient_tumor_high_ci-patient_tumor_mean), fillalpha=0.2, label="Original data", title="Tumor trajectories")
    plot!(plt1, timepoints, sim_tumor_mean, ribbon=((sim_tumor_mean-sim_tumor_low_ci),sim_tumor_high_ci-sim_tumor_mean), fillalpha=0.2, label="Simulated data")
    plot!(plt1, legendfontsize=12, tickfontsize=12, guidefontsize=12, titlefontsize=12)
    display(plt1)

    plt2 = plot(timepoints, patient_metastasis_mean, ribbon=((patient_metastasis_mean-patient_metastasis_low_ci),patient_metastasis_high_ci-patient_metastasis_mean), fillalpha=0.2, label="Original data")
    plot!(plt2, timepoints, sim_metastasis_mean, ribbon=((sim_metastasis_mean-sim_metastasis_low_ci),sim_metastasis_high_ci-sim_metastasis_mean), fillalpha=0.2, label="Simulated data")
    plot!(xlabel="Time [months]", ylabel="Number of metastasis")
    plot!(plt2, legendfontsize=12, tickfontsize=12, guidefontsize=12, titlefontsize=12)
    display(plt2)

    # plot KM curves against each other
    patient_death_lines = Vector{Any}(undef, npat)
    sim_death_lines = Vector{Any}(undef, npat)

    for i in 1:npat
        try
            patient_death_index = findfirst(patient_data[patient_data.patient_id.== i, :].death .== 1)
            patient_death_lines[i] = patient_data[patient_data.patient_id .== i, :][patient_death_index, :]
            sim_death_index = findfirst(sim_data[sim_data.patient_id .== i, :].death .== 1)
            sim_death_lines[i] = sim_data[sim_data.patient_id .== i, :][sim_death_index, :]
        catch e
            patient_death_lines[i] = patient_data[patient_data.patient_id .== i, :][end, :]
            sim_death_lines[i] = sim_data[sim_data.patient_id .== i, :][end, :]
        end
    end


    patient_survivals = Vector{Any}(undef, npat)
    sim_survivals = Vector{Any}(undef, npat)

    for i in 1:npat
        patient_survivals[i] = patient_death_lines[i].time[1]
        sim_survivals[i] = sim_death_lines[i].time[1]
    end

    # Plot Kaplan-Meier curve for survivals

    # sort by survival
    patient_survivals = sort(patient_survivals)
    sim_survivals = sort(sim_survivals)

    # get survival probabilities
    patient_survival_probs = zeros(npat) 
    sim_survival_probs = zeros(npat) 

    for i in 1:npat
        patient_survival_probs[i] = (npat - i + 1) / npat
        sim_survival_probs[i] = (npat - i + 1) / npat
    end

    # plot
    plt3 = plot(patient_survivals, patient_survival_probs, xlabel="Time", ylabel="Survival probability", label="Original data", title="KM curves")
    plot!(plt3, sim_survivals, sim_survival_probs, xlabel="Time", ylabel="Survival probability", label="SImulated data")
    plot!(plt3, legendfontsize=12, tickfontsize=12, guidefontsize=12, titlefontsize=12)
    display(plt3)

    return plt1, plt2, plt3
end






# Sankey plot try

# customized sankey-plot for patient data

function custom_sankey(
    patient_data::DataFrame,
    patient_id_column::String, 
    time_column::String,
    category_column::String;
    death_column::String="death",
    return_plot_data::Bool=false
    )

    # check columns names 
    try
        patient_data[!, patient_id_column]
    catch
        error("patient_id_column not found in patient_data")
    end
    try 
        patient_data[!, time_column]
    catch
        error("time_column not found in patient_data")
    end
    try
        patient_data[!, category_column]
    catch
        error("category_column not found in patient_data")
    end

    timepoints = patient_data[!, time_column]|> unique;
    categories = unique(patient_data[!, category_column]);

    # We add one for death if necessary
    if (death_column in names(patient_data))
        print("Including death as a state!")
        categories = vcat(categories, "d")

        all_sources = []
        all_targets = []
        final_sources = []
        final_targets = []
        final_volumes = []

        for i in unique(patient_data[!, patient_id_column])
            one_patient = patient_data[patient_data[!, patient_id_column] .== i, :]
            for j in 1:length(one_patient[!, time_column])-1
                if (one_patient[j+1, death_column] == 0) # no death_column
                    push!(all_sources, [one_patient[j, time_column], one_patient[j, category_column]])
                    push!(all_targets, [one_patient[j+1, time_column], one_patient[j+1, category_column]])
                elseif (one_patient[j, death_column] == 0 && one_patient[j+1, death_column] == 1) # new death_column
                    push!(all_sources, [one_patient[j, time_column], one_patient[j, category_column]])
                    push!(all_targets, [one_patient[j+1, time_column], "d"])
                else # staying death
                    push!(all_sources, [one_patient[j, time_column], "d"])
                    push!(all_targets, [one_patient[j+1, time_column], "d"])
                end

            end
        end
    else
        print("No death state included!")
        all_sources = []
        all_targets = []
        final_sources = []
        final_targets = []
        final_volumes = []

        for i in unique(patient_data[!, patient_id_column])
            one_patient = patient_data[patient_data[!, patient_id_column] .== i, :]

            for j in 1:length(one_patient[!, time_column])-1

                push!(all_sources, [one_patient[j, time_column], one_patient[j, category_column]])
                push!(all_targets, [one_patient[j+1, time_column], one_patient[j+1, category_column]])

            end
        end
    end

    # get unique source and target vectors and transition volumes
    unique_sources = unique(all_sources)
    unique_targets = unique(all_targets)
    volumes = []

    for s in unique_sources
        for t in unique_targets
            s_inds = findall(string.(all_sources) .== string(s))
            t_inds = findall(string.(all_targets) .== string(t))
            inds = intersect(s_inds, t_inds)
            if length(inds) > 0
                push!(final_sources, s)
                push!(final_targets, t)
                push!(final_volumes, length(inds))
            end
        end
    end

    # get nodes 
    all_nodes = unique(
        vcat(final_sources, final_targets)
    );
    source_ids = [findall(all_nodes .== final_sources[i,:])[1] for i in 1:size(final_sources)[1]];
    target_ids = [findall(all_nodes .== final_targets[i,:])[1] for i in 1:size(final_targets)[1]];

    # get timepoints
    timepoints = patient_data[!, time_column]|> unique;

    x = zeros(length(all_nodes))
    y = zeros(length(all_nodes))

    # ordering on x-axis by timepoints
    for i in eachindex(timepoints)
        t = timepoints[i]
        for j in eachindex(all_nodes)
            if (all_nodes[j][1] == t)
                x[j] = (1/(length(timepoints)+1))*i
            end
        end

        t_nodes = [all_nodes[j] for j in eachindex(all_nodes) if (all_nodes[j][1] .== t)]

        # start at top
        y_val = 0.001

        for k in categories
            node = [t, k]
            if (node in t_nodes)

                # get node index
                node_ind = [j for j in eachindex(all_nodes) if all_nodes[j] == node][1]

                # get normalized node_size
                if (t == timepoints[end]) # last timepoints are just target nodes never sources
                    node_size = sum(final_volumes[findall(target_ids .== node_ind)])/200
                else
                    node_size = sum(final_volumes[findall(source_ids .== node_ind)])/200
                end

                # increase y-value based on node size/2 to get middle point of node as y-value or minimal distance
                y_val += max(node_size/2, 0.02)

                # set y-value
                y[node_ind] =y_val

                # increase y-value again to be at end of node 
                y_val += max(node_size/2, 0.02)
            end
        end
    end

    if (return_plot_data)
        print("Plot data is returned in the following order node labels, x-coord, y-coord, source_ids, target_ids, volumes. \n")
        return all_nodes, x, y, source_ids, target_ids, final_volumes
    else
        # create plot
        PlotlyJS.plot(sankey(
            node = attr(
                label = all_nodes,
                x = x,
                y = y,
                pad = 15,
                thickness = 15,
                line =attr(color = "black", width = 0.5),
                color = "blue"
                ),
            link = attr(
                source = source_ids .-1,
                target = target_ids .-1,
                value = final_volumes
                )
            ),
            Layout(title_text="Sankey Diagram", font_size=10)
        )
    end
end;
