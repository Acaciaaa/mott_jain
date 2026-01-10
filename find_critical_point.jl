include(joinpath(@__DIR__, ".", "pad_su3.jl"))
include(joinpath(@__DIR__, ".", "pad_su2.jl"))
using .PADsu3
using .PADsu2
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf
using XLSX
using DataFrames
using Statistics
BLAS.set_num_threads(8)

function method_minimize_singlet_gap_distance()
    xlsx_file = "data/singlet_gap.xlsx"
    sheetname = "zoomin"

    tbl = XLSX.readtable(xlsx_file, sheetname; infer_eltypes=true)
    df  = DataFrame(tbl)

    x = df[!, 1]
    series_idx = (ncol(df)-3):ncol(df)
    labels = string.(names(df)[series_idx])
    Y = Matrix{Float64}(df[!, series_idx])

    col_factors = sqrt.(parse.(Float64, labels))
    Y .= Y .* col_factors'

    row_mean = sum(Y, dims=2) ./ size(Y, 2)
    diffs    = Y .- row_mean
    var_pop  = vec(sum(diffs .^ 2, dims=2) ./ size(Y, 2))

    imin  = argmin(var_pop)
    x_min = x[imin]
    v_min = var_pop[imin]
    @info x_min v_min

    fig = Figure(size=(720, 520))
    ax  = Axis(fig[1, 1];
        xlabel = string(names(df)[1]),
        ylabel = "Population variance"
    )

    lines!(ax, x, var_pop; linewidth=2)
    scatter!(ax, [x_min], [v_min]; markersize=12)
    text!(ax, @sprintf(" min at x=%.3f", x_min), position=(x_min, v_min+0.00001), align=(:left, :bottom))
    vlines!(ax, [x_min]; linestyle=:dash)

    fig
end

function method_minimize_cost_function_su3(nm1)
    mus, results_vec = PADsu3.read_results(nm1,"data/results_$(nm1).jld2")
    Qs      = similar(mus)
    factors = similar(mus)
    for (i, mu) in enumerate(mus)
        results = results_vec[i]
        E0 = results[1][1]
        enrg_cal = [
            filter(st -> st[2] ≈ 2 && st[3] ≈ 0, results)[1][1] - filter(st -> st[2] ≈ 0 && st[3] ≈ 0, results)[2][1], # ∂S - S
            filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[1][1] - E0, # J
            filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[3][1] - E0, # ϵ∂J
            filter(st -> st[2] ≈ 6 && st[3] ≈ 3, results)[1][1] - E0, # ∂J
            filter(st -> st[2] ≈ 6 && st[3] ≈ 0, results)[1][1] - E0 # T
        ]
        #@info [mu;enrg_cal]
        dim_cal = Float64[1, 2, 3, 3, 3]
        factor = (enrg_cal' * dim_cal) / (dim_cal' * dim_cal)
        factors[i] = factor
        Q = sqrt(mean(((enrg_cal ./ factor) .- dim_cal) .^ 2))
        Qs[i] = Q
    end
    i_min = argmin(Qs)
    μc = mus[i_min]

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "μ", ylabel = "Q")

    lines!(ax, mus, Qs)
    vlines!(ax, [μc]; linestyle = :dash)
    text!(ax, @sprintf("min at μc=%.4f", μc);
        position = (μc, Qs[i_min]),
        align = (:left, :bottom),)
    #display(fig)
    @info μc
    @info Qs[i_min]
    return μc,Qs[i_min]
end

function method_minimize_cost_function_su2(nm1)
    mus, results_vec = PADsu2.read_results(nm1)
    Qs      = similar(mus)
    factors = similar(mus)
    for (i, mu) in enumerate(mus)
        results = results_vec[i]
        E0 = results[1][1]
        enrg_cal = [
            filter(st -> st[2] ≈ 2 && st[3] ≈ 0, results)[1][1] - filter(st -> st[2] ≈ 0 && st[3] ≈ 0, results)[2][1], # ∂S - S
            filter(st -> st[2] ≈ 2 && st[3] ≈ 2, results)[1][1] - E0, # J
            filter(st -> st[2] ≈ 2 && st[3] ≈ 2, results)[2][1] - E0, # ϵ∂J
            filter(st -> st[2] ≈ 6 && st[3] ≈ 2, results)[1][1] - E0, # ∂J
            filter(st -> st[2] ≈ 6 && st[3] ≈ 0, results)[1][1] - E0 # T
        ]
        @info [mu;enrg_cal]
        dim_cal = Float64[1, 2, 3, 3, 3]
        factor = (enrg_cal' * dim_cal) / (dim_cal' * dim_cal)
        factors[i] = factor
        Q = sqrt(mean(((enrg_cal ./ factor) .- dim_cal) .^ 2))
        Qs[i] = Q
    end
    i_min = argmin(Qs)
    μc = mus[i_min]

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "μ", ylabel = "Q")

    lines!(ax, mus, Qs)
    vlines!(ax, [μc]; linestyle = :dash)
    text!(ax, @sprintf("min at μc=%.3f", μc);
        position = (μc, Qs[i_min]),
        align = (:left, :bottom),)
    display(fig)
    @info μc
    return
end

function tool_write_results(nm1)
    k = 7
    P = PADsu3.build_model(nm1=nm1)
    #P = PADsu2.build_model(nm1=nm1)
    mus = collect(range(0.052, 0.058, length=6))
    PADsu3.write_results(P, mus,  0.4,1.0,0.3,k)
    #PADsu2.write_results(P, mus, k)
end




#method_minimize_singlet_gap_distance()
for i in [6]
    tool_write_results(i)
end
for i in [6]
    method_minimize_cost_function_su3(i)
end

#method_minimize_cost_function_su2(3)


# P = PADsu3.build_model(nm1=5)
# k = 30
# # mus, results_vec = PADsu3.read_results(5)
# for (i, mu) in enumerate(mus)
#     #local results = results_vec[i]
#     local results = PADsu3.lowest_k_states(P, mu, k)
#     local E0 = results[1][1]
#     J = filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[1]
#     ϵ∂J = filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[2]
#     similar_ϵ∂J = filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[3]
#     enrg_cal = [
#         J[1] - E0, # J
#         J[4:6],
#         ϵ∂J[1] - E0, # ϵ∂J
#         ϵ∂J[4:6],
#         similar_ϵ∂J[1] - E0,
#         similar_ϵ∂J[4:6],
#         #filter(st -> st[2] ≈ 6 && st[3] ≈ 3, results)[1][1] - E0, # ∂J
#         #filter(st -> st[2] ≈ 6 && st[3] ≈ 3, results)[2][1] - E0,
#     ]
#     @info enrg_cal
# end