include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf
using Statistics
using JLD2, Dates
function method_minimize_cost_function_su3(nm1)
    mus, results_vec = PADsu3.read_results(nm1,joinpath(file_path, "results_$(nm1).jld2"))
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
    return μc,Qs[i_min],factors[i_min]
end

function tool_for_generator()
    results, bss = PADsu3.for_generator(P, μc,coef[1],coef[2],coef[3],100)
    jldsave(joinpath(file_path, "generator_$(nm1).jld2"); results=results, bss=bss)
    results, bs0, bsp, bsm = PADsu3.for_generator_special(P, μc,coef[1],coef[2],coef[3],200)
    jldsave(joinpath(file_path, "generator_othersector_$(nm1).jld2"); results=results, bs0=bs0, bsp=bsp, bsm=bsm)
end

file_path = "../b"
#coef = [0.4,0.9,0.3]
coef = [0.6,1.2,0.5]
nm1 = 5

k = 7
P = PADsu3.build_model(nm1=nm1)
# mus = collect(range(0.07, 0.081, length=11))
# mus = collect(range(0.14, 0.15, length=10))
# PADsu3.write_results(P, mus,  coef[1],coef[2],coef[3],k,joinpath(file_path, "results_$(nm1).jld2"))
μc, Q, factor = method_minimize_cost_function_su3(nm1)
@printf "μc:%.4f Q:%.4f factor:%.4f\n" μc Q factor
tool_for_generator()