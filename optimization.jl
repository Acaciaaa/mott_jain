include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3
using FuzzifiED
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf
using JLD2, Dates
using DataFrames, CSV
using XLSX
using Optim
TOL = √(eps(Float64))
PENALTY = 1e6
Base.:≈(x::Number, y::Number) = abs(x - y) < TOL

function cost_function(results)
    E0 = results[1][1]
    u_theo = Float64[1, 1, 1, 2, 2, 3, 3, 3]
    cand_0_0 = filter(r -> r[2]≈0 && r[3]≈0, results)
    cand_2_0 = filter(r -> r[2]≈2 && r[3]≈0, results)
    cand_6_0 = filter(r -> r[2]≈6 && r[3]≈0, results)
    cand_0_3 = filter(r -> r[2]≈0 && r[3]≈3, results)
    cand_2_3 = filter(r -> r[2]≈2 && r[3]≈3, results)
    cand_6_3 = filter(r -> r[2]≈6 && r[3]≈3, results)
    cand_2_6 = filter(r -> r[2]≈2 && r[3]≈6, results)
    cand_6_6 = filter(r -> r[2]≈6 && r[3]≈6, results)
    v_num = [
        cand_2_0[1][1] - cand_0_0[2][1], # ∂S - S
        cand_6_0[2][1] - cand_2_0[1][1], # ∂∂S - ∂S
        cand_6_6[1][1] - cand_2_6[1][1], # C2=6

        cand_0_0[3][1] - cand_0_0[2][1], # □S - S
        cand_2_3[1][1] - E0, # J
        #cand_0_3[2][1] - cand_0_3[1][1], # □O - O

        cand_2_3[3][1] - E0, # ϵ∂J
        cand_6_3[1][1] - E0, # ∂J
        cand_6_0[1][1] - E0 # T
    ]
    
    norm_v2 = sum(abs2, v_num)
    norm_u2 = sum(abs2, u_theo)
    dot_uv  = dot(u_theo, v_num)
    cost = norm_u2 - (dot_uv^2) / norm_v2
    factor = (v_num' * u_theo) / (u_theo' * u_theo)
    return cost, factor
end

nm1 = 6
k = 60
FIXED_t = 0.5
P = PADsu3.build_model(nm1=nm1)

PARAMS_CONFIG = (
    #Uf  = (init = 0.5, lower = 0.0, upper = 10.0),
    Uf0 = (init = 3.0, lower = 0.0, upper = 20.0),
    Vf0 = (init = 0.5, lower = -20.0, upper = 20.0),
    #V0 = (init = 1.0, lower = -20.0, upper = 20.0),
    μ   = (init = 0.05, lower = -100.0, upper = 100.0)
)

function run_optimization()
    param_keys = keys(PARAMS_CONFIG)
    #x0 = [0.5, 3.0, 0.5, 0.05]
    x0 = [3.0, 0.5, 0.05]

    function unpack_params(p_array)
        return NamedTuple{param_keys}(Tuple(p_array))
    end

    function optim_wrapper(params_log)
        p = unpack_params(params_log)
        for (k_sym, v) in pairs(p)
            if v < PARAMS_CONFIG[k_sym].lower || v > PARAMS_CONFIG[k_sym].upper
                return PENALTY 
            end
        end
        try
            #大满贯
            #tms = PADsu3.make_tms_hmt(P, p.μ, [p.Uf, p.Uf0, 9.0*p.Uf], [0.0, p.Vf0, 1.0], FIXED_t)
            #固定小的V
            #tms = PADsu3.make_tms_hmt(P, p.μ, [0.5, p.Uf0, 4.5], [0.0, 0.3, p.V0], FIXED_t)
            #固定大的V
            tms = PADsu3.make_tms_hmt(P, p.μ, [0.5, p.Uf0, 4.5], [0.0, p.Vf0, 1.0], FIXED_t)
            results = PADsu3.basic_solution(P, tms, k)
            cost, _ = cost_function(results)
            return cost
        catch e
            if isa(e, ErrorException) || isa(e, BoundsError)
                println("Skip params: $p \n Reason: $e")
            end
            return PENALTY
        end
    end

    res = optimize(
        optim_wrapper, 
        x0,
        NelderMead(), 
        Optim.Options(g_tol=1e-7, show_trace=true, show_every = 5)
    )
    p_best_array = Optim.minimizer(res)
    best_p = unpack_params(p_best_array)
    min_cost = Optim.minimum(res)
    #大满贯
    #best_tms = PADsu3.make_tms_hmt(P, best_p.μ, [best_p.Uf, best_p.Uf0, 9.0*best_p.Uf], [0.0, best_p.Vf0, 1.0], FIXED_t)
    #固定小的V
    #best_tms = PADsu3.make_tms_hmt(P, best_p.μ, [0.5, best_p.Uf0, 4.5], [0.0, 0.3, best_p.V0], FIXED_t)
    #固定大的V
    best_tms = PADsu3.make_tms_hmt(P, best_p.μ, [0.5, best_p.Uf0, 4.5], [0.0, best_p.Vf0, 1.0], FIXED_t)
    
    
    best_results = PADsu3.basic_solution(P, best_tms, k)
    _, best_factor = cost_function(best_results)

    println("=====================================")
    println("Optimization Results for nm1: $(nm1)")
    println("Best Coeffs :")
    for k_name in param_keys
        println("              $(rpad(k_name, 5)) = $(round(best_p[k_name], digits=6))")
    end
    println("Min Cost    : $(min_cost)")
    println("Best Factor : $(best_factor)")
    println("=====================================")
end

run_optimization()