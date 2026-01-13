include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf
function tool_for_table()
    μc = 0.0466
    nm1 = 5
    P = PADsu3.build_model(nm1=nm1)
    results = PADsu3.lowest_k_states(P, μc,0.4,1.1,0.3, 100)
    E0 = results[1][1]
    enrg_cal = [
        filter(st -> st[2] ≈ 2 && st[3] ≈ 0, results)[1][1] - filter(st -> st[2] ≈ 0 && st[3] ≈ 0, results)[2][1], # ∂S - S
        filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[1][1] - E0, # J
        filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[3][1] - E0, # ϵ∂J
        filter(st -> st[2] ≈ 6 && st[3] ≈ 3, results)[1][1] - E0, # ∂J
        filter(st -> st[2] ≈ 6 && st[3] ≈ 0, results)[1][1] - E0 # T
    ]
    dim_cal = Float64[1, 2, 3, 3, 3]
    factor = (enrg_cal' * dim_cal) / (dim_cal' * dim_cal)
    @info factor
    wanted_states1 = Dict{Tuple{Int, Int}, Any}()
    wanted_states2 = Dict{Tuple{Int, Int}, Any}()

    for (l2, c2) in [(0,0), (2,0), (6,0)]
    #for (l2, c2) in [(0,6), (2,6), (6,6)]
        wanted_states1[(l2, c2)] = filter(st -> abs(st[2]-l2) < TOL && abs(st[3]-c2) < TOL, results)
        println("(l2, c2) = ($(l2), $(c2))")
        for s in wanted_states1[(l2, c2)]
            @printf("%.4f R%d Z%d\n", (s[1]-E0)/factor, s[5], s[6])
        end
    end

    # for (l2, c2) in [(0,3), (2,3), (6,3)]
    #     wanted_states1[(l2, c2)] = filter(st -> abs(st[2]-l2) < TOL && abs(st[3]-c2) < TOL, results)
    # end

    # for (l2, c2) in [(0,3), (2,3), (6,3)]
    #     wanted_states2[(l2, c2)] = []
    #     candidates_Z1 = []
    #     energies_Z2   = Float64[]
    #     for tmp in wanted_states1[(l2, c2)]
    #         z_val = tmp[6]
    #         if abs(z_val + 1.0) < TOL      # Z=-1 Sector
    #             push!(candidates_Z1, tmp)
    #         elseif abs(z_val - 1.0) < TOL  # Z=1 Sector
    #             push!(energies_Z2, tmp[1])   # 我们只关心 Z=1 的能量，不用存向量
    #         end
    #     end
        
    #     for tmp in candidates_Z1
    #         E_curr = tmp[1]
    #         has_partner = any(e -> abs(e - E_curr) < TOL, energies_Z2)
    #         if has_partner
    #             push!(wanted_states2[(l2, c2)], tmp)
    #         end
    #     end
    # end

    # for (l2, c2) in [(0,3), (2,3), (6,3)]
    #     println("(l2, c2) = ($(l2), $(c2))")
    #     for s in wanted_states1[(l2, c2)]
    #         @printf("%.4f\n", (s[1]-E0)/factor)
    #     end
    #     println("------------")
    #     for s in wanted_states2[(l2, c2)]
    #         @printf("%.4f\n", (s[1]-E0)/factor)
    #         #@info s[5], s[6]
    #     end
    # end
end
#tool_for_table()
using JLD2, Dates
function tool_for_fixedJ()
    TOL= √(eps(Float64))
    # nm1 = 6
    # μc = 0.0556
    nm1 = 5
    μc = 0.0466
    factor = 0.0844735
    P = PADsu3.build_model(nm1=nm1)
    results, bss = PADsu3.for_generator(P, μc,0.4,1.1,0.3, 100)
    jldsave("generator/results_bss.jld2"; results=results, bss=bss)
    results, bs0, bsp, bsm = PADsu3.for_generator_special(P, μc,0.4,1.1,0.3, 150)
    jldsave("generator/results_bss_othersector.jld2"; results=results, bs0=bs0, bsp=bsp, bsm=bsm)
    # for (l2, c2) in [(0,3), (2,3), (6,3)]
    #     tmp = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results)
    #     println("(l2, c2) = ($(l2), $(c2))")
    #     for s in wanted_states[(l2, c2)]
    #         println((s[1]-E0)/factor)
    #     end
    # end
end
tool_for_fixedJ()