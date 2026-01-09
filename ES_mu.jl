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
BLAS.set_num_threads(8)
if_draw = true
μlower = -0.05#-0.2
μupper = 0.2#0.2
μs = collect(range(μlower, μupper, length=25))#40 
# for μ in μs
#     @printf("%.7f\n", μ)
# end
k = 3

if if_draw
    fig = Figure(size=(650,650)) 
    ax = Axis(fig[1,1]; 
        xlabel = "μ", 
        ylabel = "ΔE_S · √nml", 
        title = "Singlet gap vs μ", 
        aspect = 1,
        ) 
    ylims!(ax, 0.0, 0.8)
    ax.xminorticks = IntervalsBetween(10)
    ax.xminorgridvisible = true
end

io = open("run.log", "a")
atexit(() -> close(io))
for nm1 in 3:5
    local P = PADsu3.build_model(nm1=nm1)
    #local P = PADsu2.build_model(nm1=nm1)
    Δs = Float64[]
    for μ in μs 
        local results = PADsu3.lowest_k_states(P, μ,0.4,1.0,0.3,k)
        #local results = PADsu2.lowest_k_states(P, μ, k)
        E0 = results[1][1] 
        # tol = √(eps(Float64)) 
        # idx_singlet = nothing 
        # for j in 2:lastindex(results) 
        #     r = results[j]
        #     if abs(r[2]) < tol && abs(r[3]) < tol 
        #         idx_singlet = j 
        #         break
        #     end
        # end 
        # ΔE = isnothing(idx_singlet) ? -1.0 : (results[idx_singlet][1] - E0) 
        #ΔE = results[2][1] - E0
        matches = filter(st -> st[2] ≈ 0 && st[3] ≈ 0, results)
        ΔE = (length(matches) ≥ 2 ? matches[2][1] : matches[1][1]) - E0
        # @printf("%.7f\n", μ)
        #@printf("%.7f\n", ΔE)
        #@printf(io, "%.7f\n", ΔE)
        flush(io)
        push!(Δs, ΔE*sqrt(P.nm1)) 
    end 

    if_draw && lines!(ax, μs, Δs; linewidth=2, label="nm1 = $nm1"); scatter!(ax, μs, Δs; markersize=10) 
end


if_draw  && axislegend(ax; position=:rt); fig
#save("/Users/ruiqi/Desktop/8.png", fig)