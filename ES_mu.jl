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
μlower = 0.05#-0.2
μupper = 0.15#0.2
μs = collect(range(μlower, μupper, length=20))#40 
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
    ylims!(ax, 0.0, 1.0)
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
        local results = PADsu3.lowest_k_states(P, μ, [0.509, 1.835, 4.581],[0.2, 0.411, 1.023],0.5,k)
        E0 = results[1][1] 
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