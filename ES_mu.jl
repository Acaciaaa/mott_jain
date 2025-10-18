include(joinpath(@__DIR__, ".", "JAINcommon.jl"))
using .JAINcommon
using FuzzifiED
using LinearAlgebra
using SpecialFunctions 
using CairoMakie

function plot_singlet_gap_vs_mu(mus::AbstractVector{<:Real}; k::Int=30)
    x = Float64.(mus)
    y = Float64[]
    for μ in x
        ΔE, info = JAINcommon.singlet_gap(P, μ; k=k)
        if info === nothing
            @warn "μ=$(μ): 没找到 l=0,s=0 态（k太小？）"
            push!(y, NaN)
        else
            push!(y, ΔE * sqrt(P.nml))
            @info "μ=$(round(μ,digits=3))  ΔE_S=$(round(ΔE,digits=6))  sector=(R=$(info[5]),Z=$(info[6]))  <L2>=$(info[7])  <C2>=$(info[8])"
        end
    end

    fig = Figure(size=(650,650))
    ax  = Axis(fig[1,1];
        xlabel = "μ",
        ylabel = "ΔE_S · √nml",
        title  = "Singlet gap vs μ",
        aspect = 1,
        limits = ((minimum(x), maximum(x)), (0, maximum(skipmissing(y))*1.1))
    )
    lines!(ax, x, y, linewidth=2)
    fig
end

P = JAINcommon.build_model_su2u1(nml=4)
redirect_stdout(devnull) do
    mus = collect(range(0.0, 1.0, length=20))
    display(plot_singlet_gap_vs_mu(mus; k=200))
end