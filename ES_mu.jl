include(joinpath(@__DIR__, ".", "KLcommon.jl"))
using .KLcommon

P = KLcommon.build_model(nmf=6)

#############################################
# 5) Fig.2b：ΔE_S √N_mf vs μ（单一尺寸）   #
#############################################
function plot_singlet_gap_vs_mu(mus::AbstractVector{<:Real}; k::Int=30)
    x = Float64.(mus)
    y = Float64[]
    for μ in x
        ΔE, info = KLcommon.singlet_gap(P, μ; k=k)
        if info === nothing
            @warn "μ=$(μ): 没找到 l=0,s=0 态（k太小？）"
            push!(y, NaN)
        else
            push!(y, ΔE * sqrt(nmf))
            @info "μ=$(round(μ,digits=3))  ΔE_S=$(round(ΔE,digits=6))  sector=(R=$(info[5]),Z=$(info[6]))  <L2>=$(info[7])  <C2>=$(info[8])"
        end
    end

    fig = Figure(size=(650,650))
    ax  = Axis(fig[1,1];
        xlabel = "μ",
        ylabel = "ΔE_S · √N_mf",
        title  = "Singlet gap across μ",
        aspect = 1,
        limits = ((minimum(x), maximum(x)), (0, maximum(skipmissing(y))*1.1))
    )
    lines!(ax, x, y, linewidth=2)
    fig
end

# 例：μ ∈ [-0.4, 0.6]
mus = collect(range(-0.4, 0.6, length=41))
display(plot_singlet_gap_vs_mu(mus; k=30))
