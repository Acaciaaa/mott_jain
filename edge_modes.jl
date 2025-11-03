include(joinpath(@__DIR__, ".", "KLcommon.jl"))
include(joinpath(@__DIR__, ".", "JAINcommon.jl"))
include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .KLcommon
using .JAINcommon
using .PADsu3
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions
using CairoMakie


Ireg(a::Int, b::Int, x::Float64) = begin
    y = beta_inc(a, b, x, 1 - x)
    y isa Tuple ? first(y) : y
end

hemisphere_amp_single(nm::Int; x::Float64=0.5) =
    [ sqrt(Ireg(m, nm - m + 1, x)) for m in 1:nm ]

function hemisphere_alpha(P)
    if P.name == :KL
        alpha_f = Float64[]
        let αm = hemisphere_amp_single(P.nmf)
            for k in 1:P.nmf
                for _ in 1:P.nff
                    push!(alpha_f, αm[k])
                end
            end
        end
        return alpha_f, hemisphere_amp_single(P.nmb)
    elseif P.name == :JAINsu2u1
        α_light = hemisphere_amp_single(P.nml)
        α_heavy = hemisphere_amp_single(P.nmh)
        alpha_f = vcat(repeat(α_light, outer = P.nfl), α_heavy)
        return alpha_f
    elseif P.name == :PADsu3
        alpha_f = Float64[]
        let αm = hemisphere_amp_single(P.nm1)
            for k in 1:P.nm1
                for _ in 1:P.nf1
                    push!(alpha_f, αm[k])
                end
            end
        end
        append!(alpha_f, hemisphere_amp_single(P.nm0))
        return alpha_f
    end
end

function quantuminfo_a(P)
    if P.name == :KL
        qnd_a = SQNDiag[
            SQNDiag(GetNeQNDiag(P.nof), P.nob) + 2*GetBosonNeSQNDiag(P.nof, P.nob),
            SQNDiag(GetLz2QNDiag(P.nmf, P.nff), P.nob) + GetBosonLz2SQNDiag(P.nof, P.nmb, P.nfb)
            ]
        qnf_a = SQNOffd[]   # 不做离散对称分块
        Q_tot = 2*P.nmf

    elseif P.name == :JAINsu2u1
        qnd_a = QNDiag[
            QNDiag(vcat(fill(1, P.nol),fill(3, P.noh))),
            QNDiag(vcat(vcat(P.l2l, P.l2l, P.l2l), P.l2h))
            ]
        qnf_a = QNOffd[]
        Q_tot = 3*P.nml

    elseif P.name == :PADsu3
        qnd_a = P.qnd
        qnf_a = QNOffd[]
        Q_tot = 3*P.nm1
    end

    L2cap = 40   # 2Lz_A 扫描范围
    secd_lst = Vector{Vector{Vector{Int64}}}()
    for QA in 0:Q_tot
        QB = Q_tot - QA
        for lz2a in -L2cap:2:L2cap
            push!(secd_lst, [[QA, lz2a], [QB, -lz2a]])
        end
    end
    secf_lst = Vector{Vector{Vector{ComplexF64}}}([[ComplexF64[], ComplexF64[]]])
    return qnd_a, qnf_a, secd_lst, secf_lst
end

function calculate_entanglement(P, μ)
    qnd_a, qnf_a, secd_lst, secf_lst = quantuminfo_a(P)
    if P.name == :KL
        alpha_f, alpha_b = hemisphere_alpha(P)
        bestst, bestbs, bestE, bestR, bestZ = KLcommon.ground_state(P, μ)
        ent = GetEntSpec(
            bestst, bestbs, secd_lst, secf_lst;
            qnd_a=qnd_a, qnf_a=qnf_a,
            amp_ofa=ComplexF64.(alpha_f),
            amp_oba=ComplexF64.(alpha_b)
            )
    elseif P.name == :JAINsu2u1
        alpha_f = hemisphere_alpha(P)
        bestst, bestbs, bestE, bestR, bestZ = JAINcommon.ground_state_su2u1(P, μ, 0.05)
        ent = GetEntSpec(
            bestst, bestbs, secd_lst, secf_lst;
            qnd_a=qnd_a, qnf_a=qnf_a,
            amp_oa=ComplexF64.(alpha_f)
            )
    elseif P.name == :PADsu3
        alpha_f = hemisphere_alpha(P)
        bestst, bestbs, bestE, bestR, bestZ = PADsu3.ground_state(P, μ)
        ent = GetEntSpec(
            bestst, bestbs, secd_lst, secf_lst;
            qnd_a=qnd_a, qnf_a=qnf_a,
            amp_oa=ComplexF64.(alpha_f)
            )
    end

    pts, QA_sel = organize_entanglement(ent)
    return pts, QA_sel
end

function organize_entanglement(ent; QA_sel::Union{Nothing,Int}=nothing, λmin=1e-14)
    if QA_sel === nothing
        wt = Dict{Int,Float64}()
        for (key, eigs) in ent
            QA = key.secd_a[1]
            wt[QA] = get(wt, QA, 0.0) + sum(λ for λ in eigs if λ > λmin)
        end
        qs = collect(keys(wt))     
        vs = collect(values(wt))   
        QA_sel = qs[argmax(vs)]    
    end

    pts = Tuple{Int,Float64}[] 
    for (key, eigs) in ent
        key.secd_a[1] == QA_sel || continue
        lz2a = key.secd_a[2]
        lza  = lz2a ÷ 2
        for λ in eigs
            λ > λmin || continue
            push!(pts, (lza, -log(float(λ))))
        end
    end
    sort!(pts, by = x -> (x[1], x[2]))
    return pts, QA_sel
end


using Statistics 

function count_degeneracies_selected(
    pts::Vector{Tuple{Int,Float64}};
    lz_range = -15:15, ymin::Float64=0.0, ymax::Float64=10.0, tol::Float64=1e-14
)
    deg = Int[]
    for lz in lz_range
        ξs = sort([ξ for (lzv, ξ) in pts if lzv == lz && ymin < ξ < ymax])
        uniq = Float64[]
        for ξ in ξs
            if isempty(uniq) || abs(ξ - last(uniq)) > tol
                push!(uniq, ξ)
            end
        end
        println("lz=$(lz), ξs≈$(round.(uniq; digits=6))")
        push!(deg, length(uniq))
    end
    return deg, collect(lz_range)
end

function plot_edge_modes_selected(pts, QA_sel; deg=Int[], ymin=0.0, ymax=10.0)
    fig = Figure(size=(650,430))
    ax  = Axis(fig[1,1], xlabel="Lz_A", ylabel="ξ = -log λ",
               title="Entanglement (equator cut, μ=0.8, QA=$(QA_sel))")
    scatter!(ax, first.(pts), last.(pts), markersize=6)
    ylims!(ax, ymin, ymax)

    x_right = isempty(pts) ? 0 : maximum(first, pts)
    y_top   = ymax - 0.5
    if length(deg) == 0
        fig
    else
        txt = "deg(-9..-5) = [" * join(deg, ",") * "]"
        text!(ax, txt; position = Point2f(x_right, y_top), align = (:right, :top))
        fig
    end
end

#P = KLcommon.build_model(nmf=6)
#P = JAINcommon.build_model_su2u1(nml=4)
P = PADsu3.build_model(nm1=4)
pts, QA_sel = calculate_entanglement(P, 0.2)
ymin=0.0
ymax=10.0
deg, lz_used = count_degeneracies_selected(pts; lz_range = -15:15, ymin=ymin, ymax=ymax)
@info "Degeneracies for Lz=-9..-5, ξ∈($ymin,$ymax), QA=$QA_sel" lz_used deg
display(plot_edge_modes_selected(pts, QA_sel; deg=deg, ymin=ymin, ymax=ymax))
