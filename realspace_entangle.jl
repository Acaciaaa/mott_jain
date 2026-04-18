include(joinpath(@__DIR__, ".", "pad_su3.jl"))
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

hemisphere_amp_single(nm::Int; x::Float64 = 0.5) =
    [sqrt(Ireg(m, nm - m + 1, x)) for m in 1:nm]


function hemisphere_alpha_PADsu3(P; x::Float64 = 0.5)
    alpha_f = Float64[]

    # charge-1
    let αm = hemisphere_amp_single(P.nm1; x = x)
        for k in 1:P.nm1
            for _ in 1:P.nf1
                push!(alpha_f, αm[k])
            end
        end
    end

    # charge-3
    append!(alpha_f, hemisphere_amp_single(P.nm0; x = x))

    return alpha_f
end


function quantuminfo_a_PADsu3(P; QA_half_window = 2, L2cap = 14)
    nm1 = P.nm1
    nf1 = P.nf1
    no1 = nm1 * nf1

    nm0 = P.nm0
    no0 = nm0

    # 只保留两个 qnd: total charge, total 2Lz
    qnd_a = QNDiag[
        PadQNDiag(GetNeQNDiag(no1), 0, no0) + 3 * PadQNDiag(GetNeQNDiag(no0), no1, 0),
        PadQNDiag(GetLz2QNDiag(nm1, nf1), 0, no0) + PadQNDiag(GetLz2QNDiag(nm0, 1), no1, 0)
    ]

    qnf_a = QNOffd[]

    Q_tot = 3 * P.nm1
    Q_mid = Q_tot ÷ 2

    QA_min = max(0, Q_mid - QA_half_window)
    QA_max = min(Q_tot, Q_mid + QA_half_window)

    secd_lst = Vector{Vector{Vector{Int64}}}()
    for QA in QA_min:QA_max
        QB = Q_tot - QA
        for lz2a in -L2cap:2:L2cap
            push!(secd_lst, [[QA, lz2a], [QB, -lz2a]])
        end
    end

    secf_lst = [[ComplexF64[], ComplexF64[]]]

    return qnd_a, qnf_a, secd_lst, secf_lst
end


function get_groundstate_PADsu3(P, μ; U0 = 0.6, V1 = 1.2, t = 0.6, nev = 3)
    res, bss = PADsu3.for_generator(P, μ, U0, V1, t, nev)

    st_g = res[1][2]
    bs_g = bss[[res[1][5], res[1][6], res[1][7]]]

    return st_g, bs_g
end


function calculate_entanglement_PADsu3(
    P, μ;
    x = 0.5,
    U0 = 0.6,
    V1 = 1.2,
    t = 0.6,
    nev = 3,
    QA_half_window = 2,
    L2cap = 14
)
    qnd_a, qnf_a, secd_lst, secf_lst = quantuminfo_a_PADsu3(
        P;
        QA_half_window = QA_half_window,
        L2cap = L2cap
    )

    alpha_f = hemisphere_alpha_PADsu3(P; x = x)
    st_g, bs_g = get_groundstate_PADsu3(P, μ; U0 = U0, V1 = V1, t = t, nev = nev)

    ent = GetEntSpec(
        st_g, bs_g, secd_lst, secf_lst;
        qnd_a = qnd_a,
        qnf_a = qnf_a,
        amp_oa = ComplexF64.(alpha_f)
    )

    return ent
end


function organize_entanglement(ent; QA_sel::Union{Nothing,Int} = nothing, λmin = 1e-14)
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


function shift_pts_by_min(pts::Vector{Tuple{Int,Float64}})
    isempty(pts) && return Tuple{Int,Float64}[]
    xi_min = minimum(last.(pts))
    return [(lz, ξ - xi_min) for (lz, ξ) in pts]
end


function count_levels_selected(
    pts_shift::Vector{Tuple{Int,Float64}};
    xi_rel_cut::Float64 = 3.0
)
    isempty(pts_shift) && return Tuple{Int,Int}[]

    lz_vals = sort(unique(first.(pts_shift)))
    lz0 = minimum(lz_vals)

    counting = Tuple{Int,Int}[]
    for lz in lz_vals
        n = count(((lzv, ξ),) -> (lzv == lz && ξ < xi_rel_cut), pts_shift)
        n > 0 || continue
        push!(counting, (lz - lz0, n))
    end

    return counting
end


function print_lowest_xi_by_dL(
    pts_shift::Vector{Tuple{Int,Float64}};
    nshow_dL = 6,
    nshow_xi = 40
)
    isempty(pts_shift) && return

    lz_vals = sort(unique(first.(pts_shift)))
    lz0 = minimum(lz_vals)

    println("lowest (ξ - ξmin) by ΔLz:")
    for (i, lz) in enumerate(lz_vals)
        i > nshow_dL && break
        xis = sort([ξ for (lzv, ξ) in pts_shift if lzv == lz])
        dL = lz - lz0
        xis_show = xis[1:min(end, nshow_xi)]
        println("ΔLz=$(dL): ", round.(xis_show; digits = 4))
    end
end


function plot_edge_modes_selected(
    pts_shift,
    QA_sel;
    counting = Tuple{Int,Int}[],
    ymax = 10.0,
    title_str = ""
)
    fig = Figure(size = (700, 450))
    ax = Axis(
        fig[1, 1],
        xlabel = "Lz_A",
        ylabel = "ξ - ξmin",
        title = isempty(title_str) ? "Real-space entanglement (QA=$(QA_sel))" : title_str
    )

    scatter!(ax, first.(pts_shift), last.(pts_shift), markersize = 7)
    ylims!(ax, 0.0, ymax)

    if !isempty(counting) && !isempty(pts_shift)
        txt = "counting = [" * join(last.(counting), ", ") * "]"
        text!(
            ax,
            txt;
            position = Point2f(maximum(first.(pts_shift)), ymax - 0.5),
            align = (:right, :top)
        )
    end

    return fig
end


function run_realspace_PADsu3(
    nm1, μ;
    x = 0.5,
    xi_rel_cut = 3.0,
    plot_ymax = 10.0,
    QA_half_window = 2,
    L2cap = 14,
    U0 = 0.6,
    V1 = 1.2,
    t = 0.6,
    nev = 3,
    savepath = nothing,
    show_lowest = true
)
    P = PADsu3.build_model(nm1 = nm1)

    ent = calculate_entanglement_PADsu3(
        P, μ;
        x = x,
        U0 = U0,
        V1 = V1,
        t = t,
        nev = nev,
        QA_half_window = QA_half_window,
        L2cap = L2cap
    )

    pts, QA_sel = organize_entanglement(ent)
    pts_shift = shift_pts_by_min(pts)
    counting = count_levels_selected(pts_shift; xi_rel_cut = xi_rel_cut)

    println("==========================================")
    println("nm1 = ", nm1, ", μ = ", μ)
    println("dominant QA = ", QA_sel)
    println("counting (ξ - ξmin < $(xi_rel_cut)) = ", counting)

    if show_lowest
        print_lowest_xi_by_dL(pts_shift; nshow_dL = 6, nshow_xi = 40)
    end

    fig = plot_edge_modes_selected(
        pts_shift,
        QA_sel;
        counting = counting,
        ymax = plot_ymax,
        title_str = "RSES (nm1=$(nm1), μ=$(μ), QA=$(QA_sel))"
    )

    if savepath !== nothing
        save(savepath, fig)
        println("saved to: ", savepath)
    end

    return counting, QA_sel
end


nm1 = 5
counting_right, QA_right = run_realspace_PADsu3(
    nm1, 0.2;
    x = 0.5,
    xi_rel_cut = 10.0,
    plot_ymax = 25.0,
    QA_half_window = 2,
    L2cap = 20,
    savepath = "/Users/ruiqi/Desktop/rses_$(nm1)_right.png"
);

counting_left, QA_left = run_realspace_PADsu3(
    nm1, -0.15;
    x = 0.5,
    xi_rel_cut = 10.0,
    plot_ymax = 25.0,
    QA_half_window = 2,
    L2cap = 20,
    savepath = "/Users/ruiqi/Desktop/rses_$(nm1)_left.png"
);