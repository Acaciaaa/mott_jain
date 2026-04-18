using LinearAlgebra
using FuzzifiED
using Plots
include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3


function choose_sumsets(vals::Vector{Int}, nmax::Int)
    sums = [Set{Int}() for _ in 0:nmax]
    push!(sums[1], 0)
    for v in vals
        for n = nmax-1:-1:0
            for old in sums[n+1]
                push!(sums[n+2], old + v)
            end
        end
    end
    return sums
end


function enumerate_diag_sectors(vals1::Vector{Int}, vals0::Vector{Int})
    nm1_sub = length(vals1)
    nm0_sub = length(vals0)

    sums1 = choose_sumsets(vals1, nm1_sub)
    sums0 = choose_sumsets(vals0, nm0_sub)

    secs = Set{NTuple{5,Int}}()

    for n1_1 in 0:nm1_sub, n1_2 in 0:nm1_sub, n1_3 in 0:nm1_sub, n0 in 0:nm0_sub
        Q  = n1_1 + n1_2 + n1_3 + 3n0
        F3 = n1_1 - n1_2
        F8 = n1_1 + n1_2 - 2n1_3

        for lz1 in sums1[n1_1 + 1],
            lz2 in sums1[n1_2 + 1],
            lz3 in sums1[n1_3 + 1],
            lz0 in sums0[n0 + 1]

            Lz2 = lz1 + lz2 + lz3 + lz0
            push!(secs, (Q, Lz2, F3, F8, 0))
        end
    end

    return secs
end


function build_orbital_cut(P; nm1_A = P.nm1 ÷ 2, nm0_A = P.nm0 ÷ 2)
    nm1 = P.nm1
    nf1 = 3
    no1 = nm1 * nf1
    nm0 = P.nm0
    no0 = nm0
    no  = no1 + no0

    orbA = Int[]

    for m in 0:nm1_A-1, f in 1:nf1
        push!(orbA, m * nf1 + f)
    end

    for m in 1:nm0_A
        push!(orbA, no1 + m)
    end

    orbA_set = Set(orbA)
    orbB = [o for o in 1:no if !(o in orbA_set)]
    amp_oa = [o in orbA_set ? 1 : 0 for o in 1:no]

    return orbA, orbB, amp_oa
end


function get_oes_full(
    P,
    st_g,
    bs_g;
    nm1_A::Int = P.nm1 ÷ 2,
    nm0_A::Int = P.nm0 ÷ 2,
    total_charge::Int,
    total_lz2::Int,
    total_f3::Int,
    total_f8::Int,
)
    nm1 = P.nm1
    nf1 = 3
    no1 = nm1 * nf1
    nm0 = P.nm0
    no0 = nm0
    no  = no1 + no0

    orbA, orbB, amp_oa = build_orbital_cut(P; nm1_A=nm1_A, nm0_A=nm0_A)

    qnd_a = [P.qnd; GetPinOrbQNDiag(no, orbB)]
    qnd_b = [P.qnd; GetPinOrbQNDiag(no, orbA)]

    vals1_all = collect(-(nm1 - 1):2:(nm1 - 1))
    vals0_all = collect(-(nm0 - 1):2:(nm0 - 1))

    vals1_A = vals1_all[1:nm1_A]
    vals1_B = vals1_all[nm1_A+1:end]

    vals0_A = vals0_all[1:nm0_A]
    vals0_B = vals0_all[nm0_A+1:end]

    secsA = enumerate_diag_sectors(vals1_A, vals0_A)
    secsB = enumerate_diag_sectors(vals1_B, vals0_B)

    secd_lst = Vector{Vector{Int64}}[]
    for a in secsA
        b = (
            total_charge - a[1],
            total_lz2    - a[2],
            total_f3     - a[3],
            total_f8     - a[4],
            0
        )
        if b in secsB
            push!(secd_lst, [Int64[a...], Int64[b...]])
        end
    end

    sort!(secd_lst, by = x -> (x[1][1], x[1][2], x[1][3], x[1][4]))

    secf_lst = [[Int[], Int[]]]

    ent_spec = GetEntSpec(
        st_g, bs_g, secd_lst, secf_lst;
        qnd_a = qnd_a,
        qnd_b = qnd_b,
        qnf_a = QNOffd[],
        amp_oa = amp_oa,
    )

    eig_rho = vcat(values(ent_spec)...)
    tr_rho = sum(eig_rho)
    ent_entropy = -sum(p > 0 ? p * log(p) : 0.0 for p in eig_rho)

    return (
        ent_spec = ent_spec,
        eig_rho = eig_rho,
        tr_rho = tr_rho,
        ent_entropy = ent_entropy,
        secd_lst = secd_lst,
        orbA = orbA,
        orbB = orbB,
        amp_oa = amp_oa,
    )
end


function weight_by_QA(ent_spec)
    w = Dict{Int, Float64}()
    for (sec, vals) in ent_spec
        QA = sec.secd_a[1]
        w[QA] = get(w, QA, 0.0) + sum(vals)
    end
    return sort(collect(w), by = x -> x[1])
end


function weight_by_QA_F3_F8(ent_spec)
    w = Dict{NTuple{3,Int}, Float64}()
    for (sec, vals) in ent_spec
        key = (sec.secd_a[1], sec.secd_a[3], sec.secd_a[4])
        w[key] = get(w, key, 0.0) + sum(vals)
    end
    return sort(collect(w), by = x -> x[1])
end


function dominant_QA(ent_spec)
    w = weight_by_QA(ent_spec)
    isempty(w) && error("No QA sectors found.")
    idx = argmax(last.(w))
    return w[idx][1]
end


# 第一行存整数 Lz2A，第二行存 ξ
function extract_oes_QA(ent_spec; QA::Int)
    data = Vector{Tuple{Int,Float64}}()

    for (sec, vals) in ent_spec
        if sec.secd_a[1] == QA
            Lz2A = sec.secd_a[2]
            for val in vals
                if val > 0
                    push!(data, (Lz2A, -log(val)))
                end
            end
        end
    end

    isempty(data) && return zeros(2, 0)

    sort!(data, by = x -> (x[1], x[2]))

    out = zeros(2, length(data))
    for i in eachindex(data)
        out[1, i] = data[i][1]
        out[2, i] = data[i][2]
    end
    return out
end


function plot_oes(spec; title_str = "OES", ms = 3)
    size(spec, 2) == 0 && error("Empty spec.")
    plt = Plots.scatter(
        spec[1, :] ./ 2,
        spec[2, :];
        ms = ms,
        label = "",
        xlabel = "Lz_A",
        ylabel = "ξ = -log(λ)",
        title = title_str,
        ylim = (0, 40)
    )
    return plt
end


function count_low_branch(spec::AbstractMatrix; xi_cut::Float64)
    size(spec, 2) == 0 && return Tuple{Float64,Int}[]

    lz2_all = sort(unique(Int.(round.(spec[1, :]))))

    groups = Dict{Int, Int}()
    for Lz2 in lz2_all
        mask = (Int.(round.(spec[1, :])) .== Lz2) .& (spec[2, :] .<= xi_cut)
        groups[Lz2] = count(mask)
    end

    good_lz2 = sort([Lz2 for Lz2 in lz2_all if groups[Lz2] > 0])
    isempty(good_lz2) && return Tuple{Float64,Int}[]

    Lz2_0 = minimum(good_lz2)

    counting = Tuple{Float64,Int}[]
    for Lz2 in good_lz2
        dL = (Lz2 - Lz2_0) / 2
        push!(counting, (dL, groups[Lz2]))
    end

    return counting
end


function analyze_edge_counting(
    P,
    st_g,
    bs_g;
    nm1_A::Int = P.nm1 ÷ 2,
    nm0_A::Int = P.nm0 ÷ 2,
    total_charge::Int,
    total_lz2::Int,
    total_f3::Int,
    total_f8::Int,
    QA::Union{Nothing,Int} = nothing,
    xi_cut::Float64 = 8.0,
    make_plot::Bool = true,
)
    oes = get_oes_full(
        P, st_g, bs_g;
        nm1_A = nm1_A,
        nm0_A = nm0_A,
        total_charge = total_charge,
        total_lz2 = total_lz2,
        total_f3 = total_f3,
        total_f8 = total_f8,
    )

    QA_use = isnothing(QA) ? dominant_QA(oes.ent_spec) : QA
    spec = extract_oes_QA(oes.ent_spec; QA = QA_use)
    counting = count_low_branch(spec; xi_cut = xi_cut)

    plt = nothing
    if make_plot && size(spec, 2) > 0
        plt = plot_oes(spec; title_str = "OES at QA = $QA_use")
        hline!(plt, [xi_cut], linestyle = :dash, label = "xi_cut")
    end

    return (
        ent_spec = oes.ent_spec,
        eig_rho = oes.eig_rho,
        tr_rho = oes.tr_rho,
        ent_entropy = oes.ent_entropy,
        QA_weights = weight_by_QA(oes.ent_spec),
        QA_F3_F8_weights = weight_by_QA_F3_F8(oes.ent_spec),
        QA_used = QA_use,
        spec = spec,
        counting = counting,
        orbA = oes.orbA,
        orbB = oes.orbB,
        plot = plt,
    )
end


function run_oes_from_info(
    P,
    st_g, bs_g;
    nm1_A::Int = P.nm1 ÷ 2,
    nm0_A::Int = P.nm0 ÷ 2,
    total_charge::Int = 3 * P.nm1,
    total_lz2::Int = 0,
    total_f3::Int = 0,
    total_f8::Int = 0,
    QA::Union{Nothing,Int} = nothing,
    xi_cut::Float64 = 8.0,
    make_plot::Bool = true,
)
    return analyze_edge_counting(
        P,
        st_g, bs_g,
        nm1_A = nm1_A,
        nm0_A = nm0_A,
        total_charge = total_charge,
        total_lz2 = total_lz2,
        total_f3 = total_f3,
        total_f8 = total_f8,
        QA = QA,
        xi_cut = xi_cut,
        make_plot = make_plot,
    )
end


function run_left_right(nm1, side)
    P = PADsu3.build_model(nm1=nm1)

    # μ = startswith(side, "left") ? -0.15 : 0.2
    # res0, bss = PADsu3.for_generator(P, μ, 0.6, 1.2, 0.6, 3)

    μ = startswith(side, "left") ? -1.7 : -1.2
    res0, bss = PADsu3.for_generator(P, μ, 0.65, 10.0, 0.5, 3)

    st_g = res0[1][2]
    bs_g = bss[[res0[1][5], res0[1][6], res0[1][7]]]

    nm1_A = P.nm1 ÷ 2
    nm0_A = P.nm0 ÷ 2
    total_charge = 3 * P.nm1
    total_lz2    = 0
    total_f3     = 0
    total_f8     = 0
    QA_pick = nothing
    xi_cut = 10.0

    res = run_oes_from_info(
        P,
        st_g, bs_g;
        nm1_A = nm1_A,
        nm0_A = nm0_A,
        total_charge = total_charge,
        total_lz2 = total_lz2,
        total_f3 = total_f3,
        total_f8 = total_f8,
        QA = QA_pick,
        xi_cut = xi_cut,
        make_plot = true,
    )

    println("==========================================")
    println(side)
    println("μ = ", μ)
    println("trace(rho_A) = ", res.tr_rho)
    println("entanglement entropy = ", res.ent_entropy)
    println("dominant QA used = ", res.QA_used)
    println("QA weights = ", res.QA_weights)
    println("counting = ", res.counting)

    savefig(res.plot, "/Users/ruiqi/Desktop/oes_$(nm1)_$(side).png")
end

run_left_right(6, "left1")
run_left_right(6, "right1")