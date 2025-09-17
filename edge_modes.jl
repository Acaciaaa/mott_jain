include(joinpath(@__DIR__, ".", "KLcommon.jl"))
using .KLcommon

P = KLcommon.build_model(nmf=6)

# -----------------------
# 2) 赤道切割振幅 α_oa
#    文档：alpha(m) = sqrt(I_{1/2}(m, nm-m+1))
#    顺序：费米（m 外层、flavor 内层）→ 玻色（按 m）
# -----------------------
# 兼容 beta_inc 不同返回形式
Ireg(a::Int, b::Int, x::Float64) = begin
    y = beta_inc(a, b, x, 1 - x)
    y isa Tuple ? first(y) : y
end
hemisphere_amp_single(nm::Int; x::Float64=0.5) =
    [ sqrt(Ireg(m, nm - m + 1, x)) for m in 1:nm ]

# 费米 α：m 外层、flavor 内层（与 SBasis 对齐）
alpha_f = Float64[]
let αm = hemisphere_amp_single(P.nmf)
    for k in 1:P.nmf
        for _ in 1:P.nff
            push!(alpha_f, αm[k])
        end
    end
end
# 玻色 α：按 m
alpha_b = hemisphere_amp_single(P.nmb)

# -----------------------
# 3) 子系统量子数：与总扇区一致——保留 Q1_A 与 2Lz_A（不要单独 Ne/Nb）
# -----------------------
qnd_a = SQNDiag[
    SQNDiag(GetNeQNDiag(P.nof), P.nob) + 2*GetBosonNeSQNDiag(P.nof, P.nob),  # Q1_A
    SQNDiag(GetLz2QNDiag(P.nmf, P.nff), P.nob) + GetBosonLz2SQNDiag(P.nof, P.nmb, P.nfb) # 2Lz_A
]
qnf_a = SQNOffd[]   # 不做离散对称分块

const Q1_tot = 2*P.nmf
L2cap = 40   # 2Lz_A 扫描范围（偶数跨度；太大只会变慢，不更准）
secd_lst = Vector{Vector{Vector{Int64}}}()
for Q1A in 0:Q1_tot
    Q1B = Q1_tot - Q1A
    for lz2a in -L2cap:2:L2cap
        push!(secd_lst, [[Q1A, lz2a], [Q1B, -lz2a]])
    end
end
# 非对角量子数（留空）
secf_lst = Vector{Vector{Vector{ComplexF64}}}([[ComplexF64[], ComplexF64[]]])

# -----------------------
# 4) 纠缠谱（文档 7.9 的 GetEntSpec；你这版关键字是 amp_ofa / amp_oba）
# -----------------------
function entanglement_Lz_blocks(st::Vector{Float64}, bs::SBasis;
                                QA_sel::Union{Nothing,Int}=nothing, λmin=1e-14)
    ent = GetEntSpec(
        st, bs, secd_lst, secf_lst;
        qnd_a=qnd_a, qnf_a=qnf_a,
        amp_ofa=ComplexF64.(alpha_f),
        amp_oba=ComplexF64.(alpha_b)
    )

    # 如果没有指定 QA，就自动选权重最大的 QA
    if QA_sel === nothing
        wt = Dict{Int,Float64}()
        for (key, eigs) in ent
            QA = key.secd_a[1]
            wt[QA] = get(wt, QA, 0.0) + sum(λ for λ in eigs if λ > λmin)
        end
        qs = collect(keys(wt))      # 各个 QA
        vs = collect(values(wt))    # 对应权重
        QA_sel = qs[argmax(vs)]     # 选权重最大的 QA
    end


    # 仅收集该 QA 下的 (Lz_A, ξ)
    pts = Tuple{Int,Float64}[]  # (LzA, ξ)
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


using Statistics  # 需要 median

# 去重计数：同层内 |ξ_i - ξ_j| ≤ tol 视为同一态（默认 tol=1e-6）
function count_degeneracies_selected(
    pts::Vector{Tuple{Int,Float64}};
    lz_range = -9:-5, ymin::Float64=0.0, ymax::Float64=10.0, tol::Float64=1e-8
)
    deg = Int[]
    for lz in lz_range
        ξs = sort([ξ for (lzv, ξ) in pts if lzv == lz && ymin < ξ < ymax])
        # 按容差去重
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

# === 画图 + 标注 -9..-5 的计数 ===
function plot_edge_modes_selected(; ymin=0.0, ymax=10.0)
    bestst, bestbs, bestE, bestR, bestZ = KLcommon.ground_state(P, -2.0)

    # 现在返回两个值
    pts, QA_sel = entanglement_Lz_blocks(bestst, bestbs)

    deg, lz_used = count_degeneracies_selected(pts; ymin=ymin, ymax=ymax)

    @info "Degeneracies for Lz=-9..-5, ξ∈($ymin,$ymax), QA=$QA_sel" lz_used deg

    fig = Figure(size=(650,430))
    ax  = Axis(fig[1,1], xlabel="Lz_A", ylabel="ξ = -log λ",
               title="Entanglement (equator cut, μ=-2, QA=$(QA_sel))")
    scatter!(ax, first.(pts), last.(pts), markersize=6)
    ylims!(ax, ymin, ymax)

    # 标注右上角
    x_right = isempty(pts) ? 0 : maximum(first, pts)
    y_top   = ymax - 0.5
    txt = "deg(-9..-5) = [" * join(deg, ",") * "]"
    text!(ax, txt; position = Point2f(x_right, y_top), align = (:right, :top))

    fig
end

# === 运行 ===
display(plot_edge_modes_selected(ymin=0.0, ymax=10.0))
