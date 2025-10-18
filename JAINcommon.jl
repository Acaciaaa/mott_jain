module JAINcommon

using FuzzifiED
using LinearAlgebra
using SpecialFunctions  # 如果需要 beta_inc 等
using CairoMakie
using WignerSymbols

# ===================== 数值与精度 =====================
const ATOL = sqrt(eps(Float64))  # 误差容差，避免误把≈重载
Sx = [0 1 0; 1 0 0; 0 0 0]
Sy = [0 -im 0; im  0 0; 0 0 0]
Sz = [1  0 0; 0 -1 0; 0  0 0]

# ===================== 模型参数结构体 =====================
# 物理自由度：3 个 q=1 的“轻味”费米子 ψ₁, ψ₂, ψ₃ + 1 个 q=3 的“重味”费米子 χ
# 本模块不显式引入玻色子；相互作用以总电荷密度 n_e^2 书写；
# 转化项为 ∫(χ^† ψ₁ ψ₂ ψ₃ + h.c.)；并提供 SU(3) 与 SU(2)×U(1) 两套哈密顿量构造器。
Base.@kwdef mutable struct ModelParams
    name::Symbol
    nml::Int
    nfl::Int
    nol::Int
    nmh::Int
    nfh::Int
    noh::Int
    no::Int
    l2l
    l2h
    qnd
    qnf
    cfs
    nm_vec
    tms_hop
    tms_int
    tms_br
    tms_f123
    tms_f4
    tms_V1
    tms_l2
    tms_c2
end

function make_qnf_swap_and_roty(nml::Int, nmh::Int, no::Int)
    # ---------- (A) f1 <-> f2 交换 ----------
    pi_swap = collect(1:no)               # 先恒等
    for m = 0:nml-1
        o1 = 1       + m                  # f1 的第 m 个轨道
        o2 = nml     + 1 + m              # f2 的第 m 个轨道
        pi_swap[o1], pi_swap[o2] = pi_swap[o2], pi_swap[o1]
    end
    alpha_swap = ones(ComplexF64, no) 
    for m = 0:nml-1
        o2 = nml     + 1 + m  
        alpha_swap[o2] = -1
    end

    # ---------- (B) Ry(pi): m -> -m ----------
    # 在每个 flavour 块内做“反序”置换，并给相位 (-1)^(j - m)
    # 对于整数 j（LLL 上 q 为整数），这等价于在 m 索引 m_idx=0..(n-1) 上交替 ±1：alpha = (-1)^(m_idx)
    pi_roty   = collect(1:no)             # 先恒等
    alpha_roty = ones(ComplexF64, no)

    # 轻的三味（每块 nml）
    for blk = 0:2
        base = blk*nml
        for m_idx = 0:nml-1
            o_src = base + m_idx + 1
            o_dst = base + (nml-1 - m_idx) + 1   # 反序（m -> -m）
            pi_roty[o_src] = o_dst
            # 相位：(-1)^(j - m)  等价于 (-1)^(m_idx)（当 j 整数）
            alpha_roty[o_src] = iseven(m_idx) ? (1+0im) : (-1+0im)
        end
    end

    # 重的单味（块长 nmh）
    base = 3*nml
    for m_idx = 0:nmh-1
        o_src = base + m_idx + 1
        o_dst = base + (nmh-1 - m_idx) + 1
        pi_roty[o_src] = o_dst
        alpha_roty[o_src] = iseven(m_idx) ? (1+0im) : (-1+0im)
    end

    return [
        QNOffd(pi_swap, alpha_swap),   # Z2: f1<->f2
        QNOffd(pi_roty, alpha_roty)    # Z2: Ry(pi)
    ]
end

function GetElectronObsMixed(nm_per_flavour::AbstractVector{<:Integer}, f::Int;
                             offset0::Int=0, norm_r2::Float64=FuzzifiED.ObsNormRadSq)
    @assert 1 ≤ f ≤ length(nm_per_flavour)
    nmf   = nm_per_flavour[f]                 # 本 flavour 的 nm
    base  = offset0 + sum(nm_per_flavour[1:f-1])  # 本 flavour 在全局中的 1 基起点-1（下面会+1）

    # 只在 l = s = (nmf-1)/2 这一层有分量；m2 从 -l2 到 +l2（步长 2）
    get_comp = function (l2::Int, m2::Int)
        if l2 == nmf - 1
            # m 层索引 k = 0..nmf-1，对应 m = -s + k；整数除法 OK（m2 与 l2 都是偶数）
            k = (m2 + (nmf - 1)) ÷ 2
            # 全局 1 基轨道号（按 flavour 分块连续）
            oid = base + k + 1
            return Terms(1 / √norm_r2, [0, oid])  # [0, oid] = 湮灭算符 c_{oid}
        else
            return Term[]                         # 其他 l 分量为 0
        end
    end

    return SphereObs(nmf - 1, nmf - 1, get_comp)  # s2 = l2m = nmf-1
end

function GetDensityObsMixed(nm_per_flavour::AbstractVector{<:Integer},
                            mat::Union{Nothing,AbstractMatrix{<:Number}}=nothing;
                            norm_r2::Float64=FuzzifiED.ObsNormRadSq,
                            offset0::Int=0)

    nf = length(nm_per_flavour)

    M = if mat === nothing
        Matrix{Float64}(I, nf, nf)
    else
        @assert size(mat,1) == nf && size(mat,2) == nf "size(M) must be nf×nf"
        mat
    end

    # 每个 flavour 的 ψ_f(Ω)（湮灭），与官方一致先 StoreComps 以缓存组件
    el = [ StoreComps(GetElectronObsMixed(nm_per_flavour, f;
                                          offset0=offset0, norm_r2=norm_r2))
           for f in 1:nf ]

    # 累加得到密度 SphereObs；与官方保持同样模式
    obs = SphereObs(0, 0, Dict{Tuple{Int64,Int64}, Terms}())
    @inbounds for f1 in 1:nf, f2 in 1:nf
        c = M[f1, f2]
        if !(abs(c) < 1e-13)
            obs += c * (el[f1])' * el[f2]
        end
    end
    return obs
end

function GetPolTermsMixed(nm_per_flavour::AbstractVector{<:Integer},
                              Mdiag::Union{Nothing,AbstractVector{<:Number},AbstractMatrix{<:Number}}=nothing;
                              offset0::Int=0)
    nf = length(nm_per_flavour)
    # 取对角权重向量 d
    d = Mdiag === nothing ? ones(nf) :
        isa(Mdiag, AbstractVector) ? Mdiag :
        diag(Mdiag)

    # 每个 flavour 的块起点(0-based)，再加全局偏移
    bases = offset0 .+ cumsum(vcat(0, nm_per_flavour[1:end-1]))

    tms = Term[]
    @inbounds for f in 1:nf
        c = d[f]
        abs(c) < 1e-13 && continue
        base = bases[f]
        nmf  = nm_per_flavour[f]
        for k in 0:nmf-1
            oid = base + k + 1              # 1-based 轨道号
            push!(tms, Term(c, [1, oid, 0, oid]))  # c†_{oid} c_{oid}
        end
    end
    return SimplifyTerms(tms)
end

function make_V1_terms_for_flavour(nm_per_flavour::AbstractVector{<:Integer},
                                   f::Int; V1::Float64=1.0, offset0::Int=0)

    @assert 1 ≤ f ≤ length(nm_per_flavour)
    # 该 flavour 在全局排列中的块起点（0-based）与块长度
    base = offset0 + sum(nm_per_flavour[1:f-1])
    nmf  = nm_per_flavour[f]
    @assert nmf ≥ 2 "target flavour nm must be ≥ 2"

    # 单粒子角动量：2S = nmf - 1；只取 L = 2S - 1（= nmf - 2）对应 m=1 的通道
    twoS = nmf - 1
    L    = twoS - 1
    @assert L ≥ 0

    # 把块内索引 k=0..nmf-1 ↔ 2m = 2k - 2S
    two_m_of_k(k) = 2k - twoS

    # 为了 O(n^3) 而不是 O(n^4)：先按 M（更准确地按 twoM = 2M）把 (k1,k2) 分桶
    # 每桶存放 (k1,k2, c12) 其中 c12 = <S m1, S m2 | L M>
    buckets = Dict{Int, Vector{NTuple{3,Int}}}()   # 暂存 k1,k2, twoM；系数另表
    coeffs  = Dict{Tuple{Int,Int}, Float64}()

    S_r = (twoS // 2)          # 有理数，支持半整数
    @inbounds for k1 in 0:nmf-1, k2 in 0:nmf-1
        tm1 = two_m_of_k(k1);  tm2 = two_m_of_k(k2)
        twoM = tm1 + tm2       # 选择律：m1+m2 = M
        # M 范围限制：|M| ≤ L；用 twoM 表达即 |twoM| ≤ 2L
        if abs(twoM) > 2L
            continue
        end
        # CG 系数：<S m1, S m2 | L M>
        c = WignerSymbols.clebschgordan(Float64, twoS//2, tm1//2, twoS//2, tm2//2, L, twoM//2)
        if c == 0.0
            continue
        end
        push!(get!(buckets, twoM, NTuple{3,Int}[]), (k1, k2, twoM))
        coeffs[(k1,k2)] = c
    end

    # 按桶做“同一 M”的外积，生成四算符项
    tms = Term[]
    for (_twoM, pairs) in buckets
        @inbounds for a in pairs, b in pairs
            k1, k2 = a[1], a[2]
            k3, k4 = b[1], b[2]
            # 把 1/2 因子并进来，避免用 scale
            v = 0.5 * V1 * coeffs[(k1,k2)] * coeffs[(k3,k4)]
            if v != 0.0
                o1 = base + k1 + 1
                o2 = base + k2 + 1
                o3 = base + k3 + 1
                o4 = base + k4 + 1
                push!(tms, Term(v, [1,o1, 1,o2, 0,o4, 0,o3]))
            end
        end
    end

    # 直接合并同类项
    return SimplifyTerms(tms)
end

function GetL2TermsMixed(nms::AbstractVector{<:Integer})
    @assert all(>(0), nms) "nm must be positive per flavour"

    # 先汇总“总 Lz / 总 L+”，包含所有 flavour
    tLz_all = Term[]
    tLp_all = Term[]

    base = 0  # 当前 flavour 的 0-based 块起点
    for nm in nms
        s = (nm - 1) / 2

        # —— 本块的 Lz 贡献：sum_k (k - s) n_{base+k}
        @inbounds for k in 0:nm-1
            o = base + k + 1
            push!(tLz_all, Term(k - s, [1, o, 0, o]))
        end

        # —— 本块的 L+ 贡献：sum_k √(k(nm-k)) |k><k-1|（块内）
        @inbounds for k in 1:nm-1
            o_to   = base + k + 1
            o_from = base + (k-1) + 1
            coeff  = sqrt(k * (nm - k))
            push!(tLp_all, Term(coeff, [1, o_to, 0, o_from]))
        end

        base += nm
    end

    # 合并简化后再组合 L^2（这样自动带上所有交叉项）
    tLz = SimplifyTerms(tLz_all)
    tLp = SimplifyTerms(tLp_all)
    tLm = tLp'  # Hermitian 共轭

    return SimplifyTerms(tLz*tLz - tLz + tLp*tLm)  # = (∑f Lf)^2
end

function GetC2TermsMixed(nml::Int; offset0::Int=0)
    base1 = offset0             
    base2 = offset0 + nml        
    tms = Term[]

    # 1/2 (c†_{f1,m1} c_{f2,m1})(c†_{f2,m2} c_{f1,m2})
    for m1 = 0:nml-1, m2 = 0:nml-1
        o1 = base1 + m1 + 1
        o2 = base2 + m1 + 1
        o3 = base2 + m2 + 1
        o4 = base1 + m2 + 1
        tms += Terms(0.5, [1,o1,0,o2, 1,o3,0,o4])
    end

    #  - 1/(2*Nf) * sum_fa,sum_fb n_fa n_fb ；这里 Nf=2 -> -1/4
    for fa_base in (base1, base2), fb_base in (base1, base2)
        for m1 = 0:nml-1, m2 = 0:nml-1
            o1 = fa_base + m1 + 1
            o3 = fb_base + m2 + 1
            tms -= Terms(0.25, [1,o1,0,o1, 1,o3,0,o3])
        end
    end

    return SimplifyTerms(tms)
end

# ===================== 模型搭建 =====================
function build_model_su2u1(; nml::Int)
    nfl = 3
    nol = nfl * nml
    nmh = 3*nml - 2
    nfh = 1
    noh = nmh
    no = nol+noh

    FuzzifiED.ObsNormRadSq = nml

    l2l = collect(0:nml-1) .* 2 .- (nml - 1)
    l2h = collect(0:nmh-1) .* 2 .- (nmh - 1)
    qnd = [
        QNDiag(vcat(fill(1, nol),fill(3, noh))),
        QNDiag(vcat(vcat(l2l, l2l, l2l),l2h)),
        QNDiag(vcat(fill(1, nml),fill(-1, nml),fill(0, nml),fill(0, nmh)))
    ]

    qnf = make_qnf_swap_and_roty(nml, nmh, no)

    cfs = Confs(no, [nol, 0, 0], qnd) 

    nm_vec = [nml, nml, nml, nmh]        
    ψ1 = GetElectronObsMixed(nm_vec, 1)       
    ψ2 = GetElectronObsMixed(nm_vec, 2)
    ψ3 = GetElectronObsMixed(nm_vec, 3)
    ψ4 = GetElectronObsMixed(nm_vec, 4)      
    tms_hop = GetIntegral(ψ4'*ψ1*ψ2*ψ3)

    den_e = StoreComps(GetDensityObsMixed(nm_vec, Diagonal([1, 1, 1, 3])))
    tms_int = GetIntegral(den_e * den_e)

    br_1 = StoreComps(GetDensityObsMixed(nm_vec, Diagonal([1, 0, 0, 0])))
    br_2 = StoreComps(GetDensityObsMixed(nm_vec, Diagonal([0, 1, 0, 0])))
    br_3 = StoreComps(GetDensityObsMixed(nm_vec, Diagonal([0, 0, 1, 0])))
    tms_br = GetIntegral(br_1 * br_1)+GetIntegral(br_2 * br_2)- 2 * GetIntegral(br_3 * br_3)

    #tms_br = GetPolTermsMixed(nm_vec, Diagonal([1.0, 1.0, -2.0, 0.0]))
    tms_f123 = GetPolTermsMixed(nm_vec, Diagonal([1.0, 1.0, 1.0, 0.0]))
    tms_f4 = GetPolTermsMixed(nm_vec, Diagonal([0.0, 0.0, 0.0, 1.0]))

    tms_V1 = make_V1_terms_for_flavour(nm_vec, 4; V1=1.0)

    tms_l2 = GetL2TermsMixed(nm_vec)
    tms_c2 = GetC2TermsMixed(nml)

    # for (nm, t) in [(:tms_hop, tms_hop), (:tms_int, tms_int), (:tms_br, tms_br), (:tms_f123, tms_f123), (:tms_f4, tms_f4)]
    #     Δ = SimplifyTerms(t - adjoint(t))
    #     println(lpad(string(nm),10), ": ", isempty(Δ) ? "OK" : "NON-HERM")
    # end

    return ModelParams(; name=:JAINsu2u1, nml, nfl, nol, nmh, nfh, noh, no, l2l, l2h, qnd, qnf, cfs, nm_vec,
    tms_hop, tms_int, tms_br, tms_f123, tms_f4, tms_V1, tms_l2, tms_c2)
end

make_hmt_su2u1(P::ModelParams; U::Float64=1.0, η::Float64=-0.5, μ123::Float64, μ4::Float64, Δ::Float64, V1::Float64=0.6) = SimplifyTerms(
    U * P.tms_int
    + η * (P.tms_hop + P.tms_hop')
    + μ123 * P.tms_f123 + μ4 * P.tms_f4
    + Δ * P.tms_br
    + V1 * P.tms_V1
)

function ground_state_su2u1(P::ModelParams, μ::Float64, Δ::Float64, k::Int=1;
                            check_hermiticity::Bool=true,
                            tol_rel::Float64=1e-12,
                            symmetrize::Bool=true)
    U = 1.0; η = -0.5; μ123 = μ; μ4 = 0.0; V1 = 0.8
    tms_hmt = make_hmt_su2u1(P; U=U, η=η, μ123=μ123, μ4=μ4, Δ=Δ, V1=V1)

    bestE = Inf; bestst = nothing; bestbs = nothing; bestR = 0; bestZ = 0

    for Z in (1,-1), R in (-1,1)
        bs = Basis(P.cfs, [Z, R], P.qnf)
        H  = Operator(bs, tms_hmt)

        # 关键：把 OpMat 显式转成 Matrix{ComplexF64}
        A  = Matrix(OpMat(H))

        # 可选：检测厄米程度（相对量 ||A-A'||/||A||）
        if check_hermiticity
            rel = opnorm(A - A') / max(opnorm(A), eps())
            @debug "μ=$μ (Z,R)=($Z,$R) rel_nonHerm=$rel"
            @assert rel ≤ max(tol_rel, 1e-14) "H not Hermitian enough: rel=$rel at (Z,R)=($Z,$R)"
        end

        # 数值对称化 + 用厄米求解器（稳定、特征值严格实）
        A  = symmetrize ? (A + A')/2 : A
        vals, vecs = eigen(Hermitian(A))  # 升序

        if vals[1] < bestE
            bestE, bestst, bestbs, bestR, bestZ = vals[1], vecs[:,1], bs, R, Z
        end
    end

    return bestst, bestbs, bestE, bestR, bestZ
end

# ——— 在一个扇区里求前 k 个本征，并计算 <L^2>、<C2>（Mixed 版）———
function eigs_with_obs(P::ModelParams, bs::Basis, tms_hmt; k::Int=30)
    H = OpMat(Operator(bs, tms_hmt))
    en, st = GetEigensystem(H, k)                 # 列向量为本征矢
    @assert all(abs.(imag.(en)) .≤ 1e-12) "eig has large Im part: $(en)"
    en = real.(en)

    L2 = OpMat(Operator(bs, P.tms_l2))           # 这里要求 P.tms_l2 已由 GetL2TermsMixed 构造
    C2 = OpMat(Operator(bs, P.tms_c2))           # 这里要求 P.tms_c2 已由 GetC2TermsMixed 构造

    L2v = [ real(st[:, i]' * L2 * st[:, i]) for i in eachindex(en) ]
    C2v = [ real(st[:, i]' * C2 * st[:, i]) for i in eachindex(en) ]
    return en, st, L2v, C2v
end

# ——— 选出 “singlet gap”：最低的 l=0 且 s=0 的激发能量差 ΔE_S（Mixed 版）———
function singlet_gap(P::ModelParams, μ::Float64; k::Int=30,
                     tolL2::Float64=√(eps(Float64)),
                     tolC2::Float64=√(eps(Float64)))
    # —— 与 Mixed 布局一致的哈密顿量（按你给的参数）——
    U = 1.0; η = -0.5; μ123 = μ; μ4 = 0.0; V1 = 0.8
    tms_hmt = make_hmt_su2u1(P; U=U, η=η, μ123=μ123, μ4=μ4, Δ=0.05, V1=V1)

    bestΔ = Inf
    best  = nothing   # (ΔE, E0, Eexc, idx, R, Z, L2, C2)

    best_L_only = nothing
    best_S_only = nothing

    # 结构: (val, ΔE, idx, R, Z)
    nearest_L = (Inf, Inf, 0, 0, 0)
    nearest_S = (Inf, Inf, 0, 0, 0)

    for Z in (1, -1), R in (-1, 1)
        bs = Basis(P.cfs, [R, Z], P.qnf)
        en, st, L2v, C2v = eigs_with_obs(P, bs, tms_hmt; k=k)
        E0 = en[1]

        for i in 2:lastindex(en)  # 排除基态
            # —— 仅此处改动：用绝对值比较 —— 
            Lok = (abs(L2v[i]) ≤ tolL2)
            Sok = (abs(C2v[i]) ≤ tolC2)

            Δ = en[i] - E0

            if abs(L2v[i]) < nearest_L[1]
                nearest_L = (abs(L2v[i]), Δ, i, R, Z)
            end
            if abs(C2v[i]) < nearest_S[1]
                nearest_S = (abs(C2v[i]), Δ, i, R, Z)
            end

            if Lok && Sok
                # 打印该 singlet 的 L2 / C2 / ΔE（保持你要的可视化）
                println("singlet @ (R=$(R), Z=$(Z)), idx=$(i): L2=$(L2v[i]), C2=$(C2v[i]), ΔE=$(Δ)")
                if Δ < bestΔ
                    bestΔ = Δ
                    best  = (Δ, E0, en[i], i, R, Z, L2v[i], C2v[i])
                end
                break
            elseif Lok && !Sok
                if best_L_only === nothing || Δ < best_L_only[1]
                    best_L_only = (Δ, E0, en[i], i, R, Z, L2v[i], C2v[i])
                end
            elseif !Lok && Sok
                if best_S_only === nothing || Δ < best_S_only[1]
                    best_S_only = (Δ, E0, en[i], i, R, Z, L2v[i], C2v[i])
                end
            end
        end
    end

    if best === nothing
        msg = "μ=$(μ): 没找到 l=0,s=0 态（k=$k）。"
        msg *= " 最近的 |L²|=$(nearest_L[1]): ΔE=$(nearest_L[2]) @ i=$(nearest_L[3]) (R=$(nearest_L[4]),Z=$(nearest_L[5]));"
        msg *= " 最近的 |C₂|=$(nearest_S[1]): ΔE=$(nearest_S[2]) @ i=$(nearest_S[3]) (R=$(nearest_S[4]),Z=$(nearest_S[5]))."

        if nearest_L[1] > tolL2 && nearest_S[1] ≤ tolC2
            @warn msg * " 结论：主要卡在 L 条件（|L²|>tolL2）。"
        elseif nearest_L[1] ≤ tolL2 && nearest_S[1] > tolC2
            @warn msg * " 结论：主要卡在 S 条件（|C₂|>tolC2）。"
        elseif nearest_L[1] > tolL2 && nearest_S[1] > tolC2
            @warn msg * " 结论：L 与 S 条件均未满足（阈值或算符/布局需检查）。"
        else
            @warn msg * " 结论：接近阈值但未同时满足（考虑放宽 tol 或增大 k）。"
        end

        diag = (nearest_L = nearest_L, nearest_S = nearest_S,
                best_L_only = best_L_only, best_S_only = best_S_only)
        return (NaN, nothing, diag)
    else
        diag = (nearest_L = nearest_L, nearest_S = nearest_S,
                best_L_only = best_L_only, best_S_only = best_S_only)
        return (bestΔ, best, diag)
    end
end

# function build_model_su3(; nml::Int)
#     nfl = 3
#     nol = nfl * nml
#     nmh = 3*nml - 2
#     nfh = 1
#     noh = nmh
#     no = nol+noh

#     FuzzifiED.ObsNormRadSq = nml
# end

# make_hmt_su3(P::ModelParams, μ::Float64) = SimplifyTerms(
#     P.tms_int
#     - 0.5 * (P.tms_hop + P.tms_hop')
#     + μ * STerms(GetPolTerms(P.nml, P.nfl))
# )

end # module
