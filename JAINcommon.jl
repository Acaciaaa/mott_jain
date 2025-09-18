module JAINcommon

using FuzzifiED
using LinearAlgebra
using SpecialFunctions  # 如果需要 beta_inc 等
using CairoMakie

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
    tms_hop
    tms_int
    tms_br
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
    alpha_swap = ones(ComplexF64, no)     # 交换无需相位

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

    tms_hop = GetIntegral(
        GetElectronObs(nmh, nfh, 1)' *           # F†  （重：nmh, nfh=1, 第1味）
        GetElectronObs(nml, nfl, 1)  *           # f1  （轻：nml, nfl=3, 第1味）
        GetElectronObs(nml, nfl, 2)  *           # f2
        GetElectronObs(nml, nfl, 3)              # f3
    )

    den_e = StoreComps(GetDensityObs(nml, nfl) + 3*GetDensityObs(nmh, nfh))
    tms_int = GetIntegral(den_e * den_e)

    tms_br = GetPolTerms(nml, nfl, Matrix(Diagonal([1.0, 1.0, -2.0])))

    tms_l2 = GetL2Terms(nml, nfl) +
         RelabelOrbs(GetL2Terms(nmh, nfh), Dict(i => i + nol for i in 1:noh))

    tms_c2 = GetC2Terms(nml, nfl, [Sx, Sy, Sz])

    return ModelParams(; name=:JAINsu2u1, nml, nfl, nol, nmh, nfh, noh, no, l2l, l2h, qnd, qnf, cfs, 
    tms_hop, tms_int, tms_br, tms_l2, tms_c2)
end

make_hmt_su2u1(P::ModelParams, μ::Float64, Δ::Float64) = SimplifyTerms(
    P.tms_int
    - 0.5 * (P.tms_hop + P.tms_hop')
    + μ * GetPolTerms(P.nml, P.nfl)
    + Δ  * P.tms_br
)

function ground_state_su2u1(P::ModelParams, μ::Float64, Δ::Float64, k::Int=1)
    tms_hmt = make_hmt_su2u1(P, μ, Δ)
    bestE = Inf; bestst = nothing; bestbs = nothing; bestR = 0; bestZ = 0
    for Z in (1,-1), R in (-1,1)
        bs = Basis(P.cfs, [Z, R], P.qnf)
        H  = Operator(bs, tms_hmt)
        en, st = GetEigensystem(OpMat(H), k)
        if en[1] < bestE
            bestE, bestst, bestbs, bestR, bestZ = en[1], st[:,1], bs, R, Z
        end
    end
    return bestst, bestbs, bestE, bestR, bestZ
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
