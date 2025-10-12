module KLcommon
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions  # 如果需要 beta_inc 等
using CairoMakie

# —— 不要重载 ≈（Julia 已有 isapprox/≈），用常量控制容差更安全 ——
const ATOL = sqrt(eps(Float64))

# —— 公共：按 nmf 构建模型参数（不把 μ 烧进来） ——
Base.@kwdef mutable struct ModelParams
    name::Symbol
    nmf::Int
    nff::Int
    nmb::Int
    nfb::Int
    nof::Int
    nob::Int
    qnd
    qnf
    cfs
    tms_hop
    tms_int
    tms_l2
    tms_c2
end

function build_model(; nmf::Int)
    nff = 2
    nof = nff * nmf
    nmb = 2*nmf - 1
    nfb = 1
    nob = nmb

    FuzzifiED.ObsNormRadSq = nmf

    qnd = [
        SQNDiag(GetNeQNDiag(nof), nob) + 2*GetBosonNeSQNDiag(nof, nob),
        SQNDiag(GetLz2QNDiag(nmf, nff), nob) + GetBosonLz2SQNDiag(nof, nmb, nfb),
        SQNDiag(GetFlavQNDiag(nmf, nff, [1, -1]), nob)
    ]
    qnf = [
        SQNOffd(GetFlavPermQNOffd(nmf, nff, [2, 1], [1, -1]), nob),
        SQNOffd(GetRotyQNOffd(nmf, nff), nob) * GetBosonRotySQNOffd(nof, nmb, nfb)
    ]

    # 选定总扇区：Q1_tot=2*nmf, 2Lz=0, flav-U(1)=0
    cfs = SConfs(nof, nob, nmf, [2*nmf, 0, 0], qnd)

    # 哈密顿各项（不含 μ，μ 之后外部传入）
    tms_hop = GetIntegral(GetBosonSObs(nmb, nfb, 1)' *
                          GetFermionSObs(nmf, nff, 1) *
                          GetFermionSObs(nmf, nff, 2))
    den_e   = StoreComps(GetFerDensitySObs(nmf, nff) + 2*GetBosDensitySObs(nmb, nfb))
    tms_int = GetIntegral(den_e * den_e)

    # 观测算符
    tms_l2 = GetL2STerms(nmf, nff, nmb, nfb)
    tms_c2 = STerms(GetC2Terms(nmf, nff, :SU))

    return ModelParams(; name=:KL, nmf, nff, nmb, nfb, nof, nob, qnd, qnf, cfs, 
    tms_hop, tms_int, tms_l2, tms_c2)
end

# —— 把 μ 拼进哈密顿量；IM 显式表示耦合到总费米数 ——
make_hmt_terms(P::ModelParams, μ::Float64) = SimplifyTerms(
    P.tms_int - 0.5*(P.tms_hop + P.tms_hop') + μ * STerms(GetPolTerms(P.nmf, P.nff))
)

# —— 4 个扇区中找基态（返回能量最低的扇区与能量） ——
function ground_state(P::ModelParams, μ::Float64, k::Int=1)
    tms_hmt = make_hmt_terms(P, μ)
    bestE = Inf; bestst = nothing; bestbs = nothing; bestR = 0; bestZ = 0
    for Z in (1,-1), R in (-1,1)
        bs = SBasis(P.cfs, [R, Z], P.qnf)
        H  = SOperator(bs, tms_hmt)
        en, st = GetEigensystem(OpMat(H), k)
        @assert abs(imag(en[1])) ≤ 1e-12 "eig has large Im part: $(en)"
        if real(en[1]) < bestE
            bestE, bestst, bestbs, bestR, bestZ = real(en[1]), st[:,1], bs, R, Z
        end
    end
    return bestst, bestbs, bestE, bestR, bestZ
end

# ——— 在一个扇区里求前 k 个本征，并计算 <L^2>、<C2> ———
function eigs_with_obs(P::ModelParams, bs::SBasis, tms_hmt; k::Int=30)
    H    = OpMat(SOperator(bs, tms_hmt))
    en, st = GetEigensystem(H, k)     # en::Vector, st::Matrix（列=本征矢）
    @assert all(abs.(imag.(en)) .≤ 1e-12) "eig has large Im part: $(en)"
    en = real.(en)

    L2   = OpMat(SOperator(bs, P.tms_l2))
    C2   = OpMat(SOperator(bs, P.tms_c2))

    L2v = [ real(st[:,i]' * L2 * st[:,i]) for i in eachindex(en) ]
    C2v = [ real(st[:,i]' * C2 * st[:,i]) for i in eachindex(en) ]
    return en, st, L2v, C2v
end

# 4) 选出 “singlet gap”：最低的 l=0 & s=0 的激发能量差 ΔE_S #
function singlet_gap(P::ModelParams, μ::Float64; k::Int=30,
                     tolL2::Float64=√(eps(Float64)),
                     tolC2::Float64=√(eps(Float64)))
    tms_hmt = make_hmt_terms(P, μ)

    bestΔ = Inf
    best  = nothing  # (ΔE, E0, Eexc, idx, R, Z, L2, C2)

    for Z in (1, -1), R in (-1, 1)
        bs = SBasis(P.cfs, [R, Z], P.qnf)
        en, st, L2v, C2v = eigs_with_obs(P, bs, tms_hmt; k=k)
        E0 = en[1]

        # 从第2个开始（排除基态）
        for i in 2:lastindex(en)
            (L2v[i] ≤ tolL2 && C2v[i] ≤ tolC2) || continue  # 仍然只要 l=0 & s=0

            Δ = en[i] - E0

            # —— 新增：打印该扇区找到的 singlet 的 L2/C2 数值（以及能隙）——
            println("singlet @ sector (R=$(R), Z=$(Z)), idx=$(i): ",
                    "L2=$(L2v[i]), C2=$(C2v[i]), ΔE=$(Δ)")

            if Δ < bestΔ
                bestΔ = Δ
                best  = (Δ, E0, en[i], i, R, Z, L2v[i], C2v[i])
            end
            break  # 本扇区已找到最靠下的 singlet，换下一个扇区
        end
    end

    return best === nothing ? (NaN, nothing) : (bestΔ, best)
end

end # module
