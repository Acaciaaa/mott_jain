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
    tms_f123
    tms_f4
    # tms_l2
    # tms_c2
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

"""
    GetElectronObsMixed(nm_per_flavour::AbstractVector{<:Integer}, f::Int;
                        offset0::Int=0, norm_r2::Float64=ObsNormRadSq) :: SphereObs

按“每个 flavour 连续分块”的全局编号约定，返回第 `f` 个 flavour 的
电子**湮灭**算符 ψ_f(Ω) 的球面展开（SphereObs）。

- `nm_per_flavour`：各 flavour 的 LLL 轨道数向量，如 [nml, nml, nml, nmh]
- `f`：flavour 下标（1 开始）
- `offset0`：这个电子扇区在“全系统”中的起始全局偏移（1 基索引的前缀长度，默认接在最前）
- `norm_r2`：R^2 归一化（与包一致；默认为 `ObsNormRadSq`）

注意：产生算符用转置 `'` 获得，例如 `GetElectronObsMixed(..., f)'`。
"""
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

"""
    GetDensityObsMixed(nm_per_flavour::AbstractVector{<:Integer},
                       mat::Union{Nothing,AbstractMatrix{<:Number}}=nothing;
                       norm_r2::Float64=ObsNormRadSq,
                       offset0::Int=0) :: SphereObs

按“每个 flavour 连续分块”的全局编号约定，返回
    n(Ω) = ∑_{f,f'} ψ_f^†(Ω) * M_{ff'} * ψ_{f'}(Ω)
的 SphereObs 表示。

- `nm_per_flavour`: 各 flavour 的 LLL 轨道数向量，如 [nml, nml, nml, nmh]
- `mat`: 味空间矩阵 M（若为 `nothing` 则用单位阵）
- `norm_r2`: R^2 归一化（与包一致；默认 `ObsNormRadSq`）
- `offset0`: 这个电子扇区在“全系统”全局编号的起始偏移（若前面有其他粒子段时使用）

注意：需要已有 `GetElectronObsMixed`（我们之前给过）。
"""
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
            obs += c * el'[f1] * el[f2]
        end
    end
    return obs
end


"""
    GetPolTermsMixed(nm_per_flavour::AbstractVector{<:Integer},
                     M::Union{Nothing,AbstractMatrix{<:Number}}=nothing;
                     fld_m_per_flavour::Union{Nothing,Vector{<:AbstractVector{<:Number}}}=nothing,
                     offset0::Int=0) :: Terms

混合 nm、按 flavour 连续分块的极化项：
    ∑_{m,f,f'} h_m^{(f)} c†_{m f} M_{ff'} c_{m f'}

- `nm_per_flavour`: 各 flavour 的 LLL 轨道数向量，如 [nml, nml, nml, nmh]
- `M`: 味矩阵（nf×nf）。默认 `I`。
- `fld_m_per_flavour`: 每个 flavour 的 m 层权重向量列表；默认各味全 1。
  每个向量长度必须等于对应的 `nm_f`。
- `offset0`: 若前面有其他粒子段（需整体平移电子段全局编号），给出它们的总长度。
"""
function GetPolTermsMixed(nm_per_flavour::AbstractVector{<:Integer},
                          M::Union{Nothing,AbstractMatrix{<:Number}}=nothing;
                          fld_m_per_flavour::Union{Nothing,Vector{<:AbstractVector{<:Number}}}=nothing,
                          offset0::Int=0)

    nf = length(nm_per_flavour)
    bases0 = cumsum(vcat(0, nm_per_flavour[1:end-1]))  # 每味块 0-based 起点
    bases  = offset0 .+ bases0                          # 加全局偏移

    # 味矩阵
    if M === nothing
        M = Matrix{Float64}(I, nf, nf)
    else
        @assert size(M,1) == nf && size(M,2) == nf "size(M) must be nf×nf"
    end

    # m 层权重
    if fld_m_per_flavour === nothing
        fld_m_per_flavour = [ones(Float64, nm_per_flavour[f]) for f in 1:nf]
    else
        @assert length(fld_m_per_flavour) == nf "need fld_m_per_flavour for each flavour"
        @inbounds for f in 1:nf
            @assert length(fld_m_per_flavour[f]) == nm_per_flavour[f] "length(fld_m[$f]) must equal nm_per_flavour[$f]"
        end
    end

    # 预存每味的 l2 = 2s = nm_f - 1
    l2 = [nm_per_flavour[f] - 1 for f in 1:nf]

    tms = Term[]
    @inbounds for f1 in 1:nf
        nm1  = nm_per_flavour[f1]
        l21  = l2[f1]
        base1 = bases[f1]
        h1   = fld_m_per_flavour[f1]

        # 枚举 f1 的每个 m 层：k1=0..nm1-1，对应 2m = -l21 + 2k1
        for k1 in 0:nm1-1
            m2val = -l21 + 2*k1
            o1 = base1 + k1 + 1  # 全局 1-based 轨道号

            # 与所有 f2 在“同一 m 层”耦合：需要该 m 层在 f2 中也存在
            for f2 in 1:nf
                coeff = M[f1, f2]
                if !(abs(coeff) > 1e-13); continue; end

                l22 = l2[f2]
                # m 层存在性的充要条件：|2m| ≤ l22 且 (2m + l22) 为偶数（可整除 2）
                if abs(m2val) <= l22 && iseven(m2val + l22)
                    k2 = (m2val + l22) ÷ 2    # f2 中的层索引
                    o2 = bases[f2] + k2 + 1
                    push!(tms, Term(coeff * h1[k1+1], [1, o1, 0, o2]))  # c†_{o1} c_{o2}
                end
            end
        end
    end
    return SimplifyTerms(tms)
end

# 便捷重载：对角 M（如 diag(a,b,c,d)）
function GetPolTermsMixed(nm_per_flavour::AbstractVector{<:Integer},
                          diagM::AbstractVector{<:Number};
                          fld_m_per_flavour::Union{Nothing,Vector{<:AbstractVector{<:Number}}}=nothing,
                          offset0::Int=0)
    @assert length(diagM) == length(nm_per_flavour) "length(diagM) must equal nf"
    return GetPolTermsMixed(nm_per_flavour, Diagonal(diagM);
                            fld_m_per_flavour=fld_m_per_flavour, offset0=offset0)
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

    shift = Dict(i => i + nol for i in 1:noh)

    nm_vec = [nml, nml, nml, nmh]              # 4 个 flavour 的 nm
    ψ1 = GetElectronObsMixed(nm_vec, 1)        # 轻 flavour 1 湮灭
    ψ2 = GetElectronObsMixed(nm_vec, 2)
    ψ3 = GetElectronObsMixed(nm_vec, 3)
    ψ4 = GetElectronObsMixed(nm_vec, 4)        # 重 flavour 湮灭
    tms_hop = GetIntegral(ψ4'*ψ1*ψ2*ψ3)

    den_e = StoreComps(GetDensityObsMixed(nm_vec, Diagonal([1, 1, 1, 3])))
    tms_int = GetIntegral(den_e * den_e)

    tms_br = GetPolTermsMixed(nm_vec, [1.0, 1.0, -2.0, 0.0])
    tms_f123 = GetPolTermsMixed(nm_vec, [1.0, 1.0, 1.0, 0.0])
    tms_f4 = GetPolTermsMixed(nm_vec, [0.0, 0.0, 0.0, 1.0])

    # tms_l2 = GetL2Terms(nml, nfl) +
    #     RelabelOrbs(GetL2Terms(nmh, nfh), shift)

    # tms_c2 = GetC2Terms(nml, nfl, [Sx, Sy, Sz])

    return ModelParams(; name=:JAINsu2u1, nml, nfl, nol, nmh, nfh, noh, no, l2l, l2h, qnd, qnf, cfs, 
    tms_hop, tms_int, tms_br, tms_f123, tms_f4) #, tms_l2, tms_c2)
end

make_hmt_su2u1(P::ModelParams; U::Float64=1.0, η::Float64=-0.5, μ123::Float64, μ4::Float64, Δ::Float64) = SimplifyTerms(
    U * P.tms_int
    + η * (P.tms_hop + P.tms_hop')
    + μ123 * P.tms_f123 + μ4 * P.tms_f4
    + Δ * P.tms_br
)

function ground_state_su2u1(P::ModelParams, μ::Float64, Δ::Float64, k::Int=1)
    tms_hmt = make_hmt_su2u1(P; U=1.0, η=-0.5, μ123=μ, μ4=0.0, Δ)
    bestE = Inf; bestst = nothing; bestbs = nothing; bestR = 0; bestZ = 0
    for Z in (1,-1), R in (-1,1)
        bs = Basis(P.cfs, [Z, R], P.qnf)
        H  = Operator(bs, tms_hmt)
        en, st = GetEigensystem(OpMat(H), k)
        #println("en=$en")
        @assert abs(imag(en[1])) ≤ 1e-12 "eig has large Im part: $(en)"
        if real(en[1]) < bestE
            bestE, bestst, bestbs, bestR, bestZ = real(en[1]), st[:,1], bs, R, Z
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
