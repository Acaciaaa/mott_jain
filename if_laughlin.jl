include(joinpath(@__DIR__, ".", "KLcommon.jl"))
include(joinpath(@__DIR__, ".", "JAINcommon.jl"))
using .KLcommon
using .JAINcommon
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions  # 如果需要 beta_inc 等
using CairoMakie
using WignerSymbols

function make_V3_terms_for_flavour(nm_per_flavour::AbstractVector{<:Integer},
                                   f::Int; V3::Float64=1.0, offset0::Int=0)

    @assert 1 ≤ f ≤ length(nm_per_flavour)
    # 该 flavour 在全局排列中的块起点（0-based）与块长度
    base = offset0 + sum(nm_per_flavour[1:f-1])
    nmf  = nm_per_flavour[f]
    @assert nmf ≥ 4 "m=3 通道需要 nmf ≥ 4（否则 L=2S-3<0 不存在）"

    # 单粒子角动量：2S = nmf - 1；取 L = 2S - 3（对应 m=3 的通道）
    twoS = nmf - 1
    L    = twoS - 3
    @assert L ≥ 0

    # 把块内索引 k=0..nmf-1 ↔ 2m = 2k - 2S
    two_m_of_k(k) = 2k - twoS

    # 为了 O(n^3)：按 twoM=2M 分桶
    buckets = Dict{Int, Vector{NTuple{3,Int}}}()   # 暂存 k1,k2,twoM
    coeffs  = Dict{Tuple{Int,Int}, Float64}()      # (k1,k2) => CG

    @inbounds for k1 in 0:nmf-1, k2 in 0:nmf-1
        tm1 = two_m_of_k(k1);  tm2 = two_m_of_k(k2)
        twoM = tm1 + tm2                    # 选择律：m1+m2=M
        if abs(twoM) > 2L                   # |M| ≤ L
            continue
        end
        # CG 系数：<S m1, S m2 | L M>
        c = WignerSymbols.clebschgordan(Float64, twoS//2, tm1//2,
                                                   twoS//2, tm2//2,
                                                   L,        twoM//2)
        if c == 0.0
            continue
        end
        push!(get!(buckets, twoM, NTuple{3,Int}[]), (k1, k2, twoM))
        coeffs[(k1,k2)] = c
    end

    # 同一 M 的外积，生成四算符项；把 1/2 因子并进系数里
    tms = Term[]
    for (_twoM, pairs) in buckets
        @inbounds for a in pairs, b in pairs
            k1, k2 = a[1], a[2]
            k3, k4 = b[1], b[2]
            v = 0.5 * V3 * coeffs[(k1,k2)] * coeffs[(k3,k4)]
            if v != 0.0
                o1 = base + k1 + 1
                o2 = base + k2 + 1
                o3 = base + k3 + 1
                o4 = base + k4 + 1
                push!(tms, Term(v, [1,o1, 1,o2, 0,o4, 0,o3]))
            end
        end
    end

    return SimplifyTerms(tms)
end

function check_laughlin(bestst, bestbs, P; f_heavy::Int=4, tol0::Float64=1e-10)
    ψ = bestst ./ norm(bestst)
    # tms_V3 = make_V3_terms_for_flavour(P.nm_vec, 4; V3=1.0)
    # MV1 = Matrix(OpMat(Operator(bestbs, tms_V3)))
    MV1 = Matrix(OpMat(Operator(bestbs, P.tms_V1)))

    # 2) 期望值（父哈密顿量判据）
    E_V1 = real(dot(ψ, MV1 * ψ))

    # 3) 零模重叠（更严格）
    vals, vecs = eigen(Hermitian(MV1))
    Z = findall(v -> v < tol0, vals)         # 近零本征值索引
    overlap2 = isempty(Z) ? 0.0 : sum(abs2, vecs[:,Z]' * ψ)

    return E_V1, overlap2
end


P = JAINcommon.build_model_su2u1(nml=4)
bestst, bestbs, bestE, bestR, bestZ = JAINcommon.ground_state_su2u1(P, 0.8, 0.05)
E_V1, overlap2 = check_laughlin(bestst, bestbs, P; f_heavy=4, tol0=1e-10)
println("⟨H_V1⟩ = ", E_V1)
println("overlap with ker(H_V1) = ", overlap2)

# 判定（经验阈值，可按规模放宽 1e-9~1e-8）
if (E_V1 < 1e-10) && (overlap2 > 1 - 1e-8)
    println("⇒ 是 Laughlin(1/3) ✅")
else
    println("⇒ 不是理想 Laughlin（或被其它项轻微混合）❌")
end