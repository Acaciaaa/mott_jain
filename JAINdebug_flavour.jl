using LinearAlgebra
using FuzzifiED
using .JAINcommon

# -------------------- 基础工具 --------------------
const _EPS = 1e-13

IsZeroTerms(t)      = isempty(SimplifyTerms(t))
is_herm_terms(t)    = IsZeroTerms(SimplifyTerms(t - adjoint(t)))
anti_terms(t)       = SimplifyTerms(t - adjoint(t))
rel_nonherm(A)      = opnorm(A - A') / max(opnorm(A), eps())

# 3×3 味矩阵嵌入到 4×4（前三味=轻，最后一味=重为 0）
function _embed3to4(M3)
    M4 = zeros(eltype(M3), 4, 4)
    @views M4[1:3, 1:3] .= M3
    return M4
end

# 由 3×3 味矩阵 M3 构造 Terms：T(M) = ∫ ψ† M ψ
function flavor_generator_terms(P::JAINcommon.ModelParams, M3::AbstractMatrix)
    M4  = _embed3to4(M3)
    den = StoreComps(JAINcommon.GetDensityObsMixed(P.nm_vec, M4))
    GetIntegral(den)
end

# 在单块 (Z,R) 下取矩阵（空 Terms → 零矩阵）
function to_opmat_in_block(P::JAINcommon.ModelParams, t; Z::Int=1, R::Int=1)
    bs = Basis(P.cfs, [Z,R], P.qnf)
    if isempty(t)
        Aref = Matrix(OpMat(Operator(bs, P.tms_f123)))
        return zeros(eltype(Aref), size(Aref))
    else
        return Matrix(OpMat(Operator(bs, t)))
    end
end

# 提升同构检查：[T(M1),T(M2)] ?= T([M1,M2])
function check_lifted_commutator(P, M1, M2; Z::Int=1, R::Int=1, tol::Float64=1e-12)
    t1  = flavor_generator_terms(P, M1)
    t2  = flavor_generator_terms(P, M2)
    t12 = SimplifyTerms(t1*t2 - t2*t1)
    tM  = flavor_generator_terms(P, M1*M2 - M2*M1)

    Δ_terms = SimplifyTerms(t12 - tM)
    if isempty(Δ_terms)
        return true, 0.0
    end
    A = to_opmat_in_block(P, t12; Z=Z, R=R)
    B = to_opmat_in_block(P, tM;  Z=Z, R=R)
    rel = opnorm(A - B) / max(opnorm(B), eps())
    return rel ≤ tol, rel
end

# ========== 全局（跨分块）不变性检查 ==========
function _all_blocks(P; μ=0.2, Δ=0.0, U=1.0, η=-0.5)
    blocks = Tuple{Int,Int,Basis,Matrix{ComplexF64}}[]
    for Z in (1, -1), R in (-1, 1)
        bs = Basis(P.cfs, [Z, R], P.qnf)
        H  = Operator(bs, JAINcommon.make_hmt_su2u1(P; U=U, η=η, μ123=μ, μ4=0.0, Δ=Δ))
        push!(blocks, (Z, R, bs, Matrix(OpMat(H))))
    end
    return blocks
end

function _op_between_blocks(P, t, bs_from::Basis, bs_to::Basis)
    if isempty(t)
        A_to   = Matrix(OpMat(Operator(bs_to,   P.tms_f123)))
        A_from = Matrix(OpMat(Operator(bs_from, P.tms_f123)))
        return zeros(eltype(A_to), size(A_to,1), size(A_from,1))
    else
        return Matrix(OpMat(Operator(bs_to, bs_from, t)))
    end
end

function check_invariance(P; μ=0.2, Δ=0.0, Ms::Vector, tol=1e-12)
    blocks = _all_blocks(P; μ=μ, Δ=Δ)
    rels_per_M = Float64[]
    ok_all = true

    for M in Ms
        tM = flavor_generator_terms(P, M)
        resid_max = 0.0

        for (_,_,bs_from,H_from) in blocks, (_,_,bs_to,H_to) in blocks
            T = _op_between_blocks(P, tM, bs_from, bs_to)
            nT = opnorm(T)
            nT == 0 && continue

            m_to   = size(H_to,   1)
            m_from = size(H_from, 1)
            r, c   = size(T)

            if r == m_to && c == m_from
                A = H_to * T - T * H_from
                rel = opnorm(A) / nT
            elseif r == m_from && c == m_to
                A = H_from * T - T * H_to
                rel = opnorm(A) / nT
            else
                @warn "Shape mismatch in block operator" size_T=size(T) size_Hfrom=size(H_from) size_Hto=size(H_to)
                continue
            end

            resid_max = max(resid_max, rel)
        end

        push!(rels_per_M, resid_max)
        ok_all &= (resid_max ≤ tol)
    end

    return ok_all, rels_per_M
end

# ========== SU(2) / SU(3) flavour 生成元 ==========
function su2_generators_on12()
    σx = [0 1 0; 1 0 0; 0 0 0]
    σy = [0 -im 0; im 0 0; 0 0 0]
    σz = [1 0 0; 0 -1 0; 0 0 0]
    return [σx/2, σy/2, σz/2]
end

function su3_generators()
    λ1 = [0 1 0; 1 0 0; 0 0 0]
    λ2 = [0 -im 0; im 0 0; 0 0 0]
    λ3 = [1 0 0; 0 -1 0; 0 0 0]
    λ4 = [0 0 1; 0 0 0; 1 0 0]
    λ5 = [0 0 -im; 0 0 0; im 0 0]
    λ6 = [0 0 0; 0 0 1; 0 1 0]
    λ7 = [0 0 0; 0 0 -im; 0 im 0]
    λ8 = (1/sqrt(3.0)) * [1 0 0; 0 1 0; 0 0 -2]
    return [λ/2 for λ in (λ1,λ2,λ3,λ4,λ5,λ6,λ7,λ8)]
end

# ========== 包装函数 ==========
# 矩阵层检查（可能误判）
function check_SU2(P; μ=0.2, Δ=0.0)
    Ms = su2_generators_on12()
    okH, relsH = check_invariance(P; μ=μ, Δ=Δ, Ms=Ms)
    okAlg = all([check_lifted_commutator(P, Ms[i], Ms[j])[1]
                 for i in 1:3, j in 1:3])
    return okH, relsH, okAlg
end

# Terms 层检查（推荐）
function check_SU2_terms(P; μ=0.2, Δ=0.0)
    H = JAINcommon.make_hmt_su2u1(P; μ123=μ, μ4=0.0, Δ=Δ)
    Ms = su2_generators_on12()
    rels = Float64[]
    ok_all = true
    for M in Ms
        tM = flavor_generator_terms(P, M)
        Δ_terms = SimplifyTerms(H * tM - tM * H)
        push!(rels, isempty(Δ_terms) ? 0.0 : 1.0)
        ok_all &= isempty(Δ_terms)
    end
    return ok_all, rels
end

function check_SU3(P; μ=0.2, Δ=0.0)
    Ms = su3_generators()
    okH, relsH = check_invariance(P; μ=μ, Δ=Δ, Ms=Ms)
    pairs = [(1,2),(1,3),(2,3),(4,5),(6,7),(3,8)]
    oks = Bool[]; rels = Float64[]
    for (i,j) in pairs
        ok, rel = check_lifted_commutator(P, Ms[i], Ms[j])
        push!(oks, ok); push!(rels, rel)
    end
    okAlg = all(oks)
    return okH, relsH, okAlg, rels
end

println("Δ=0.05 → SU(3): ", check_SU3(P; μ=0.35, Δ=0.05))
println("Δ=0.05 → SU(2) (matrix层): ", check_SU2(P; μ=0.35, Δ=0.05))
println("Δ=0.05 → SU(2) (Terms层): ", check_SU2_terms(P; μ=0.35, Δ=0.05))
