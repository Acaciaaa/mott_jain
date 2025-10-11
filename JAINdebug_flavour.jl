using LinearAlgebra
using FuzzifiED
using .JAINcommon

# -------------------- 基础工具 --------------------
const _EPS = 1e-13

IsZeroTerms(t)      = isempty(SimplifyTerms(t))
is_herm_terms(t)    = IsZeroTerms(SimplifyTerms(t - adjoint(t)))
anti_terms(t)       = SimplifyTerms(t - adjoint(t))
rel_nonherm(A)      = opnorm(A - A') / max(opnorm(A), eps())

# 3×3 味矩阵嵌入到 4×4（前三味=轻，第四味=重为 0）
function _embed3to4(M3::AbstractMatrix)
    @assert size(M3) == (3,3) "su(2)/su(3) generator must be 3×3; got $(size(M3))"
    M4 = zeros(eltype(M3), 4, 4)
    @views M4[1:3, 1:3] .= M3
    return M4
end

# 由 3×3 味矩阵 M3 构造二次量子化 Terms：T(M) = ∫ ψ† M ψ
function flavor_generator_terms(P::JAINcommon.ModelParams, M3::AbstractMatrix)
    M4  = _embed3to4(M3)
    den = StoreComps(JAINcommon.GetDensityObsMixed(P.nm_vec, M4))
    return GetIntegral(den)
end

# -------------------- 单块矩阵（保留以兼容） --------------------
function to_opmat_in_block(P::JAINcommon.ModelParams, t; Z::Int=1, R::Int=1)
    bs = Basis(P.cfs, [Z,R], P.qnf)
    if isempty(t)
        Aref = Matrix(OpMat(Operator(bs, P.tms_f123)))
        return zeros(eltype(Aref), size(Aref))
    else
        return Matrix(OpMat(Operator(bs, t)))
    end
end

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

# -------------------- 全局拼块 --------------------
function _blocks_and_H(P; μ=0.2, Δ=0.0, U=1.0, η=-0.5)
    infos = Tuple{Int,Int,Basis,Matrix{ComplexF64}}[]
    for Z in (1, -1), R in (-1, 1)
        bs = Basis(P.cfs, [Z, R], P.qnf)
        Ht = JAINcommon.make_hmt_su2u1(P; U=U, η=η, μ123=μ, μ4=0.0, Δ=Δ)
        H  = Matrix(OpMat(Operator(bs, Ht)))
        push!(infos, (Z, R, bs, H))
    end
    return infos
end

function _T_between(P, tM, bs_from::Basis, bs_to::Basis)
    if isempty(tM)
        Aref_to = Matrix(OpMat(Operator(bs_to,   P.tms_f123)))
        Aref_fr = Matrix(OpMat(Operator(bs_from, P.tms_f123)))
        return zeros(eltype(Aref_to), size(Aref_to,1), size(Aref_fr,1))
    else
        return Matrix(OpMat(Operator(bs_to, bs_from, tM)))
    end
end

function assemble_global_ops(P; μ=0.2, Δ=0.0, M3s::Vector{<:AbstractMatrix})
    infos = _blocks_and_H(P; μ=μ, Δ=Δ)         # [(Z,R,bs,H)]
    nb    = length(infos)
    dims  = [ size(infos[k][4],1) for k=1:nb ] # 各块维度
    offs  = cumsum(vcat(0, dims[1:end-1]))
    Ntot  = sum(dims)

    # 大 H：块对角
    Hbig = zeros(ComplexF64, Ntot, Ntot)
    for k in 1:nb
        H = infos[k][4]
        i0 = offs[k] + 1
        Hbig[i0:i0+dims[k]-1, i0:i0+dims[k]-1] .= H
    end

    # 针对一组生成元 M3s，返回对应的大 T 列表
    Tbig_list = Matrix{ComplexF64}[]
    for M3 in M3s
        tM = flavor_generator_terms(P, M3)
        Tbig = zeros(ComplexF64, Ntot, Ntot)

        for a in 1:nb, b in 1:nb
            bs_to   = infos[a][3]; da = dims[a]; ia = offs[a]+1
            bs_from = infos[b][3]; db = dims[b]; ib = offs[b]+1
            Tab = _T_between(P, tM, bs_from, bs_to)  # (da×db) 或 (db×da)
            r, c = size(Tab)

            if r == da && c == db
                Tbig[ ia:ia+da-1, ib:ib+db-1 ] .= Tab
            elseif r == db && c == da
                # 若返回了反向形状，则放到转置位置
                Tbig[ ib:ib+db-1, ia:ia+da-1 ] .= Tab
            elseif r == 0 || c == 0
                # 空块
            else
                @warn "Unexpected block shape for T(M)" r=r c=c da=da db=db a=a b=b
            end
        end

        push!(Tbig_list, Tbig)
    end

    return Hbig, Tbig_list
end

# -------------------- SU(2)/SU(3) 生成元（全部 3×3！） --------------------
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

# -------------------- 全局（跨块）对称性检查 --------------------
function _global_comm_residuals(P; μ=0.35, Δ=0.05, Ms::Vector{<:AbstractMatrix})
    Hbig, Tlist = assemble_global_ops(P; μ=μ, Δ=Δ, M3s=Ms)
    rels = Float64[]
    for T in Tlist
        nT = opnorm(T)
        if nT == 0
            push!(rels, 0.0)
        else
            C = Hbig*T - T*Hbig
            push!(rels, opnorm(C) / nT)
        end
    end
    return rels
end

function check_SU2(P; μ=0.35, Δ=0.05, tol=1e-12)
    Ms   = su2_generators_on12()
    rels = _global_comm_residuals(P; μ=μ, Δ=Δ, Ms=Ms)
    return all(r -> r ≤ tol, rels), rels, true
end

function check_SU3(P; μ=0.35, Δ=0.05, tol=1e-12)
    Ms   = su3_generators()
    rels = _global_comm_residuals(P; μ=μ, Δ=Δ, Ms=Ms)

    # 代数提升抽样验证（单块足够）
    pairs = [(1,2),(1,3),(2,3),(4,5),(6,7),(3,8)]
    Ms3 = su3_generators()
    oks = Bool[]; rels_alg = Float64[]
    for (i,j) in pairs
        ok, rel = check_lifted_commutator(P, Ms3[i], Ms3[j])
        push!(oks, ok); push!(rels_alg, rel)
    end
    okAlg = all(oks)

    return all(r -> r ≤ tol, rels), rels, okAlg, rels_alg
end

# -------------------- Terms 层（符号）SU(2) 检查 --------------------
function check_SU2_terms(P; μ=0.35, Δ=0.05)
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


P = JAINcommon.build_model_su2u1(nml=4)
# println("SU(2)  global: ", check_SU2(P; μ=0.35, Δ=0.05))
# println("SU(3)  global: ", check_SU3(P; μ=0.35, Δ=0.05))
# println("SU(2)  Terms : ", check_SU2_terms(P; μ=0.35, Δ=0.05))

# 把 Terms 层的 Δ_terms 投影到一个具体块 (Z,R)
function _project_commutator_to_block(P::JAINcommon.ModelParams, Δ_terms; Z::Int=1, R::Int=1)
    # Δ_terms 可能为空：直接返回 0
    isempty(Δ_terms) && return 0.0
    bs = Basis(P.cfs, [Z,R], P.qnf)
    A  = Matrix(OpMat(Operator(bs, Δ_terms)))
    return opnorm(A)
end

# 数值（投影）版：对每个 SU(2) 生成元，算 Δ_terms，再投影到四个块看范数
function check_SU2_numeric_projected(P; μ::Float64=0.35, Δ::Float64=0.05, tol::Float64=1e-12)
    Ht = JAINcommon.make_hmt_su2u1(P; μ123=μ, μ4=0.0, Δ=Δ)
    Ms = su2_generators_on12()
    res = Vector{NTuple{4,Float64}}()
    ok_all = true
    for M in Ms
        tM  = flavor_generator_terms(P, M)
        Δt  = SimplifyTerms(Ht*tM - tM*Ht)   # Terms 层对易子（可能为空）
        r11 = _project_commutator_to_block(P, Δt; Z= 1, R= 1)
        r1m = _project_commutator_to_block(P, Δt; Z= 1, R=-1)
        rm1 = _project_commutator_to_block(P, Δt; Z=-1, R= 1)
        rmm = _project_commutator_to_block(P, Δt; Z=-1, R=-1)
        push!(res, (r11, r1m, rm1, rmm))
        ok_all &= (maximum((r11, r1m, rm1, rmm)) ≤ tol)
    end
    return ok_all, res
end

ok_proj, res_proj = check_SU2_numeric_projected(P; μ=0.35, Δ=0.05)
println("SU(2) projected per-block residuals: ", res_proj, "  ok=", ok_proj)

