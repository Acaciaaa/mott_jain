# ===================== JAINdebug_functions.jl =====================
# 对 JAINcommon 构造的算符/哈密顿量进行：代数恒等 + 数值自洽 + 群对称检查
# - 代数：厄米、对易、两种密度构造等价
# - 数值：矩阵级“非厄米度”、FH 检查、保真度
# - 对称：SU(2)/SU(3) 生成元与提升同构；全局（跨分块）不变性检查

using LinearAlgebra
using FuzzifiED
using .JAINcommon

# -------------------- 基础工具 --------------------
const _EPS = 1e-13

IsZeroTerms(t)      = isempty(SimplifyTerms(t))
is_herm_terms(t)    = IsZeroTerms(SimplifyTerms(t - adjoint(t)))
anti_terms(t)       = SimplifyTerms(t - adjoint(t))
rel_nonherm(A)      = opnorm(A - A') / max(opnorm(A), eps())

function show_top_anti_entries(A::AbstractMatrix; topk::Int=5)
    B = A - A'
    n1, n2 = size(B)
    vals = Vector{Tuple{Float64,Int,Int}}()
    @inbounds for i in 1:n1, j in 1:n2
        v = abs(B[i,j])
        v > 0 && push!(vals, (v,i,j))
    end
    isempty(vals) && (println("    (A - A') is exactly zero."); return)
    sort!(vals, by = x -> -x[1])
    println("    Top-$topk |A - A'| entries:")
    for (cnt,(v,i,j)) in enumerate(vals[1:min(topk, length(vals))])
        println("      $cnt) |Δ|=$v at ($i,$j)")
    end
end

# -------------------- Terms 层级厄米性报告 --------------------
function report_terms_hermiticity(P::JAINcommon.ModelParams)
    # 仅对“应当厄米”的项判定；tms_hop 本身不厄米，单独处理
    items = [
        (:tms_int,  P.tms_int),
        (:tms_br,   P.tms_br),
        (:tms_f123, P.tms_f123),
        (:tms_f4,   P.tms_f4),
    ]
    println("=== Terms-level hermiticity ===")
    for (nm,t) in items
        ok = is_herm_terms(t)
        println(rpad(string(nm), 10), ": ", ok ? "✅ Hermitian" : "❌ NON-Hermitian")
    end
    # tms_hop：检查 h.c. 与成对厄米
    Δ_hc   = SimplifyTerms(adjoint(P.tms_hop) - P.tms_hop')    # 库内部 adjoint vs '
    t_pair = P.tms_hop + adjoint(P.tms_hop)
    Δ_pair = SimplifyTerms(t_pair - adjoint(t_pair))
    println(rpad(":tms_hop", 10), "  self-Herm? NON (expected),  hc match: ",
            isempty(Δ_hc) ? "OK" : "MISMATCH", ",  pair Herm: ",
            isempty(Δ_pair) ? "OK" : "NON-HERM")
end

# -------------------- 分块矩阵构造 --------------------
function piece_mats(P::JAINcommon.ModelParams; μ123::Float64, Δ::Float64, U::Float64=1.0, η::Float64=-0.5, Z::Int=1, R::Int=1)
    bs = Basis(P.cfs, [Z,R], P.qnf)
    mats = Dict{Symbol,Matrix{ComplexF64}}()
    mats[:int]  = Matrix(OpMat(Operator(bs, P.tms_int)))
    mats[:hop]  = Matrix(OpMat(Operator(bs, P.tms_hop)))
    mats[:hopH] = Matrix(OpMat(Operator(bs, adjoint(P.tms_hop))))
    mats[:br]   = Matrix(OpMat(Operator(bs, P.tms_br)))
    mats[:f123] = Matrix(OpMat(Operator(bs, P.tms_f123)))
    mats[:f4]   = Matrix(OpMat(Operator(bs, P.tms_f4)))
    mats[:H]    = U*mats[:int] + η*(mats[:hop] + mats[:hopH]) + μ123*mats[:f123] + 0.0*mats[:f4] + Δ*mats[:br]
    return mats
end

# -------------------- 链路自洽（积分前后） --------------------
function check_hop_chain(nml::Int, nmh::Int; norm_r2_unified::Real=nml)
    R2 = Float64(norm_r2_unified)
    nm_vec = [nml, nml, nml, nmh]
    ψ1 = JAINcommon.GetElectronObsMixed(nm_vec, 1; norm_r2=R2)
    ψ2 = JAINcommon.GetElectronObsMixed(nm_vec, 2; norm_r2=R2)
    ψ3 = JAINcommon.GetElectronObsMixed(nm_vec, 3; norm_r2=R2)
    ψ4 = JAINcommon.GetElectronObsMixed(nm_vec, 4; norm_r2=R2)
    obs = ψ4' * ψ1 * ψ2 * ψ3
    t   = GetIntegral(obs; norm_r2=R2)
    Δt  = SimplifyTerms(t - adjoint(t))
    println("hop chain (integrated) hermiticity: ", isempty(Δt) ? "OK" : "NON-Herm")
end

function check_density_chain(nml::Int, nmh::Int; norm_r2_unified::Real=nml)
    R2 = Float64(norm_r2_unified)
    nm_vec = [nml, nml, nml, nmh]
    den_e = StoreComps(JAINcommon.GetDensityObsMixed(nm_vec, Diagonal([1,1,1,3]); norm_r2=R2))
    obs_anti = JAINcommon.adjoint(den_e) - den_e
    t_obs_anti = GetIntegral(obs_anti; norm_r2=R2)
    println("density obs Hermiticity (integrated): ", isempty(SimplifyTerms(t_obs_anti)) ? "OK" : "NON-Herm")
    t_int = GetIntegral(den_e * den_e; norm_r2=R2)
    println("tms_int Hermiticity: ", isempty(SimplifyTerms(t_int - adjoint(t_int))) ? "OK" : "NON-Herm")
end

# -------------------- 总览：逐块非厄米度与定位 --------------------
function debug_nonherm(P::JAINcommon.ModelParams; μ::Float64, Δ::Float64, topk::Int=5, U::Float64=1.0, η::Float64=-0.5)
    println("==== Terms-level check ====")
    report_terms_hermiticity(P)

    println("\n==== Rebuild hop-chain pre/post integral ====")
    check_hop_chain(P.nml, P.nmh; norm_r2_unified=P.nml)

    println("\n==== Rebuild density-chain pre/post integral ====")
    check_density_chain(P.nml, P.nmh; norm_r2_unified=P.nml)

    for Z in (1,-1), R in (-1,1)
        println("\n==== Block (Z,R)=($Z,$R) ====")
        mats = piece_mats(P; μ123=μ, Δ=Δ, U=U, η=η, Z=Z, R=R)
        for key in (:int, :hop, :hopH, :br, :f123, :f4, :H)
            A  = mats[key]
            nh = rel_nonherm(A)
            nA = opnorm(A)
            println(rpad(string(key),6), " ||A|| = ", nA, "  rel_nonHerm = ", nh)
            if nh > 1e-12
                show_top_anti_entries(A; topk=topk)
            end
        end
    end
    println("\nDone.")
end

# -------------------- FH 检查 --------------------
fh_center_diff(fE::Function, λ::Float64; h::Float64=1e-5) = (fE(λ+h) - fE(λ-h)) / (2h)
fh_one_sided(fE::Function, λ::Float64; h::Float64=1e-5, side::Symbol=:right) =
    side === :right ? (fE(λ+h) - fE(λ)) / h :
    side === :left  ? (fE(λ)   - fE(λ-h)) / h :
    error("side must be :right or :left")

function fh_check_mu123(P::JAINcommon.ModelParams; μ::Float64=0.2, Δ::Float64=0.05, h::Float64=1e-5, mode::Symbol=:center, Z::Int=1, R::Int=1)
    st, bs, _, _, _ = JAINcommon.ground_state_su2u1(P, μ, Δ)
    function E0(μx)
        H = Operator(bs, JAINcommon.make_hmt_su2u1(P; U=1.0, η=-0.5, μ123=μx, μ4=0.0, Δ=Δ))
        first(eigen(Hermitian(Matrix(OpMat(H)))).values)
    end
    dE = mode === :center ? fh_center_diff(E0, μ; h=h) :
         mode === :right  ? fh_one_sided(E0, μ; h=h, side=:right) :
         mode === :left   ? fh_one_sided(E0, μ; h=h, side=:left)  :
         error("mode must be :center/:right/:left")
    N123_op = Operator(bs, P.tms_f123)
    N123    = real(st' * N123_op * st)
    return dE, N123
end

function fh_check_delta(P::JAINcommon.ModelParams; μ::Float64=0.2, Δ::Float64=0.05, h::Float64=1e-5, mode::Symbol=:center, Z::Int=1, R::Int=1)
    st, bs, _, _, _ = JAINcommon.ground_state_su2u1(P, μ, Δ)
    function E0(Δx)
        H = Operator(bs, JAINcommon.make_hmt_su2u1(P; U=1.0, η=-0.5, μ123=μ, μ4=0.0, Δ=Δx))
        first(eigen(Hermitian(Matrix(OpMat(H)))).values)
    end
    dE = mode === :center ? fh_center_diff(E0, Δ; h=h) :
         mode === :right  ? fh_one_sided(E0, Δ; h=h, side=:right) :
         mode === :left   ? fh_one_sided(E0, Δ; h=h, side=:left)  :
         error("mode must be :center/:right/:left")
    BR_op = Operator(bs, P.tms_br)
    BR    = real(st' * BR_op * st)
    return dE, BR
end

function fh_degenerate(P::JAINcommon.ModelParams, μ::Float64, Δ::Float64; g::Int=2, wrt::Symbol=:mu123, Z::Int=1, R::Int=1)
    st, bs, _, _, _ = JAINcommon.ground_state_su2u1(P, μ, Δ)
    H = Operator(bs, JAINcommon.make_hmt_su2u1(P; U=1.0, η=-0.5, μ123=μ, μ4=0.0, Δ=Δ))
    A = Hermitian(Matrix(OpMat(H)))
    vals, vecs = eigen(A)
    V = vecs[:, 1:g]
    dH = wrt === :mu123 ? Operator(bs, P.tms_f123) :
          wrt === :Delta ? Operator(bs, P.tms_br)  :
          error("wrt must be :mu123 or :Delta")
    B = Matrix(OpMat(dH))
    return real(tr(V' * B * V)) / g
end

# -------------------- 守恒量/等价性 --------------------
function commute_check_Qe(P::JAINcommon.ModelParams; μ::Float64=0.2, Δ::Float64=0.05, Z::Int=1, R::Int=1)
    bs = Basis(P.cfs, [Z,R], P.qnf)
    H  = Operator(bs, JAINcommon.make_hmt_su2u1(P; μ123=μ, μ4=0.0, U=1.0, η=-0.5, Δ=Δ))
    Qe = Operator(bs, GetIntegral(StoreComps(JAINcommon.GetDensityObsMixed(P.nm_vec, Diagonal([1,1,1,3])))))
    A = Matrix(OpMat(H))
    B = Matrix(OpMat(Qe))
    opnorm(A*B - B*A) / max(opnorm(A), eps())
end

function density_equivalence_check(P::JAINcommon.ModelParams; Z::Int=1, R::Int=1)
    bs = Basis(P.cfs, [Z,R], P.qnf)
    A = Matrix(OpMat(Operator(bs, P.tms_f123)))
    den123     = StoreComps(JAINcommon.GetDensityObsMixed(P.nm_vec, Diagonal([1,1,1,0])))
    t_from_rho = GetIntegral(den123)
    B = Matrix(OpMat(Operator(bs, t_from_rho)))
    opnorm(A - B) / max(opnorm(A), eps())
end

# -------------------- 期望与保真度 --------------------
function show_expectations(P::JAINcommon.ModelParams; μ::Float64=0.2, Δ::Float64=0.05, Z::Int=1, R::Int=1)
    st, bs, _, _, _ = JAINcommon.ground_state_su2u1(P, μ, Δ)
    items = Dict(
        :f123 => P.tms_f123, :int => P.tms_int, :hop => P.tms_hop,
        :br   => P.tms_br,   :hopH=> adjoint(P.tms_hop), :f4  => P.tms_f4
    )
    for (k,t) in items
        val = st' * Operator(bs, t) * st
        println(rpad(string(k),5), " ⟨·⟩ = ", val, "  (imag=", imag(val), ")")
    end
end

function fidelity(u::AbstractVector{T}, v::AbstractVector{T}) where {T<:Complex}
    ϕ = angle(dot(u, v))
    abs(dot(u, v * exp(-im*ϕ)))^2
end

# -------------------- 一键跑 --------------------
"""
run_all(P; μ, Δ)

- Terms 层级厄米性与 hop/density 链路
- 矩阵级非厄米度与定位
- FH（中心差分）：∂E/∂μ123 vs ⟨tms_f123⟩，∂E/∂Δ vs ⟨tms_br⟩
- [H,Qe] 与密度等价性
- 各项期望值（虚部~0）
"""
function run_all(P::JAINcommon.ModelParams; μ::Float64=0.35, Δ::Float64=0.05, topk::Int=8)
    println(">>> run_all: μ=$μ, Δ=$Δ")
    debug_nonherm(P; μ=μ, Δ=Δ, topk=topk)

    println("\n==== FH (center) checks ====")
    dEμ, N123 = fh_check_mu123(P; μ=μ, Δ=Δ, h=1e-5, mode=:center)
    println("∂E/∂μ123 ≈ $dEμ ;  ⟨tms_f123⟩ = $N123 ;  diff = ", dEμ - N123)
    dEΔ, BR = fh_check_delta(P; μ=μ, Δ=Δ, h=1e-5, mode=:center)
    println("∂E/∂Δ    ≈ $dEΔ ;  ⟨tms_br⟩   = $BR   ;  diff = ", dEΔ - BR)

    println("\n==== Commutation & equivalence ====")
    relQ = commute_check_Qe(P; μ=μ, Δ=Δ)
    println("[H,Qe] rel-norm = ", relQ)
    relD = density_equivalence_check(P)
    println("density equivalence rel-norm = ", relD)

    println("\n==== Expectations (imag parts should be ~0) ====")
    show_expectations(P; μ=μ, Δ=Δ)

    println("\nDone run_all.")
end


# ===================== 结束 =====================
P = JAINcommon.build_model_su2u1(nml=4)


run_all(P; μ=0.35, Δ=0.05)


