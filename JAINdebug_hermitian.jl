using LinearAlgebra
using .JAINcommon   # 你已 include 模块后，这行才能用

# ============ 工具：Terms 级别的“非厄米检测” ============
IsZeroTerms(t) = isempty(SimplifyTerms(t))
is_herm_terms(t) = IsZeroTerms(SimplifyTerms(t - adjoint(t)))
anti_terms(t) = SimplifyTerms(t - adjoint(t))

function report_terms_hermiticity(P)
    items = [
        (:tms_hop,  P.tms_hop),
        (:tms_int,  P.tms_int),
        (:tms_br,   P.tms_br),
        (:tms_f123, P.tms_f123),
        (:tms_f4,   P.tms_f4),
    ]
    println("=== Terms-level hermiticity ===")
    for (nm,t) in items
        ok = is_herm_terms(t)
        println(rpad(string(nm), 10), ": ", ok ? "✅ Hermitian" : "❌ NON-Hermitian")
        if !ok
            Δ = anti_terms(t)
            println("    -> anti Terms not empty (t - t†)")
        end
    end
end

# ============ 工具：矩阵级别的“非厄米程度” ============
rel_nonherm(A) = opnorm(A - A') / max(opnorm(A), eps())

# 打印反厄米部分里最大的几条矩阵元，辅助定位索引问题
function show_top_anti_entries(A::AbstractMatrix; topk=5)
    B = A - A'
    n1, n2 = size(B)
    vals = Vector{Tuple{Float64,Int,Int}}()
    for i in 1:n1, j in 1:n2
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

# ============ 把每一项单独成矩阵检查 ============
function piece_mats(P::JAINcommon.ModelParams; μ123::Float64, Δ::Float64, U::Float64=1.0, η::Float64=-0.5, Z::Int=1, R::Int=1)
    bs = Basis(P.cfs, [Z,R], P.qnf)

    mats = Dict{Symbol,Matrix{ComplexF64}}()

    mats[:int]  = Matrix(OpMat(Operator(bs, P.tms_int)))
    mats[:hop]  = Matrix(OpMat(Operator(bs, P.tms_hop)))
    mats[:hopH] = Matrix(OpMat(Operator(bs, adjoint(P.tms_hop))))
    mats[:br]   = Matrix(OpMat(Operator(bs, P.tms_br)))
    mats[:f123] = Matrix(OpMat(Operator(bs, P.tms_f123)))
    mats[:f4]   = Matrix(OpMat(Operator(bs, P.tms_f4)))

    mats[:H] = U*mats[:int] + η*(mats[:hop] + mats[:hopH]) + μ123*mats[:f123] + 0.0*mats[:f4] + Δ*mats[:br]
    return mats
end

# ============ 额外：重建 tms_hop 的“前置组件”，验证积分前后 ============
# 有时候问题出在 ψ 构造/索引；这里把 hop 的 SphereObs 再构造一遍，看看积分前后是否自洽。
function check_hop_chain(nml::Int, nmh::Int; norm_r2_unified::Real=nml)
    R2 = Float64(norm_r2_unified)   # 关键：把 Int 转成 Float64
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

    # den_e = ρ_e(Ω) = ρ_light + 3 * ρ_heavy（与你 build_model_su2u1 相同）
    den_e = StoreComps(JAINcommon.GetDensityObsMixed(nm_vec, Diagonal([1,1,1,3]); norm_r2=R2))

    # 1) 观测量自身是否 Hermitian
    obs_anti = JAINcommon.adjoint(den_e) - den_e
    # 均匀积分后的反厄米部分（应为 0）
    t_obs_anti = GetIntegral(obs_anti; norm_r2=R2)
    println("density obs Hermiticity (integrated): ", isempty(SimplifyTerms(t_obs_anti)) ? "OK" : "NON-Herm")

    # 2) 相互作用项 tms_int = ∫ (den_e * den_e)
    t_int = GetIntegral(den_e * den_e; norm_r2=R2)
    println("tms_int Hermiticity: ", isempty(SimplifyTerms(t_int - adjoint(t_int))) ? "OK" : "NON-Herm")
end

# ============ 总入口：一键诊断 ============
"""
debug_nonherm(P; μ, Δ, topk=5)

逐块 (Z,R)：
- 打印每个 Terms 的“是否厄米”
- 对每项/总和生成矩阵，打印相对非厄米度
- 打印反厄米部分最大若干矩阵元，辅助定位

并额外检查 hop 链：ψ4'ψ1ψ2ψ3 -> 积分前后的自洽性。
"""
function debug_nonherm(P::JAINcommon.ModelParams; μ::Float64, Δ::Float64, topk::Int=5)
    println("==== Terms-level check ====")
    report_terms_hermiticity(P)

    println("\n==== Rebuild hop-chain pre/post integral ====")
    check_hop_chain(P.nml, P.nmh; norm_r2_unified=P.nml)

    println("\n==== Rebuild density-chain pre/post integral ====")
    check_density_chain(P.nml, P.nmh; norm_r2_unified=P.nml)

    for Z in (1,-1), R in (-1,1)
        println("\n==== Block (Z,R)=($Z,$R) ====")
        mats = piece_mats(P; μ123=μ, Δ=Δ, Z=Z, R=R)

        for key in (:int, :hop, :hopH, :br, :f123, :f4, :H)
            A = mats[key]
            rel = rel_nonherm(A)
            println(rpad(string(key),6), " rel_nonHerm = ", rel)
            if rel > 1e-12
                show_top_anti_entries(A; topk=topk)
            end
        end
    end
    println("\nDone.")
end

P = JAINcommon.build_model_su2u1(nml=4)
debug_nonherm(P; μ=0.35, Δ=0.05, topk=8)