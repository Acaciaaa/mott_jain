using LsqFit, Statistics

# —— 你原有的数据接口 —— 
# 必须已存在：PADsu3.read_results(nm1)  →  (mus::Vector{Float64}, results_vec::Vector{Any})
# 其中 results 是形如 [(E, L2, C2), ...]，且每个 μ 下第一个元素是基态 (E0, ...)

# 从单个 (μ, results) 提取 5 个通道的能隙（完全保留你“固定索引”的规则）
struct Gaps; dS_minus_S::Float64; J::Float64; e_dJ::Float64; dJ::Float64; T::Float64; end
function extract_gaps_from_results(results)::Gaps
    atol = 1e-8
    E0 = results[1][1]
    # S块(ℓ=0,C2=0)：[1]≈E0, [2]≈S
    Sblk = sort([st[1] for st in results if isapprox(st[2],0.0;atol=atol)&&isapprox(st[3],0.0;atol=atol)])
    @assert length(Sblk) ≥ 2 "need identity+S in (ℓ=0,C2=0)"
    ES = Sblk[2]; gS = ES - E0
    # ∂S：ℓ=1,C2=0 → 第1个
    dSblk = sort([st[1] for st in results if isapprox(st[2],2.0;atol=atol)&&isapprox(st[3],0.0;atol=atol)])
    @assert !isempty(dSblk) "no ∂S in (ℓ=1,C2=0)"
    EdS = dSblk[1]; gdS = EdS - E0
    # J：ℓ=1,C2=3 → 第1个
    Jblk = sort([st[1] for st in results if isapprox(st[2],2.0;atol=atol)&&isapprox(st[3],3.0;atol=atol)])
    @assert length(Jblk) ≥ 1 "no J in (ℓ=1,C2=3)"
    EJ = Jblk[1]; gJ = EJ - E0
    # ε∂J：同块第3个
    @assert length(Jblk) ≥ 3 "need ε∂J as 3rd in (ℓ=1,C2=3)"
    EeJ = Jblk[3]; geJ = EeJ - E0
    # ∂J：ℓ=2,C2=3 → 第1个
    dJblk = sort([st[1] for st in results if isapprox(st[2],6.0;atol=atol)&&isapprox(st[3],3.0;atol=atol)])
    @assert !isempty(dJblk) "no ∂J in (ℓ=2,C2=3)"
    EdJ = dJblk[1]; gdJ = EdJ - E0
    # T：ℓ=2,C2=0 → 第1个
    Tblk = sort([st[1] for st in results if isapprox(st[2],6.0;atol=atol)&&isapprox(st[3],0.0;atol=atol)])
    @assert !isempty(Tblk) "no T in (ℓ=2,C2=0)"
    ET = Tblk[1]; gT = ET - E0
    return Gaps(gdS - gS, gJ, geJ, gdJ, gT)  # 顺序: [∂S-S, J, ε∂J, ∂J, T]
end

# 尺寸到“长度”映射（按你模型改；先用 √nm1）
length_from_nm1(nm1::Int) = sqrt(nm1)

# 组装跨尺寸/跨 μ 的数据集
function build_dataset(nm1_list::Vector{Int})
    X = Vector{NTuple{3,Float64}}(); y = Float64[]; Lset = Float64[]
    for nm1 in nm1_list
        mus, results_vec = PADsu3.read_results(nm1)
        L = length_from_nm1(nm1); push!(Lset, L)
        for (μ, results) in zip(mus, results_vec)
            g = extract_gaps_from_results(results)
            gvec = (g.dS_minus_S, g.J, g.e_dJ, g.dJ, g.T)
            for (k, gk) in enumerate(gvec)
                push!(X, (L, μ, k)); push!(y, gk)
            end
        end
    end
    return X, y, unique(sort!(Lset))
end

# 有限尺寸标度模型（带硬约束）：α_L*g_k = Δ*_k + a_k t L^{y_t} + b_k L^{-ω}
function model(params, X)
    # 解包参数向量
    idx=1
    μc = params[idx]; idx+=1
    y_t= params[idx]; idx+=1
    ω  = params[idx]; idx+=1
    ΔS = params[idx]; idx+=1            # Δ_S^*（未知）
    a  = params[idx:idx+4]; idx+=5      # a1..a5（各通道 μ 线性系数）
    b  = params[idx:idx+4]; idx+=5      # b1..b5（各通道 1/L^ω 系数）
    # α_L（每个 L 一个）
    Ls = unique(sort!([x[1] for x in X]))
    nL = length(Ls)
    α  = params[idx:idx+nL-1]

    L_to_i = Dict(Ls[i]=>i for i in eachindex(Ls))
    yhat = similar(X, Float64)
    for (j,(L, μ, k)) in enumerate(X)
        αL = α[L_to_i[L]]
        t  = μ - μc
        # CFT 极限维度（硬约束）
        Δ⋆ = k==1 ? 1.0 : (k==2 ? 2.0 : 3.0)   # [∂S-S]=1,  J=2,  ε∂J/∂J/T=3
        # （备注：ΔS 目前只用来可视化或扩展；这里 ∂S-S 直接=1，不需显式 ΔS）
        Δpred = Δ⋆ + a[k]*t*(L^y_t) + b[k]*(L^(-ω))
        yhat[j] = Δpred / αL                  # 预测的原始能隙
    end
    return yhat
end

# 全局拟合：返回拟合对象与关键结果
function estimate_critical_point(nm1_list::Vector{Int})
    X, y, Ls = build_dataset(nm1_list)
    nL = length(Ls)
    npar = 4 + 5 + 5 + nL  # [μc,y_t,ω,ΔS] + a1..a5 + b1..b5 + α_Ls

    # 初值：μc=所有 μ 的中位数，y_t≈1, ω≈1, ΔS≈1.6, a,b≈0, α_L≈1/median(y)
    μ_all = [x[2] for x in X][1:5:end]
    p0 = zeros(npar)
    p0[1] = median(μ_all); p0[2] = 1.0; p0[3] = 1.0; p0[4] = 1.6
    α0 = 1.0 / max(median(y), eps());  p0[(4+5+5+1):end] .= α0

    # 合理边界：y_t, ω, α_L > 0
    lower = fill(-Inf, npar); upper = fill(Inf, npar)
    lower[2] = 1e-6; lower[3] = 1e-6
    lower[(4+5+5+1):end] .= 1e-8

    fit = curve_fit(model, p0, X, y; lower=lower, upper=upper)
    p̂ = coef(fit)

    μc  = p̂[1]
    y_t = p̂[2];  ν = 1/y_t
    ω   = p̂[3]
    ΔS  = p̂[4]
    return (; μc, ν, ω, ΔS, fit, Ls)
end
