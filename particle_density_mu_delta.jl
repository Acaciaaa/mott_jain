using LinearAlgebra
using CairoMakie
using .JAINcommon

# 选择密度算符（直接用 P.nm_vec）
# which = :n123 -> 三轻味总密度（范围 [0,3]）
# which = :n1   -> 第一味密度（范围 [0,1]）
function make_density_op(P, bestbs; which::Symbol=:n123)
    diagw = which === :n123 ? Diagonal([1.0, 1.0, 1.0, 0.0]) :
            which === :n1   ? Diagonal([1.0, 0.0, 0.0, 0.0]) :
            error("which must be :n123 or :n1")
    opN  = Operator(bestbs, JAINcommon.GetPolTermsMixed(P.nm_vec, diagw); red_q=1)
    denom = P.nml  # 平均到每个 LLL 轨道
    return opN, denom
end

# 稳定的厄米基态求解（不依赖外部 GetEigensystem）
function ground_state_su2u1_herm(P::JAINcommon.ModelParams, μ::Float64, Δ::Float64; k::Int=2)
    U = 1.0; η = -0.5; μ123 = μ; μ4 = 0.0
    tms_hmt = JAINcommon.make_hmt_su2u1(P; U=U, η=η, μ123=μ123, μ4=μ4, Δ=Δ)

    bestE = Inf; bestst = nothing; bestbs = nothing; bestvals = nothing
    bestR = 0; bestZ = 0
    for Z in (1,-1), R in (-1,1)
        bs = Basis(P.cfs, [Z, R], P.qnf)
        H  = Operator(bs, tms_hmt)
        A  = Matrix(OpMat(H));  A = (A + A')/2
        vals, vecs = eigen(Hermitian(A))
        if vals[1] < bestE
            bestE, bestst, bestbs, bestvals, bestR, bestZ =
                vals[1], vecs[:,1], bs, vals, R, Z
        end
    end
    return bestst, bestbs, bestE, bestvals, bestR, bestZ
end

# 单点 (μ,Δ) 的密度与能隙
function avg_density_at(P, μ::Float64, Δ::Float64; which::Symbol=:n123)
    bestst, bestbs, bestE, bestvals, bestR, bestZ = ground_state_su2u1_herm(P, μ, Δ; k=2)
    opN, denom = make_density_op(P, bestbs; which=which)
    Nexp = real(bestst' * opN * bestst)
    navg = Nexp / denom
    gap  = length(bestvals) ≥ 2 ? (bestvals[2] - bestvals[1]) : NaN
    return navg, Nexp, bestE, gap
end

# 扫描 (μ,Δ)
function sweep_mu_delta(P; mu_range=-2:0.1:2, delta_range=0.0:0.05:0.5, which::Symbol=:n123)
    nμ, nΔ = length(mu_range), length(delta_range)
    navg = fill(NaN, nμ, nΔ)
    E0   = fill(NaN, nμ, nΔ)
    gap  = fill(NaN, nμ, nΔ)
    for (i, μ) in enumerate(mu_range), (j, Δ) in enumerate(delta_range)
        try
            n, Ntot, E, g = avg_density_at(P, μ, Δ; which=which)
            navg[i,j] = n;  E0[i,j] = E;  gap[i,j] = g
        catch err
            @warn "fail at μ=$μ, Δ=$Δ : $err"
        end
    end
    return navg, E0, gap
end

# 热图（⟨n⟩ 与 |∂⟨n⟩/∂μ|）
function plot_density_phase_map(mu_range, delta_range, navg; which::Symbol=:n123)
    fig = Figure(resolution=(1100,420))
    ax1 = Axis(fig[1,1], title="⟨n⟩(μ,Δ) ($(String(which)))", xlabel="μ", ylabel="Δ")
    hm1 = heatmap!(ax1, collect(mu_range), collect(delta_range), navg'; interpolate=false)
    Colorbar(fig[1,2], hm1, label="⟨n⟩ per LLL orbital")

    Dμ = similar(navg); Dμ .= NaN
    for j in axes(navg,2), i in 2:length(mu_range)-1
        if isfinite(navg[i+1,j]) && isfinite(navg[i-1,j])
            Dμ[i,j] = 0.5*(navg[i+1,j] - navg[i-1,j]) /
                      (float(mu_range[i+1]) - float(mu_range[i-1]))
        end
    end
    ax2 = Axis(fig[1,3], title="|∂⟨n⟩/∂μ|", xlabel="μ", ylabel="Δ")
    hm2 = heatmap!(ax2, collect(mu_range), collect(delta_range), abs.(Dμ)'; interpolate=false)
    Colorbar(fig[1,4], hm2, label="|∂n/∂μ|")
    return fig
end

# μ–切片
function plot_mu_cuts(mu_range, delta_range, navg; deltas_to_show=(0.0, 0.1, 0.2), which::Symbol=:n123)
    fig = Figure(resolution=(720,420))
    ax  = Axis(fig[1,1], xlabel="μ", ylabel="⟨n⟩ per LLL orbital",
               title = "μ-cuts at fixed Δ ($(String(which)))")
    for Δ in deltas_to_show
        j = argmin(abs.(delta_range .- Δ))
        lines!(ax, collect(mu_range), navg[:,j], label="Δ=$(round(delta_range[j], digits=3))")
    end
    axislegend(position=:rb)
    return fig
end

# ================= 使用示例 =================
P = JAINcommon.build_model_su2u1(nml=4)   # 你的模型；P 里已有 nm_vec

mu_range    = range(-2, 2; length=10)
delta_range = range(0.0, 0.5; length=10)

which = :n123   # 或 :n1
navg, E0, gap = sweep_mu_delta(P; mu_range=mu_range, delta_range=delta_range, which=which)

fig_map  = plot_density_phase_map(mu_range, delta_range, navg; which=which)
display(fig_map)

fig_cuts = plot_mu_cuts(mu_range, delta_range, navg; deltas_to_show=(0.0, 0.1, 0.2, 0.3), which=which)
display(fig_cuts)
