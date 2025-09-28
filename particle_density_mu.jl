include(joinpath(@__DIR__, ".", "KLcommon.jl"))
include(joinpath(@__DIR__, ".", "JAINcommon.jl"))
using .KLcommon
using .JAINcommon
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions  # 如果需要 beta_inc 等
using CairoMakie

# ==== 3) 观测 <N_f> 与 平均占据 <n_f>=<N_f>/nmf ====
function avg_nf_for_mu(P, μ::Float64)
    if P.name == :KL
        bestst, bestbs, bestE, bestR, bestZ = KLcommon.ground_state(P, μ)
        op_N  = SOperator(bestbs, STerms(GetPolTerms(P.nmf, P.nff)); red_q=1)  # 如果 GetPolTerms==N_f
        N_exp = real(bestst' * op_N * bestst)
        n_avg = N_exp / P.nmf
    elseif P.name == :JAINsu2u1
        bestst, bestbs, bestE, bestR, bestZ = JAINcommon.ground_state_su2u1(P, μ, 0.05)
        #op_N  = Operator(bestbs, JAINcommon.GetPolTermsMixed(P.nm_vec, Diagonal([1.0, 1.0, 1.0, 0.0])); red_q=1)
        op_N  = Operator(bestbs, JAINcommon.GetPolTermsMixed(P.nm_vec, Diagonal([1.0, 1.0, 0.0, 0.0])); red_q=1)
        #op_N  = Operator(bestbs, JAINcommon.GetPolTermsMixed(P.nm_vec, Diagonal([0.0, 0.0, 1.0, 0.0])); red_q=1)
        N_exp = real(bestst' * op_N * bestst)
        n_avg = N_exp / P.nml
    end
    return n_avg, N_exp
end

# ==== 4) 扫 μ 并画图 ====
#P = KLcommon.build_model(nmf=4)
P = JAINcommon.build_model_su2u1(nml=4)
μlower = -2.0
μupper = 2.0
mus = collect(range(μlower, μupper, length=10))
nf_avg_list = Float64[]; Nf_list = Float64[]
for μ in mus
    nf_avg, Nf_tot = avg_nf_for_mu(P, μ)
    push!(nf_avg_list, nf_avg); push!(Nf_list, Nf_tot)
    @info "μ=$(round(μ,digits=3))  <n_f>≈$(round(nf_avg,digits=4))  <N_f>≈$(round(Nf_tot,digits=4))"
end

fig = Figure(size = (650, 650))
ax  = Axis(fig[1, 1];
    xlabel = "μ",
    ylabel = "⟨n_f⟩ per LLL orbital",
    title  = "Average fermion density vs μ",
    aspect = 1,                               # ← 轴面板变成正方形
    limits = ((μlower, μupper), (-1.0, 4.0))       # ← x,y 范围
)

lines!(ax, mus, nf_avg_list, linewidth = 2)
hlines!(ax, [0, 1, 2], color = :gray, linestyle = :dash, linewidth = 1)
fig

