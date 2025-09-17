include(joinpath(@__DIR__, ".", "KLcommon.jl"))
using .KLcommon

P = KLcommon.build_model(nmf=6)

# ==== 3) 观测 <N_f> 与 平均占据 <n_f>=<N_f>/nmf ====
function avg_nf_for_mu(μ::Float64)
    bestst, bestbs, bestE, bestR, bestZ = KLcommon.ground_state(P, μ)
    op_Nf  = SOperator(bestbs, STerms(GetPolTerms(P.nmf, P.nff)); red_q=1)  # 如果 GetPolTerms==N_f
    Nf_exp = real(bestst' * op_Nf * bestst)
    nf_avg = Nf_exp / nmf
    return nf_avg, Nf_exp
end

# ==== 4) 扫 μ 并画图 ====
mus = collect(range(-0.4, 0.6, length=10))
nf_avg_list = Float64[]; Nf_list = Float64[]
for μ in mus
    nf_avg, Nf_tot = avg_nf_for_mu(μ)
    push!(nf_avg_list, nf_avg); push!(Nf_list, Nf_tot)
    @info "μ=$(round(μ,digits=3))  <n_f>≈$(round(nf_avg,digits=4))  <N_f>≈$(round(Nf_tot,digits=4))"
end

fig = Figure(size = (650, 650))
ax  = Axis(fig[1, 1];
    xlabel = "μ",
    ylabel = "⟨n_f⟩ per LLL orbital",
    title  = "Average fermion density vs μ",
    aspect = 1,                               # ← 轴面板变成正方形
    limits = ((-0.4, 0.6), (0.0, 2.0))       # ← x,y 范围
)

lines!(ax, mus, nf_avg_list, linewidth = 2)
hlines!(ax, [0, 1, 2], color = :gray, linestyle = :dash, linewidth = 1)
fig

