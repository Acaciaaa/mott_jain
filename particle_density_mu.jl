include(joinpath(@__DIR__, "pad_su3.jl"))
include(joinpath(@__DIR__, "pad_su2.jl"))
using .PADsu3
using .PADsu2
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions
using CairoMakie

function avg_nf_for_mu(P, μ::Float64)
    if P.name == :KL
        bestst, bestbs, bestE, bestR, bestZ = KLcommon.ground_state(P, μ)
        op_N  = SOperator(bestbs, STerms(GetPolTerms(P.nmf, P.nff)); red_q=1)  # 如果 GetPolTerms==N_f
        N_exp = real(bestst' * op_N * bestst)
        n_avg = N_exp / P.nmf
    elseif P.name == :JAINsu2u1
        bestst, bestbs, bestE, bestR, bestZ = JAINcommon.ground_state_su2u1(P, μ, 0.05)
        op_N  = Operator(bestbs, JAINcommon.GetPolTermsMixed(P.nm_vec, Diagonal([1.0, 1.0, 1.0, 0.0])); red_q=1)
        N_exp = real(bestst' * op_N * bestst)
        n_avg = N_exp / P.nml
    elseif P.name in (:PADsu3, :PADsu2)
        res, bss = PADsu3.for_generator(P, μ, 0.5, 1.0, 0.5, 3)
        bestst = res[1][2]
        bestbs = bss[[res[1][5], res[1][6], res[1][7]]]
        #bestst, bestbs, bestE, bestR, bestZ = PADsu2.ground_state(P, μ)
        op_N  = Operator(bestbs, GetPolTerms(P.nm1, P.nf1); red_q=1)
        N_exp = real(bestst' * op_N * bestst)
        n_avg = N_exp / P.nm1
    end
    return n_avg, N_exp
end

#P = KLcommon.build_model(nmf=4)
P = PADsu3.build_model(nm1=5)
#P = PADsu2.build_model(nm1=5)
μlower = -0.5
μupper = 0.5
mus = collect(range(μlower, μupper, length=15))
nf_avg_list = Float64[]; Nf_list = Float64[]
for μ in mus
    nf_avg, Nf_tot = avg_nf_for_mu(P, μ)
    push!(nf_avg_list, nf_avg); push!(Nf_list, Nf_tot)
    @info "μ=$(round(μ,digits=3))  <n_f>≈$(round(nf_avg,digits=4))  <N_f>≈$(round(Nf_tot,digits=4))"
end

fig = Figure(size = (650, 650))
ax  = Axis(fig[1, 1];
    xlabel = "μ",
    ylabel = "⟨n_123⟩",
    title  = "light fermion density vs μ",
    aspect = 1, 
    limits = ((μlower, μupper), (0.0, 3.0))   
)

lines!(ax, mus, nf_avg_list, linewidth = 2)
hlines!(ax, [0, 1, 2], color = :gray, linestyle = :dash, linewidth = 1)
fig

