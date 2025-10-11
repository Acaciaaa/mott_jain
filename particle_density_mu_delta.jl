using LinearAlgebra
using CSV, DataFrames

include(joinpath(@__DIR__, "JAINcommon.jl"))
using .JAINcommon
using FuzzifiED

P = JAINcommon.build_model_su2u1(nml=4)
const choices = Dict(
    :n123 => (Diagonal([1.0, 1.0, 1.0, 0.0]), P.nml),
    :n12  => (Diagonal([1.0, 1.0, 0.0, 0.0]), P.nml),
    :n3   => (Diagonal([0.0, 0.0, 1.0, 0.0]), P.nml),
    :n4   => (Diagonal([0.0, 0.0, 0.0, 1.0]), P.nmh)
)

@inline function avg_nf(P, bestst, bestbs, which::Symbol)
    diagw, denom = choices[which]
    opN  = Operator(bestbs, JAINcommon.GetPolTermsMixed(P.nm_vec, diagw); red_q=1)
    N_exp = real(bestst' * opN * bestst)
    n_avg = N_exp / denom
    return n_avg, N_exp
end

mu_range    = collect(range(0.0,  1.0; length=20))
delta_range = collect(range(-1.0,  1.0; length=10))
rows = NamedTuple[]

for μ in mu_range
    for Δ in delta_range
        bestst, bestbs, bestE, bestR, bestZ = JAINcommon.ground_state_su2u1(P, μ, Δ)

        n123,  _ = avg_nf(P, bestst, bestbs, :n123)
        n12,   _ = avg_nf(P, bestst, bestbs, :n12)
        n3,    _ = avg_nf(P, bestst, bestbs, :n3)
        n4,    _ = avg_nf(P, bestst, bestbs, :n4)

        push!(rows, (mu=μ, delta=Δ, n123=n123, n12=n12, n3=n3, n4=n4))
    end
end

df = DataFrame(rows)
CSV.write("data/density_mu_delta.csv", df)
println("Saved: density_mu_delta.csv  (long form)")
# Z = reshape(df_long.density, (length(delta_range), length(mu_range)))
# df_grid = DataFrame([Symbol("Δ\\μ")=>delta_range,
#     (Symbol("μ=$(round(m,digits=4))") => Z[:,i] for (i,m) in enumerate(mu_range))...])
# CSV.write("density_grid.csv", df_grid)
# println("Saved: density_grid.csv   (grid form)")
