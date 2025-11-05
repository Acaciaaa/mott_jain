include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf

μc = 0.096
k = 30
P = PADsu3.build_model(nm1=6)
results = PADsu3.lowest_k_states(P, μc, k) #已经排好序了 
E0 = results[1][1]
Eref = filter(st -> st[2] ≈ 6 && st[3] ≈ 0, results)[1][1]
factor = (Eref - E0) / 3


tol = 1e-2
to_l(L2) = (sqrt(1 + 4L2) - 1) / 2
filtered_dim = [
    (Δ = (st[1] - E0) / factor,
     l = round(Int, to_l(st[2]) + tol),
     C2 = st[3])
    for st in results if (st[2] < 21 && st[3] < 7)
]

colors = Dict(0.0=>:red, 3.0=>:blue, 6.0=>:green)
labels = Dict(0.0=>"C₂=0", 3.0=>"C₂=3", 6.0=>"C₂=6")
dx = 0.15

fig = Figure(size=(600,700))
ax = Axis(fig[1,1];
    xlabel = "l", ylabel = "Δ",
    title = "scaling dimension vs l",
    aspect = DataAspect() 
)

ax.xticks = 0:1:4
xlims!(ax, -0.3, 4.3)
ax.yticks = 0:1:6
ylims!(ax, -0.3, 6.3)

for st in filtered_dim
    c = get(colors, st.C2, :gray)
    lines!(ax, [st.l - dx, st.l + dx], [st.Δ, st.Δ]; color=c, linewidth=2)
    scatter!(ax, [st.l], [st.Δ]; color=c, markersize=10)
end

for (c2, name) in labels
    lines!(ax, [NaN, NaN], [NaN, NaN]; color=colors[c2], linewidth=2, label=name)
end


axislegend(ax, position=:rb, framevisible=false)
fig
