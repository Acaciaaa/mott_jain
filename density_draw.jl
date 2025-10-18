using CSV, DataFrames
using CairoMakie

# === 读文件 ===
df = CSV.read("data/density_mu.csv", DataFrame)

# 假定文件就是两列：mu, nf
mus = df[:, 1]
nfs = df[:, 2]

# 去掉缺失值
mask = .!ismissing.(nfs)
mus = mus[mask]
nfs = nfs[mask]

# === 画图 ===
fig = Figure(size = (650, 650))
ax = Axis(fig[1, 1];
    xlabel = "μ",
    ylabel = "⟨n_123⟩",
    title  = "light fermion density vs μ",
    aspect = 1,
    limits = ((minimum(mus), maximum(mus)), (-1.0, 4.0))
)

lines!(ax, mus, nfs, linewidth = 2)
hlines!(ax, [0, 1, 2], color = :gray, linestyle = :dash, linewidth = 1)

fig
