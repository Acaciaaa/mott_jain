using XLSX
using DataFrames
using CairoMakie

xlsx_file = "data/singlet_gap.xlsx"
#xlsx_file = "data_square/singlet_gap.xlsx"
sheetname = "zoomout"

tbl = XLSX.readtable(xlsx_file, sheetname; infer_eltypes=true)
df  = DataFrame(tbl)

cols = names(df)
x    = df[!, 1]

fig = Figure(size=(800,650))
ax = Axis(fig[1,1];
    xlabel = "μ",
    ylabel = "ΔE_S · √nml",
    title  = "Singlet gap vs μ",
    #aspect = 1,
)
ax.xminorticks = IntervalsBetween(10)
ax.xminorgridvisible = true

for j in 2:ncol(df)
    colname = String(cols[j])
    nml = parse(Int, colname)
    factor = sqrt(nml)
    y = df[!, j] .* factor
    lines!(ax, x, y; label="nml=$colname", linewidth=2)
    scatter!(ax, x, y; markersize=10)
end

axislegend(ax; position=:rb)
save("/Users/ruiqi/Documents/tmp/mott_jain/singlet_gap.png", fig)
