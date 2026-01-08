using CairoMakie

x = [1/2, 1/sqrt(5), 1/sqrt(6),1/sqrt(7)]
y = [0.2545, 0.1590, 0.0967, 0.0678]
y = [0.2281, 0.1425, 0.0967, 0.0682]

# 取值范围（刻度语义）
xmin, xmax = 0.0, 0.5
ymin, ymax = 0.0, 0.3

# 轴内留白
padx = 0.06 * (xmax - xmin)
pady = 0.06 * (ymax - ymin)

fig = Figure(size = (420, 420))
ax = Axis(fig[1, 1];
    limits = (xmin - padx, xmax + padx, ymin - pady, ymax + pady),
    xticks = 0:0.1:0.5,
    yticks = 0:0.05:0.3,
    xlabel = "",
    ylabel = ""
)

# 散点
scatter!(ax, x, y; color = :black, markersize = 10)

# ===== 新增：c / x^3 曲线 =====
c = 5.4           # ← 你自己调这个
xs = range(0.0, 0.5; length = 400)  
ys = c .* xs.^4.5

lines!(ax, xs, ys; color = :black, linewidth = 1.5)

fig
