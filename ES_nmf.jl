include(joinpath(@__DIR__, ".", "KLcommon.jl"))
using .KLcommon

# ——— 主程序：μ=0.312，N_mf=4..9；画 ΔE_S vs 1/√N_mf（面板正方形） ———
function fig2c_singlet_gap(μc::Float64 = 0.312; Ns = 4:9, k::Int=40)
    xs = Float64[]   # 1/√N_mf
    ys = Float64[]   # ΔE_S
    ann = String[]   # 标注 (R,Z)
    for N in Ns
        P = KLcommon.build_model(nmf=N)
        ΔE, info = KLcommon.singlet_gap(P, μc; k=k)
        push!(xs, 1 / sqrt(N))
        push!(ys, ΔE)
        push!(ann, info === nothing ? "NA" : "(R=$(info[5]),Z=$(info[6]))")
        @info "N_mf=$N  ΔE_S=$(round(ΔE,digits=6))  sector=$(last(ann))  <L2>=$(info === nothing ? NaN : info[7])  <C2>=$(info === nothing ? NaN : info[8])"
    end

    fig = Figure(size=(650,650))
    ax  = Axis(fig[1,1];
        xlabel = "1 / √N_mf",
        ylabel = "ΔE_S (l=0, s=0)",
        title  = "Critical scaling at μ = $(μc)",
        aspect = 1
    )
    scatter!(ax, xs, ys, markersize=10)
    lines!(ax, xs, ys, linewidth=1.5)
    # 可选：标注每个点对应的 (R,Z)
    for (x,y,txt) in zip(xs,ys,ann)
        text!(ax, txt, position=(x,y), align=(:left,:bottom), fontsize=10)
    end
    fig
end

# 运行：
display(fig2c_singlet_gap(0.312; Ns=4:7, k=40))
