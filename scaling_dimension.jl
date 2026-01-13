include(joinpath(@__DIR__, ".", "pad_su3.jl"))
include(joinpath(@__DIR__, ".", "pad_su2.jl"))
using .PADsu3
using .PADsu2
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf

μc = 0.1156
nm1 = 5

function su3()
    mus, results_vec = PADsu3.read_results(nm1,"data/results_$(nm1).jld2")
    #mus, results_vec = PADsu3.read_results(nm1,"critical_point/results_$(nm1).jld2")
    idx = argmin(abs.(mus .- μc)); val = mus[idx]; results = results_vec[idx]
    #P = PADsu3.build_model(nm1=nm1)
    #results = PADsu3.lowest_k_states(P, μc,0.4,1.0,0.3, 10)
    #@printf("%.7f", val)
    #@info results
    E0 = results[1][1]
    # Eref = filter(st -> st[2] ≈ 6 && st[3] ≈ 0, results)[1][1]
    # factor = (Eref - E0) / 3

    enrg_cal = [
            filter(st -> st[2] ≈ 2 && st[3] ≈ 0, results)[1][1] - filter(st -> st[2] ≈ 0 && st[3] ≈ 0, results)[2][1], # ∂S - S
            filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[1][1] - E0, # J
            filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results)[3][1] - E0, # ϵ∂J
            filter(st -> st[2] ≈ 6 && st[3] ≈ 3, results)[1][1] - E0, # ∂J
            filter(st -> st[2] ≈ 6 && st[3] ≈ 0, results)[1][1] - E0 # T
        ]
    #@info [mu;enrg_cal]
    dim_cal = Float64[1, 2, 3, 3, 3]
    factor = (enrg_cal' * dim_cal) / (dim_cal' * dim_cal)
    @info factor
    
    #P = PADsu3.build_model(nm1=nm1)
    #results_ = PADsu3.lowest_k_states_adjoint(P, μc,0.4,0.9,0.7, 7)
    # @info results
    #ratio = (filter(st -> st[2] ≈ 0 && st[3] ≈ 3, results_)[1][1]-E0)/(filter(st -> st[2] ≈ 2 && st[3] ≈ 3, results_)[1][1]-E0)
    #@info ratio
    
    tol = 1e-2
    to_l(L2) = (sqrt(1 + 4L2) - 1) / 2
    filtered_dim = [
        (Δ = (st[1] - E0) / factor,
         l = round(Int, to_l(st[2]) + tol),
         C2 = st[3])
        for st in results if (st[2] < 21 && st[3] < 9)
    ]
    #@info filtered_dim
    
    colors = Dict(0.0=>:red, 3.0=>:blue, 6.0=>:green, 8.0=>:purple)
    labels = Dict(0.0=>"C₂=0", 3.0=>"C₂=3", 6.0=>"C₂=6", 8.0=>"C₂=8")
    dx = 0.15
    
    fig = Figure(size=(600,700))
    ax = Axis(fig[1,1];
        xlabel = "l", ylabel = "Δ",
        title = "Nm=$(nm1): scaling dimension vs l",
        aspect = DataAspect() 
    )
    
    ax.xticks = 0:1:4
    xlims!(ax, -0.3, 4.3)
    ax.yticks = 0:1:5
    ylims!(ax, -0.3, 5.3)
    
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
    #save("/Users/ruiqi/Documents/tmp/mott_jain/scaling_dimension_$(nm1).png", fig)
    #save("/Users/ruiqi/Desktop/$(nm1)_k30.png", fig)
    
end

function su2()
    mus, results_vec = PADsu2.read_results(nm1)
    idx = argmin(abs.(mus .- μc)); val = mus[idx]; results = results_vec[idx]
    @printf("%.7f", val)
    @info results
    E0 = results[1][1]
    Eref = filter(st -> st[2] ≈ 6 && st[3] ≈ 0, results)[1][1]
    factor = (Eref - E0) / 3
    
    
    tol = 1e-2
    to_l(L2) = (sqrt(1 + 4L2) - 1) / 2
    filtered_dim = [
        (Δ = (st[1] - E0) / factor,
         l = round(Int, to_l(st[2]) + tol),
         s = round(Int, to_l(st[3]) + tol))
        for st in results if (st[2] < 21 && st[3] < 7)
    ]
    
    colors = Dict(0.0=>:red, 1.0=>:blue, 2.0=>:green)
    labels = Dict(0.0=>"s=0", 1.0=>"s=1", 2.0=>"s=2")
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
        c = get(colors, st[3], :gray)
        lines!(ax, [st.l - dx, st.l + dx], [st.Δ, st.Δ]; color=c, linewidth=2)
        scatter!(ax, [st.l], [st.Δ]; color=c, markersize=10)
    end
    
    for (c2, name) in labels
        lines!(ax, [NaN, NaN], [NaN, NaN]; color=colors[c2], linewidth=2, label=name)
    end
    
    
    axislegend(ax, position=:rb, framevisible=false)
    fig
    
end

su3()
#su2()