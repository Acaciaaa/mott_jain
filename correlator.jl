include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3
using FuzzifiED
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf
using JLD2, Dates
using LegendrePolynomials
TOL= √(eps(Float64))
coef = [0.6,1.4,0.6]
nm1 = 5
μc = 0.1156
factor = 0.13149

P = PADsu3.build_model(nm1=nm1)
# data = load("generator/results_bss.jld2")
# results = data["results"]
# bss = data["bss"]
results, bss = PADsu3.for_generator(P, μc, coef[1], coef[2], coef[3],7)
tmp = filter(st -> abs(st[3]) < TOL && abs(st[4]) < TOL, results)
st_0, bs_0, E_0 = tmp[1][2], bss[[tmp[1][5],tmp[1][6],tmp[1][7]]], tmp[1][1]
st_S, bs_S, E_S = tmp[2][2], bss[[tmp[2][5],tmp[2][6],tmp[2][7]]], tmp[2][1]
obs_n = StoreComps(GetDensityObs(P.nm1, P.nf1))
bs_even = bss[[0, 1, 1]]
bs_odd = bss[[0, 1, -1]]


op_n00_vac = Operator(bs_0, bs_0, GetComponent(obs_n, 0, 0))
val_n0 = st_0' * op_n00_vac * st_0
lambda_0_sq = abs(val_n0)^2

op_n00_S = Operator(bs_0, bs_S, GetComponent(obs_n, 0, 0))
ovl_S = st_S' * op_n00_S * st_0
lambda_S_sq = abs(ovl_S)^2

#@printf "Background (λ0^2): %.6f\n" lambda_0_sq
#@printf "Normalization (λS^2): %.6f\n" norm_factor

ops = Dict([ l => Operator(bs_0, iseven(l) ? bs_even : bs_odd, GetComponent(obs_n, l, 0.0)) 
                    for l = 0 : P.nm1 - 1])
cor_l = [ begin 
    st = ops[l] * st_0
    st' * st / lambda_S_sq
end for l = 0 : P.nm1 - 1]
Cor(θ) = ([Pl(cos(θ), l) * (2 * l + 1) for l in 0 : P.nm1 - 1]' * cor_l) - lambda_0_sq / lambda_S_sq


θs = 0.1 : π / 20 : 2 * π 
pic = lines(θs, Cor.(θs), 
    axis = (; title = "Correlation", xlabel = "Theta")
)
display(pic)
# println("\n=== 3. 关联函数表 (Correlation Function Table) ===")
# @printf "%-10s | %-15s\n" "Theta/π" "C(theta)"
# @printf "%-10s | %-15s\n" "----------" "---------------"
# for theta_val in 0.1 : 0.1 : 1.0
#     val = GetCorrelator(theta_val * π)
#     @printf "%.2f       | %.8f\n" theta_val val
# end

# Δ = -0.5 * log2( C(π) )
C_pi = Cor(π)
Delta_value = -0.5 * log2(abs(C_pi))
# Δ = 2 * C''(π) / C(π)
h = 1e-4
C_pi_minus = Cor(π - h)
C_double_prime = 2 * (C_pi_minus - C_pi) / (h^2)
Delta_curv = 2 * C_double_prime / C_pi

u_val = (Delta_value + Delta_curv) / 2

@printf "Delta (Value)    : %.4f\n" Delta_value
@printf "Delta (Curvature): %.4f\n" Delta_curv
@printf "Combined u       : %.4f\n" u_val
@printf "ED results       : %.4f\n" (E_S-E_0)/factor