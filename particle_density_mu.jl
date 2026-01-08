include(joinpath(@__DIR__, "pad_su3.jl"))
include(joinpath(@__DIR__, "pad_su2.jl"))
using .PADsu3
using .PADsu2
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions  # 如果需要 beta_inc 等
using CairoMakie

# ==== 3) 观测 <N_f> 与 平均占据 <n_f>=<N_f>/nmf ====
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
        bestst, bestbs, bestE, bestR, bestZ = PADsu3.ground_state(P, μ,[0.4,1.2,4.0],0.9,0.2)
        #bestst, bestbs, bestE, bestR, bestZ = PADsu2.ground_state(P, μ)
        op_N  = Operator(bestbs, GetPolTerms(P.nm1, P.nf1); red_q=1)
        N_exp = real(bestst' * op_N * bestst)
        n_avg = N_exp / P.nm1
    end
    return n_avg, N_exp
end

# ==== 4) 扫 μ 并画图 ====
#P = KLcommon.build_model(nmf=4)
P = PADsu3.build_model(nm1=5)
#P = PADsu2.build_model(nm1=5)
μlower = -0.1
μupper = 0.15
mus = collect(range(μlower, μupper, length=10))
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


# using CSV, DataFrames

# # --- 你的输入 --- #
# P = JAINcommon.build_model_su2u1(nml=5)
# μlower, μupper = 0.0, 0.8
# mus = collect(range(μlower, μupper, length=10))
# path_csv = "data/density_mu.csv"

# # avg_nf_for_mu(P, μ) 已存在；返回 (nf_avg::Float64, Nf_tot::Float64)

# # ===== 工具函数 =====
# function init_csv(path::AbstractString, mus::AbstractVector{<:Real})
#     mkpath(dirname(path))
#     df = DataFrame(mu = Float64.(mus),
#                    nf = Vector{Union{Missing,Float64}}(missing, length(mus)))
#     CSV.write(path, df)
#     return df
# end

# function load_csv_or_init(path::AbstractString, mus::Vector{Float64})
#     if !isfile(path)
#         return init_csv(path, mus)
#     end
#     df = CSV.read(path, DataFrame)

#     # 若缺列，补列；若列名/类型不对，强制修正
#     if !(:mu in names(df)); df.mu = Float64[]; end
#     if !(:nf in names(df)); df.nf = Vector{Union{Missing,Float64}}(missing, nrow(df)); end
#     df.mu = Float64.(df.mu)
#     df.nf = Union{Missing,Float64}.(df.nf)

#     # 若网格不匹配（长度或数值），重建
#     if nrow(df) != length(mus) || any(abs.(df.mu .- mus) .> 1e-12)
#         df = init_csv(path, mus)
#     end
#     return df
# end

# # ===== 主流程：断点续算并写回 =====
# df = load_csv_or_init(path_csv, Float64.(mus))

# start_idx = findfirst(ismissing, df.nf)
# if start_idx === nothing
#     @info "所有 μ 的 nf 都已计算完成。"
# else
#     for i in start_idx:nrow(df)
#         if ismissing(df.nf[i])
#             μ = df.mu[i]
#             nf_avg, _ = avg_nf_for_mu(P, μ)    # 只取第一个返回值
#             df.nf[i] = Float64(nf_avg)
#             CSV.write(path_csv, df)            # 立刻落盘
#             @info "μ=$(round(μ, digits=3))  nf≈$(round(nf_avg, digits=6))  [$i/$(nrow(df))]"
#         end
#     end
# end