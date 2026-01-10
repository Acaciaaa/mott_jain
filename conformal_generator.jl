include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3
using FuzzifiED
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf
using JLD2, Dates
TOL= √(eps(Float64))
nm1 = 5
μc = 0.0644
factor = 0.0856
# nm1 = 6
# μc = 0.0556
# factor = 0.075455

P = PADsu3.build_model(nm1=nm1)
data = load("generator/results_bss.jld2")
results = data["results"]
bss = data["bss"]
#results, bss = PADsu3.for_generator(P, μc,0.4,1.0,0.3, 30)

function make_lambda_candidates()
    s=P.s
    # --- A. 小费米子算符 ---
    cf = [GetElectronMod(P.nm1, P.nf1, i) for i in 1:3]

    # --- B. 大费米子算符 ---
    c0 = PadAngModes(GetElectronMod(P.nm0, P.nf0, 1), P.no1)

    # --- A. f-f 配对 ---
    Delta_ff = Dict()
    for i in 1:3, j in 1:3
        if i < j
            Delta_ff[(i,j)] = FilterL2(cf[i] * cf[j], 2 * s) 
        end
    end

    # --- B. f0-f0 配对 ---
    pair_00 = c0 * c0
    Delta_0_U0 = FilterL2(pair_00, 6 * s - 1)
    Delta_0_V1 = FilterL2(pair_00, 6 * s - 3)

    # --- C. f0-f 混合配对 ---
    Delta_0f = [FilterL2(c0 * cf[i], 4 * s) for i in 1:3]

    # --- D. 三体算符 Trion ---
    pair_12_2s = FilterL2(cf[1] * cf[2], 2 * s)
    Trion = FilterL2(pair_12_2s * cf[3], 3 * s)

    # ==========================================
    # 4. 组装生成元 Lambda (L=1 投影)
    # ==========================================
    tms_cand = Vector{Terms}()

    # 1. U_f 项
    for i in 1:3, j in (i+1):3
        term = Delta_ff[(i,j)]' * Delta_ff[(i,j)]
        push!(tms_cand, GetComponent(FilterL2(term, 1), 1, 0))
    end

    # 2. U_0 项
    term_U0 = Delta_0_U0' * Delta_0_U0
    push!(tms_cand, GetComponent(FilterL2(term_U0, 1), 1, 0))

    # 3. V_1 项
    term_V1 = Delta_0_V1' * Delta_0_V1
    push!(tms_cand, GetComponent(FilterL2(term_V1, 1), 1, 0))

    # 4. U_0f 项
    for i in 1:3
        term = Delta_0f[i]' * Delta_0f[i]
        push!(tms_cand, GetComponent(FilterL2(term, 1), 1, 0))
    end

    # 5. 隧穿项 t
    term_t = c0' * Trion + Trion' * c0
    push!(tms_cand, GetComponent(FilterL2(term_t, 1), 1, 0))

    # 6. 化学势项 mu
    for i in 1:3
        term_mu = cf[i]' * cf[i]
        push!(tms_cand, GetComponent(FilterL2(term_mu, 1), 1, 0))
    end
    return tms_cand
end
function find_state(l2, c2, rank, name)
    info = wanted_states[(l2, c2)][rank]
    st = info[2]
    qn = [info[5], info[6], info[7]]
    return st, bss[qn], info[1]
end
function find_state_othersector(l2, c2, rank, name)
    info = wanted_states_othersector[(l2, c2)][rank]
    st = info[2]
    return st, info[1]
end
function print_coefficients()
    println("================================")
    # 1. 打印 Overlap 和 Fidelity
    # 重构生成的态: |ψ_gen> = Σ c_i |v_i>
    # vecs_op 是之前算好的算符向量列表
    vec_gen = zeros(ComplexF64, length(st_dS))
    for i in eachindex(coeffs)
        vec_gen .+= coeffs[i] .* vecs_op[i]
    end

    overlap = dot(st_dS, vec_gen)
    norm_sq = real(dot(vec_gen, vec_gen))
    fidelity = abs(overlap)^2 / norm_sq

    @printf "Overlap = %.6f\n" fidelity
    println("--------------------------------")

    idx = 1 # 计数器

    println("--- 1. U_f ---")
    # 对应循环: for i in 1:3, j in (i+1):3
    for i in 1:3, j in (i+1):3
        @printf "  U_f (%d,%d) : %10.6f\n" i j abs(coeffs[idx])
        global idx += 1
    end

    println("--- 2. U_0 ---")
    @printf "  U_0        : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 3. V_1 ---")
    # === 这里是你最关心的 ===
    @printf "  V_1        : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 4. U_0f ---")
    # 对应循环: for i in 1:3
    for i in 1:3
        @printf "  U_0f (%d)   : %10.6f\n" i abs(coeffs[idx])
        global idx += 1
    end

    println("--- 5. t ---")
    @printf "  t          : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 6. mu ---")
    # 对应循环: for i in 1:3
    for i in 1:3
        @printf "  mu (%d)     : %10.6f\n" i abs(coeffs[idx])
        global idx += 1
    end

    println("================================")
end

E0 = results[1][1]
wanted_states = Dict{Tuple{Int, Int}, Any}()
wanted_states_othersector = Dict{Tuple{Int, Int}, Any}()
for (l2, c2) in [(0,0), (2,0), (6,0)]
    wanted_states[(l2, c2)] = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results)
    # println("(l2, c2) = ($(l2), $(c2))")
    # for s in wanted_states[(l2, c2)]
    #     println((s[1]-E0)/factor)
    # end
end

# Ground (l2=0 c2=0 Rank 1)
st_0, bs_0, E_0 = find_state(0, 0, 1, "0")
@assert abs(E0-E_0)<TOL "ground state error"
# S (l2=0 c2=0 Rank 2)
st_S, bs_S, E_S = find_state(0, 0, 2, "S")
# □S (l2=0 c2=0 Rank 3) 
st_boxS, bs_boxS, E_boxS = find_state(0, 0, 3, "□S")
# S' (l2=0 c2=0 Rank 4) 
st_Sprime, bs_Sprime, E_Sprime = find_state(0, 0, 4, "S'")
# ∂S (l2=2 c2=0 Rank 1)
st_dS, bs_dS, E_dS = find_state(2, 0, 1, "∂S")
# □∂S (l2=2 c2=0 Rank 2)
st_boxdS, bs_boxdS, E_boxdS = find_state(2, 0, 2, "□∂S")
# ∂S' (l2=2 c2=0 Rank 3)
st_dSprime, bs_dSprime, E_dSprime = find_state(2, 0, 3, "∂S'")
# T (l2=6 c2=0 Rank 1)
st_T, bs_T, E_T = find_state(6, 0, 1, "T")
# ∂∂S (l2=6 c2=0 Rank 2)
st_ddS, bs_ddS, E_ddS = find_state(6, 0, 2, "∂∂S")

data = load("generator/results_bss_othersector.jld2")
results_othersector = data["results"]
bs0, bsp, bsm = data["bs0"], data["bsp"], data["bsm"]
#results_J, bs0, bsp, bsm = PADsu3.for_generator_J(P, μc,0.4,1.0,0.3, 50)
for (l2, c2) in [(0,3), (2,3), (6,3)]
    wanted_states_othersector[(l2, c2)] = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results_othersector)
    # println("(l2, c2) = ($(l2), $(c2))")
    # for s in wanted_states[(l2, c2)]
    #     println((s[1]-E0)/factor)
    # end
    wanted_states[(l2, c2)] = []
    candidates_Z1 = []
    energies_Z2   = Float64[]
    tmps = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results)
    for tmp in tmps
        z_val = tmp[6]
        if abs(z_val + 1.0) < TOL      # Z=-1 Sector
            push!(candidates_Z1, tmp)
        elseif abs(z_val - 1.0) < TOL  # Z=1 Sector
            push!(energies_Z2, tmp[1])   # 我们只关心 Z=1 的能量，不用存向量
        end
    end
    
    for tmp in candidates_Z1
        E_curr = tmp[1]
        has_partner = any(e -> abs(e - E_curr) < TOL, energies_Z2)
        if has_partner
            push!(wanted_states[(l2, c2)], tmp)
        end
    end
end

# ∂⋅J' (l2=0 c2=3 Rank 1)
st_divJprime, bs_divJprime, E_divJprime = find_state(0, 3, 1, "∂⋅J'")
# J (l2=2 c2=3 Rank 1)
st_J, bs_J, E_J = find_state(2, 3, 1, "J")
st_J_o, E_J_o = find_state_othersector(2, 3, 1, "J")
# ϵ∂J (l2=2 c2=3 Rank 2) 
st_curlJ, bs_curlJ, E_curlJ = find_state(2, 3, 2, "ϵ∂J")
st_curlJ_o, E_curlJ_o = find_state_othersector(2, 3, 2, "ϵ∂J")
# □J (l2=2 c2=3 Rank 3) 
st_boxJ, bs_boxJ, E_boxJ = find_state(2, 3, 4, "□J")
st_boxJ_o, E_boxJ_o = find_state_othersector(2, 3, 4, "□J")
# J' (l2=2 c2=3 Rank unknown)
st_Jprime, bs_Jprime, E_Jprime = find_state(2, 3, 3, "J'")
st_Jprime_o, E_Jprime_o = find_state_othersector(2, 3, 3, "J'")
# ϵ∂J' (l2=2 c2=3 Rank unknown) 
st_curlJprime, bs_curlJprime, E_curlJprime = find_state(2, 3, 6, "ϵ∂J'")
st_curlJprime_o, E_curlJprime_o = find_state_othersector(2, 3, 6, "ϵ∂J'")
# ∂J (l2=6 c2=3 Rank 1)
st_dJ, bs_dJ, E_dJ = find_state(6, 3, 1, "∂J")
st_dJ_o, E_dJ_o = find_state_othersector(6, 3, 1, "∂J")
# ϵ∂∂J (l2=6 c2=3 Rank 2)
st_epsddJ, bs_epsddJ, E_epsddJ = find_state(6, 3, 3, "ϵ∂∂J")
st_epsddJ_o, E_epsddJ_o = find_state_othersector(6, 3, 3, "ϵ∂∂J")
# ∂J' (l2=6 c2=3 Rank unknown)
st_dJprime, bs_dJprime, E_dJprime = find_state(6, 3, 5, "∂J'")

tms_cand = make_lambda_candidates()
vecs_op = []
for tms in tms_cand
    op = Operator(bs_S, bs_dS, tms)
    push!(vecs_op, op * st_S)
end
n_cand = length(tms_cand)
M = zeros(ComplexF64, n_cand, n_cand)
b = zeros(ComplexF64, n_cand)
for i in 1:n_cand
    b[i] = dot(vecs_op[i], st_dS)
    for j in 1:n_cand
        M[i,j] = dot(vecs_op[i], vecs_op[j])
    end
end
coeffs = pinv(M) * b

#print_coefficients()

# ==============================================================================
# 阶段 4: 复现 Table IV
# ==============================================================================

tms_Lambdaz = SimplifyTerms(coeffs' * tms_cand)
function remove_level_mixing()
    op = Operator(bs_dS, bs_ddS, tms_Lambdaz)
    v = op * st_dS

    c1 = dot(st_T, v)
    c2 = dot(st_ddS, v)

    st_ddS_star = c1 .* st_T .+ c2 .* st_ddS
    normalize!(st_ddS_star)
    st_T_star = c2 .* st_T .- c1 .* st_ddS
    normalize!(st_T_star)
    return st_ddS_star, st_T_star
end

st_ddS_star,_ = remove_level_mixing()

"""
project_L_state(st, op_l2, target_l, all_ls)

功能: 从混合态 st 中分离出具有特定角动量 target_l 的纯态。
原理: 构造投影算符 P = Π (L^2 - λ_bad) / (λ_target - λ_bad)，依次剔除不需要的分量。

参数:
- st:       输入的混合态矢量 (Vector)
- op_l2:    L^2 算符矩阵 (OpMat 或 Matrix)
- target_l: 你想要的那个角动量量子数 (Int, 如 1)
- all_ls:   该态中混入的所有角动量量子数列表 (Vector{Int}, 如 [1, 2, 3])

返回:
- 投影后的纯态矢量
"""
function project_state(st, op_l2, l_target, all_ls)
    st_out = copy(st)
    l2_target = l_target * (l_target + 1)
    
    for l in all_ls
        if l == l_target
            continue
        end
        l2_bad = l * (l + 1)
        denominator = l2_target - l2_bad
        st_out = (op_l2 * st_out .- l2_bad .* st_out) ./ denominator
    end
    return st_out
end
function get_possible_ls(l_input::Int)
    if l_input == 0
        return [1]
    end
    return [l_input - 1, l_input, l_input + 1]
end
function calc_row(label_in, l_input, input_tuple, l_target, targets_list)
    (st_in, bs_in) = input_tuple
    bs_tgt = targets_list[1][2] # Use basis of first target
    op = Operator(bs_in, bs_tgt, tms_Lambdaz)
    v = op * st_in
    v_proj = project_state(v, Operator(bs_tgt, bs_tgt, P.tms_l2), l_target, get_possible_ls(l_input))
    denom_sq = Real(dot(v_proj, v_proj))
    @assert denom_sq>0.1 "denom_sq invalid"
    #println("denom_sq:$denom_sq")
    
    @printf "| %-8s | %-2d |" label_in l_target
    total_ovlp = 0.0
    
    for (name_t, _, st_t) in targets_list
        #println("tmp:$(abs(dot(st_t, v))^2)")
        val = abs(dot(st_t, v))^2 / denom_sq
        @printf " %-6s %.4f |" name_t val
        total_ovlp += val
    end
    
    if length(targets_list) < 2
        print("        -      |")
    end
    @printf " %.4f |\n" total_ovlp
end

function calc_row_sameang(label_in, l_input, st_in, l_target, targets_list)
    op_Lm = Operator(bs0, bsm, P.tms_lm)
    op = Operator(bsm, bsm, tms_Lambdaz)
    op_Lp = Operator(bsm, bs0, P.tms_lp)
    v_othersector = op_Lm * st_in
    v_othersector_result = op * v_othersector
    v = op_Lp *  v_othersector_result
    v_proj = project_state(v, Operator(bs0, bs0, P.tms_l2), l_target, get_possible_ls(l_input))
    denom_sq = Real(dot(v_proj, v_proj))
    @assert denom_sq>0.1 "denom_sq invalid"
    #println("denom_sq:$denom_sq")
    
    @printf "| %-8s | %-2d |" label_in l_target
    total_ovlp = 0.0
    
    for (name_t, st_t) in targets_list
        #println("tmp:$(abs(dot(st_t, v))^2)")
        val = abs(dot(st_t, v))^2 / denom_sq
        @printf " %-6s %.4f |" name_t val
        total_ovlp += val
    end
    
    if length(targets_list) < 2
        print("        -      |")
    end
    @printf " %.4f |\n" total_ovlp
end

println("| Input    | l' | Target 1 Ovlp | Target 2 Ovlp | Total  |")
println("|----------|----|---------------|---------------|--------|")

# --- Row 1: S -> ∂S ---
calc_row("S", 0, (st_S, bs_S),
    1, [("∂S", bs_dS, st_dS)])

# --- Row 2: ∂S -> S, □S ---
calc_row("∂S", 1, (st_dS, bs_dS),
    0, [("S", bs_S, st_S), ("□S", bs_boxS, st_boxS)])

# --- Row 3: ∂S -> ∂∂S(*) ---
calc_row("∂S", 1, (st_dS, bs_dS),
    2, [("∂∂S(*)", bs_ddS, st_ddS_star)])

# --- Row 4: □S -> ∂S, □∂S ---
calc_row("□S", 0, (st_boxS, bs_boxS),
    1, [("∂S", bs_dS, st_dS), ("□∂S", bs_boxdS, st_boxdS)])

# --- Row 5: ∂∂S(*) -> ∂S, □∂S ---
calc_row("∂∂S(*)", 2, (st_ddS_star, bs_ddS),
    1, [("∂S", bs_dS, st_dS), ("□∂S", bs_boxdS, st_boxdS)])

println("|----------|----|---------------|---------------|--------|")

# --- Row 6: S' -> ∂S' ---
calc_row("S'", 0, (st_Sprime, bs_Sprime),
    1, [("∂S'", bs_dSprime, st_dSprime)])

println("|----------|----|---------------|---------------|--------|")
    
# --- Row 7: J -> ϵ∂J (same ang!) ---
calc_row_sameang("J", 1, st_J_o,
        1, [("ϵ∂J", st_curlJ_o)])

# --- Row 8: J -> ∂J ---
calc_row("J", 1, (st_J, bs_J),
        2, [("∂J", bs_dJ, st_dJ)])
    
# --- Row 9: ∂J -> J, □J ---
calc_row("∂J", 2, (st_dJ, bs_dJ),
        1, [("J", bs_J, st_J), ("□J", bs_boxJ, st_boxJ)])
    
# --- Row 10: ∂J -> ϵ∂∂J (same ang!) ---
calc_row_sameang("∂J", 2, st_dJ_o,
        2, [("ϵ∂∂J", st_epsddJ_o)])
    
# --- Row 11: ϵ∂J -> J, □J (same ang!) ---
calc_row_sameang("ϵ∂J", 1, st_curlJ_o,
        1, [("J", st_J_o), ("□J", st_boxJ_o)])
    
# --- Row 12: ϵ∂J -> ϵ∂∂J ---
calc_row("ϵ∂J", 1, (st_curlJ, bs_curlJ),
        2, [("ϵ∂∂J", bs_epsddJ, st_epsddJ)])

println("|----------|----|---------------|---------------|--------|")

# --- Row 13: J' -> ∂⋅J' ---
calc_row("J'", 1, (st_Jprime, bs_Jprime),
        0, [("∂⋅J'", bs_divJprime, st_divJprime)])
    
# --- Row 14: J' -> ϵ∂J' (same ang!) ---
calc_row_sameang("J'", 1, st_Jprime_o,
        1, [("ϵ∂J'", st_curlJprime_o)])
    
# --- Row 15: J' -> ∂J' ---
calc_row("J'", 1, (st_Jprime, bs_Jprime),
        2, [("∂J'", bs_dJprime, st_dJprime)])
    
println("|----------|----|---------------|---------------|--------|")

function check_current_conservation()
    # op = Operator(bs0, bs0, tms_Lambdaz)
    # v = op * st_J_o
    # v_proj = project_state(v, Operator(bs0, bs0, P.tms_l2), 0, get_possible_ls(1))
    op = Operator(bs_J, bs_dJ, tms_Lambdaz)
    v = op * st_J
    v_proj = project_state(v, Operator(bs_dJ, bs_dJ, P.tms_l2), 0, get_possible_ls(1))
    @info Real(dot(v_proj, v_proj))/Real(dot(v, v))
end
function check_tensor_conservation()
    _ ,st_T_star = remove_level_mixing()
    info_T = wanted_states[(6, 0)][1]
    bs_T = bss[[info_T[5], info_T[6], info_T[7]]]
    bs_after = bss[[info_T[5], info_T[6], -info_T[7]]]
    op = Operator(bs_T, bs_after, tms_Lambdaz)
    v = op * st_T_star
    v_proj = project_state(v, Operator(bs_after, bs_after, P.tms_l2), 1, get_possible_ls(2))
    @info Real(dot(v_proj, v_proj))/Real(dot(v, v))
end
check_current_conservation()
check_tensor_conservation()