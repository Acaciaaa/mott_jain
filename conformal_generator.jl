include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3
using FuzzifiED
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf
TOL= √(eps(Float64))
nm1 = 5
μc = 0.0644
factor = 0.0856

# nm1 = 6
# μc = 0.0975
# factor = 0.04573

P = PADsu3.build_model(nm1=nm1)
results, bss = PADsu3.for_generator(P, μc,0.4,1.0,0.3, 100)

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
    if c2 == 0
        qn = [info[5], info[6], info[7]]
        return st, bss[qn], info[1]
    elseif c2 == 3
        qn = [info[5], info[6]]
        return st, bss_J[qn], info[1]
    end
end
function print_coefficients()
    println("\n=== 最终结果 (Final Results) ===")

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

    @printf "Fidelity = %.6f\n" fidelity
    println("--------------------------------")

    idx = 1 # 计数器

    println("--- 1. U_f (Interaction among small fermions) ---")
    # 对应循环: for i in 1:3, j in (i+1):3
    for i in 1:3, j in (i+1):3
        @printf "  U_f (%d,%d) : %10.6f\n" i j abs(coeffs[idx])
        global idx += 1
    end

    println("--- 2. U_0 (Interaction 6s-1) ---")
    @printf "  U_0        : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 3. V_1 (Interaction 6s-3) ---")
    # === 这里是你最关心的 ===
    @printf "  V_1        : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 4. U_0f (Interaction large-small) ---")
    # 对应循环: for i in 1:3
    for i in 1:3
        @printf "  U_0f (%d)   : %10.6f\n" i abs(coeffs[idx])
        global idx += 1
    end

    println("--- 5. Tunneling t ---")
    @printf "  t          : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 6. Chemical Potential mu ---")
    # 对应循环: for i in 1:3
    for i in 1:3
        @printf "  mu (%d)     : %10.6f\n" i abs(coeffs[idx])
        global idx += 1
    end

    println("================================")
end

E0 = results[1][1]
wanted_states = Dict{Tuple{Int, Int}, Any}()
for (l2, c2) in [(0,0), (2,0), (6,0)]
    wanted_states[(l2, c2)] = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results)
    # println("(l2, c2) = ($(l2), $(c2))")
    # for s in wanted_states[(l2, c2)]
    #     println((s[1]-E0)/factor)
    # end
end

# Ground (l2=0 c2=0 Rank 1)
st_0, bs_0, _ = find_state(0, 0, 1, "0")
# S (l2=0 c2=0 Rank 2)
st_S, bs_S, _ = find_state(0, 0, 2, "S")
# □S (l2=0 c2=0 Rank 3) 
st_boxS, bs_boxS, _ = find_state(0, 0, 3, "□S")
# S' (l2=0 c2=0 Rank 4) 
st_Sprime, bs_Sprime, _ = find_state(0, 0, 4, "S'")
# ∂S (l2=2 c2=0 Rank 1)
st_dS, bs_dS, _ = find_state(2, 0, 1, "∂S")
# □∂S (l2=2 c2=0 Rank 2)
st_boxdS, bs_boxdS, _ = find_state(2, 0, 2, "□∂S")
# ∂S' (l2=2 c2=0 Rank 3)
st_dSprime, bs_dSprime, _ = find_state(2, 0, 3, "∂S'")
# T (l2=6 c2=0 Rank 1)
st_T, bs_T, _ = find_state(6, 0, 1, "T")
# ∂∂S (l2=6 c2=0 Rank 2)
st_ddS, bs_ddS, _ = find_state(6, 0, 2, "∂∂S")

results_J, bss_J = PADsu3.for_generator_J(P, μc,0.4,1.0,0.3, 30)
for (l2, c2) in [(0,3), (2,3), (6,3)]
    wanted_states[(l2, c2)] = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results_J)
    # println("(l2, c2) = ($(l2), $(c2))")
    # for s in wanted_states[(l2, c2)]
    #     println((s[1]-E0)/factor)
    # end

    # wanted_states[(l2, c2)] = []
    # candidates_Z1 = []
    # energies_Z2   = Float64[]
    # tmps = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results)
    # for tmp in tmps
    #     z_val = tmp[6]
    #     if abs(z_val + 1.0) < TOL      # Z=-1 Sector
    #         push!(candidates_Z1, tmp)
    #     elseif abs(z_val - 1.0) < TOL  # Z=1 Sector
    #         push!(energies_Z2, tmp[1])   # 我们只关心 Z=1 的能量，不用存向量
    #     end
    # end
    
    # for tmp in candidates_Z1
    #     E_curr = tmp[1]
    #     has_partner = any(e -> abs(e - E_curr) < TOL, energies_Z2)
    #     if has_partner
    #         push!(wanted_states[(l2, c2)], tmp)
    #     end
    # end
end

# ∂⋅J' (l2=0 c2=3 Rank 1)
st_divJprime, bs_divJprime, _ = find_state(0, 3, 1, "∂⋅J'")
# J (l2=2 c2=3 Rank 1)
st_J, bs_J, _ = find_state(2, 3, 1, "J")
# ϵ∂J (l2=2 c2=3 Rank 2) 
st_curlJ, bs_curlJ, _ = find_state(2, 3, 2, "ϵ∂J")
# □J (l2=2 c2=3 Rank 3) 
st_boxJ, bs_boxJ, _ = find_state(2, 3, 3, "□J")
# J' (l2=2 c2=3 Rank unknown)
st_Jprime, bs_Jprime, _ = find_state(2, 3, 4, "J'")
# ϵ∂J' (l2=2 c2=3 Rank unknown) 
st_curlJprime, bs_curlJprime, _ = find_state(2, 3, 5, "ϵ∂J'")
# ∂J (l2=6 c2=3 Rank 1)
st_dJ, bs_dJ, _ = find_state(6, 3, 1, "∂J")
# ϵ∂∂J (l2=6 c2=3 Rank 2)
st_epsddJ, bs_epsddJ, _ = find_state(6, 3, 2, "ϵ∂∂J")
# ∂J' (l2=6 c2=3 Rank unknown)
st_dJprime, bs_dJprime, _ = find_state(6, 3, 3, "∂J'")

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

function calc_row(label_in, input_tuple, c2, l_target, targets_list)
    (st_in, bs_in) = input_tuple
    bs_tgt = targets_list[1][2] # Use basis of first target
    op = Operator(bs_in, bs_tgt, tms_Lambdaz)
    v = op * st_in

    l_target2 = l_target * (l_target + 1)
    denom_sq = 0.0
    for r in wanted_states[(l_target2, c2)]
        v_eig = r[2]
        denom_sq += abs(dot(v_eig, v))^2
    end
    
    @printf "| %-8s | %-2d |" label_in l_target
    total_ovlp = 0.0
    
    for (name_t, _, st_t) in targets_list
        val = abs(dot(st_t, v))^2 / denom_sq
        @printf " %-6s %.4f |" name_t val
        total_ovlp += val
    end
    
    if length(targets_list) < 2
        print("        -      |")
    end
    @printf " Total: %.4f |\n" total_ovlp
end

println("| Input    | L' | Target 1 Ovlp | Target 2 Ovlp | Total  |")
println("|----------|----|---------------|---------------|--------|")

# --- Row 1: S -> ∂S ---
calc_row("S", (st_S, bs_S), 0,
    1, [("∂S", bs_dS, st_dS)])

# --- Row 2: ∂S -> S, □S ---
calc_row("∂S", (st_dS, bs_dS), 0,
    0, [("S", bs_S, st_S), ("□S", bs_boxS, st_boxS)])

# --- Row 3: ∂S -> ∂∂S(*) ---
calc_row("∂S", (st_dS, bs_dS), 0,
    2, [("∂∂S(*)", bs_ddS, st_ddS_star)])

# --- Row 4: □S -> ∂S, □∂S ---
calc_row("□S", (st_boxS, bs_boxS), 0, 
    1, [("∂S", bs_dS, st_dS), ("□∂S", bs_boxdS, st_boxdS)])

# --- Row 5: ∂∂S(*) -> ∂S, □∂S ---
calc_row("∂∂S(*)", (st_ddS_star, bs_ddS), 0,
    1, [("∂S", bs_dS, st_dS), ("□∂S", bs_boxdS, st_boxdS)])

println("|----------|----|---------------|---------------|--------|")

# --- Row 6: S' -> ∂S' ---
calc_row("S'", (st_Sprime, bs_Sprime), 0,
    1, [("∂S'", bs_dSprime, st_dSprime)])

println("|----------|----|---------------|---------------|--------|")
    
# --- Row 7: J -> ϵ∂J ---
calc_row("J", (st_J, bs_J), 3, 
        1, [("ϵ∂J", bs_curlJ, st_curlJ)])

# --- Row 8: J -> ∂J ---
calc_row("J", (st_J, bs_J), 3, 
        2, [("∂J", bs_dJ, st_dJ)])
    
# --- Row 9: ∂J -> J, □J ---
calc_row("∂J", (st_dJ, bs_dJ), 3,
        1, [("J", bs_J, st_J), ("□J", bs_boxJ, st_boxJ)])
    
# --- Row 10: ∂J -> ϵ∂∂J ---
calc_row("∂J", (st_dJ, bs_dJ), 3, 
        2, [("ϵ∂∂J", bs_epsddJ, st_epsddJ)])
    
# --- Row 11: ϵ∂J -> J, □J ---
calc_row("ϵ∂J", (st_curlJ, bs_curlJ), 3,
        1, [("J", bs_J, st_J), ("□J", bs_boxJ, st_boxJ)])
    
# --- Row 12: ϵ∂J -> ϵ∂∂J ---
calc_row("ϵ∂J", (st_curlJ, bs_curlJ), 3,
        2, [("ϵ∂∂J", bs_epsddJ, st_epsddJ)])

println("|----------|----|---------------|---------------|--------|")

# --- Row 13: J' -> ∂⋅J' ---
calc_row("J'", (st_Jprime, bs_Jprime), 3,
        0, [("∂⋅J'", bs_divJprime, st_divJprime)])
    
# --- Row 14: J' -> ϵ∂J' ---
calc_row("J'", (st_Jprime, bs_Jprime), 3,
        1, [("ϵ∂J'", bs_curlJprime, st_curlJprime)])
    
# --- Row 15: J' -> ∂J' ---
calc_row("J'", (st_Jprime, bs_Jprime), 3,
        2, [("∂J'", bs_dJprime, st_dJprime)])
    
println("|----------|----|---------------|---------------|--------|")