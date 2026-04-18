include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .PADsu3
using FuzzifiED
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using Printf
using JLD2, Dates
using DataFrames, CSV
using XLSX
TOL= √(eps(Float64))
# coef = [0.4,0.9,0.8]
# file_path = "../c"
# nm1 = 6
# factor = 0.088
# coef = [0.4,0.9,0.3]
# file_path = "../a"
# nm1 = 7
# factor = 0.0692
file_path = "/Users/ruiqi/Desktop"
coef = [[0.5, 1.771, 4.5],[0.0, 0.395, 1.0],0.5]
nm1 = 5
μc=0.064
factor = 0.042

P = PADsu3.build_model(nm1=nm1)
# data = load(joinpath(file_path, "generator_$(nm1).jld2"))
# results = data["results"]
# bss = data["bss"]
results, bss = PADsu3.for_generator(P, μc, coef[1], coef[2], coef[3], 200)

mutable struct StateInfo
    label
    name
    l2::Int
    c2::Int
    rank::Int
    st
    bs
    E
    st_o
    E_o
end

struct StateStore
    by_label::Dict{Symbol, StateInfo}
    function StateStore()
        new(Dict{Symbol, StateInfo}())
    end
end
function add_state!(store::StateStore, label, name, l2, c2, rank, if_othersector)
    scalar_info = wanted_states[(l2, c2)][rank]
    qn = [scalar_info[5], scalar_info[6], scalar_info[7]]
    if if_othersector
        adjoint_info = wanted_states_othersector[(l2, c2)][rank]
        state_info = StateInfo(label, name, l2, c2, rank, scalar_info[2], bss[qn], scalar_info[1], 
                    adjoint_info[2], adjoint_info[1])
    else
        state_info = StateInfo(label, name, l2, c2, rank, scalar_info[2], bss[qn], scalar_info[1], 
                    nothing, nothing)
    end
    store.by_label[label] = state_info
end
function make_lambda_candidates()
    s=P.s
    # --- A. 小费米子算符 ---
    cf = [GetElectronMod(P.nm1, P.nf1, i) for i in 1:3]

    # --- B. 大费米子算符 ---
    c0 = PadAngModes(GetElectronMod(P.nm0, P.nf0, 1), P.no1)

    # --- A. f-f 配对 ---
    Delta_ff_Uf = Dict()
    Delta_ff_Vf = Dict()
    for i in 1:3, j in 1:3
        if i < j
            Delta_ff_Uf[(i,j)] = FilterL2(cf[i] * cf[j], 2 * s) 
            Delta_ff_Vf[(i,j)] = FilterL2(cf[i] * cf[j], 2 * s - 2)
        end
    end

    # --- B. f0-f0 配对 ---
    pair_00 = c0 * c0
    Delta_0_U0 = FilterL2(pair_00, 6 * s - 1)
    Delta_0_V0 = FilterL2(pair_00, 6 * s - 3)

    # --- C. f0-f 混合配对 ---
    Delta_0f_Uf0 = [FilterL2(c0 * cf[i], 4 * s) for i in 1:3]
    Delta_0f_Vf0 = [FilterL2(c0 * cf[i], 4 * s - 2) for i in 1:3]

    # --- D. 三体算符 Trion ---
    pair_12_2s = FilterL2(cf[1] * cf[2], 2 * s)
    Trion = FilterL2(pair_12_2s * cf[3], 3 * s)

    # ==========================================
    # 4. 组装生成元 Lambda (L=1 投影)
    # ==========================================
    tms_cand = Vector{Terms}()

    # 1. U_f 项
    for i in 1:3, j in (i+1):3
        term = Delta_ff_Uf[(i,j)]' * Delta_ff_Uf[(i,j)]
        push!(tms_cand, GetComponent(FilterL2(term, 1), 1, 0))
    end

    # 2. V_f 项
    for i in 1:3, j in (i+1):3
        term = Delta_ff_Vf[(i,j)]' * Delta_ff_Vf[(i,j)]
        push!(tms_cand, GetComponent(FilterL2(term, 1), 1, 0))
    end

    # 3. U_0 项
    term_U0 = Delta_0_U0' * Delta_0_U0
    push!(tms_cand, GetComponent(FilterL2(term_U0, 1), 1, 0))

    # 4. V_0 项
    term_V0 = Delta_0_V0' * Delta_0_V0
    push!(tms_cand, GetComponent(FilterL2(term_V0, 1), 1, 0))

    # 5. U_f0 项
    for i in 1:3
        term = Delta_0f_Uf0[i]' * Delta_0f_Uf0[i]
        push!(tms_cand, GetComponent(FilterL2(term, 1), 1, 0))
    end

    # 6. V_f0 项
    for i in 1:3
        term = Delta_0f_Vf0[i]' * Delta_0f_Vf0[i]
        push!(tms_cand, GetComponent(FilterL2(term, 1), 1, 0))
    end

    # 7. 隧穿项 t
    term_t = c0' * Trion + Trion' * c0
    push!(tms_cand, GetComponent(FilterL2(term_t, 1), 1, 0))

    # 8. 化学势项 mu
    for i in 1:3
        term_mu = cf[i]' * cf[i]
        push!(tms_cand, GetComponent(FilterL2(term_mu, 1), 1, 0))
    end
    return tms_cand
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

    idx = 1

    println("--- 1. U_f (s-wave) ---")
    # 对应循环: for i in 1:3, j in (i+1):3
    for i in 1:3, j in (i+1):3
        @printf "  U_f (%d,%d) : %10.6f\n" i j abs(coeffs[idx])
        global idx += 1
    end

    println("--- 2. V_f (d-wave, 新增) ---")
    for i in 1:3, j in (i+1):3
        @printf "  V_f (%d,%d) : %10.6f\n" i j abs(coeffs[idx])
        global idx += 1
    end

    println("--- 3. U_0 (p-wave) ---")
    @printf "  U_0        : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 4. V_0 (f-wave, 原 V_1) ---")
    @printf "  V_0        : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 5. U_0f (s-wave) ---")
    for i in 1:3
        @printf "  U_0f (%d)   : %10.6f\n" i abs(coeffs[idx])
        global idx += 1
    end

    println("--- 6. V_f0 (d-wave, 新增) ---")
    for i in 1:3
        @printf "  V_f0 (%d)   : %10.6f\n" i abs(coeffs[idx])
        global idx += 1
    end

    println("--- 7. t (Tunneling) ---")
    @printf "  t          : %10.6f\n" abs(coeffs[idx])
    global idx += 1

    println("--- 8. mu (Chemical Potential) ---")
    for i in 1:3
        @printf "  mu (%d)     : %10.6f\n" i abs(coeffs[idx])
        global idx += 1
    end

    println("================================")
end
function print_full_spectrum(filename=joinpath(file_path, "spectrum_$(nm1).xlsx"))
    sectors_order = [(0, 0),(2, 0),(6, 0),(0, 3),(2, 3),(6, 3)]
    lookup_map = Dict{Tuple{Int, Int, Int}, String}()
    for info in values(db.by_label)
        lookup_map[(info.l2, info.c2, info.rank)] = info.name
    end

    max_rows = 0
    for (l2, c2) in sectors_order
    # 使用 get 函数，如果没有这个 key，就返回空数组 []
    # 这样 length([]) 就是 0，不会报错
        len1 = length(get(wanted_states, (l2, c2), []))
        len2 = length(get(wanted_states_othersector, (l2, c2), []))
        max_rows = max(max_rows, len1, len2)
    end

    df = DataFrame()
    for (l2, c2) in sectors_order
        sector_data = wanted_states[(l2, c2)]
        col_E = Vector{Union{Float64, Missing}}(missing, max_rows)
        col_Name = Vector{String}(undef, max_rows)
        fill!(col_Name, "")

        for r in eachindex(sector_data)
            col_E[r] = round((sector_data[r][1]-E0)/factor, digits=2)
            if haskey(lookup_map, (l2, c2, r))
                col_Name[r] = lookup_map[(l2, c2, r)]
            else
                col_Name[r] = ""
            end
        end

        df[!, "E_$(l2)_$(c2)"] = col_E
        df[!, "OP_$(l2)_$(c2)"] = col_Name
    end

    XLSX.writetable(filename, "Result" => df, overwrite=true)
end
function test_capacity(filename=joinpath(file_path, "capacity_$(nm1).xlsx"))
    sectors_order = [(0, 0),(2, 0),(6, 0),(0, 3),(2, 3),(6, 3)]
    othersectors_order = [(0, 3),(2, 3),(6, 3)]
    max_rows = 0
    for (l2, c2) in sectors_order
        len1 = length(get(wanted_states, (l2, c2), []))
        len2 = length(get(wanted_states_othersector, (l2, c2), []))
        max_rows = max(max_rows, len1, len2)
    end

    df = DataFrame()
    for (l2, c2) in sectors_order
        sector_data = wanted_states[(l2, c2)]        
        col_E = Vector{Union{Float64, Missing}}(missing, max_rows)
        for r in eachindex(sector_data)
            col_E[r] = round((sector_data[r][1]-E0)/factor, digits=2)
        end
        df[!, "E_$(l2)_$(c2)"] = col_E
    end
    for (l2, c2) in othersectors_order
        sector_data = wanted_states_othersector[(l2, c2)]        
        col_E = Vector{Union{Float64, Missing}}(missing, max_rows)
        for r in eachindex(sector_data)
            col_E[r] = round((sector_data[r][1]-E0)/factor, digits=2)
        end

        df[!, "Eo_$(l2)_$(c2)"] = col_E
    end

    XLSX.writetable(filename, "Result" => df, overwrite=true)
end
function test_adjoint_scalar(l0=1)
    T3 = zeros(ComplexF64, 3, 3)
    T3[1,1] =  0.5
    T3[2,2] = -0.5
    J3  = GetDensityObs(P.nm1, 3, T3)
    J3l = FilterComponent(J3, (l,m) -> l == l0)
    JJ  = J3l * J3l 
    
    # T8 = zeros(ComplexF64, 3, 3)
    # T8[1,1] =  1/(2*sqrt(3))
    # T8[2,2] =  1/(2*sqrt(3))
    # T8[3,3] = -1/sqrt(3)
    # J8  = GetDensityObs(P.nm1, 3, T8)
    # J8l = FilterComponent(J8, (l,m) -> l == l0)
    # JJ  = J8l * J8l

    terms = GetComponent(JJ, 0, 0)             
    bs_G, st_G = db.by_label[:G].bs, db.by_label[:G].st
    bs_O, st_O = db.by_label[:O].bs, db.by_label[:O].st
    op = Operator(bs_G, bs_O, terms)
    amp = st_O' * op * st_G           
    og  = op * st_G                  
    return abs(amp) / (sqrt(real(og' * og)) + 1e-300)
end

E0 = results[1][1]
wanted_states = Dict{Tuple{Int, Int}, Any}()
wanted_states_othersector = Dict{Tuple{Int, Int}, Any}()
for (l2, c2) in [(0,0), (2,0), (6,0)]
    wanted_states[(l2, c2)] = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results)
end

db = StateStore()
add_state!(db, :G, "G", 0, 0, 1, false)
add_state!(db, :Sprime, "S'", 0, 0, 4, false)
add_state!(db, :dSprime, "∂S'", 2, 0, 3, false) #ab是3; c是5
#add_state!(db, :dSprime_mix, "∂S'_mix", 2, 0, 5, false)

add_state!(db, :S, "S", 0, 0, 2, false)
add_state!(db, :boxS, "□S", 0, 0, 3, false)
add_state!(db, :dS, "∂S", 2, 0, 1, false)
add_state!(db, :boxdS, "□∂S", 2, 0, 2, false)
add_state!(db, :ddS, "∂∂S", 6, 0, 2, false)

add_state!(db, :T, "T", 6, 0, 1, false)

# data = load(joinpath(file_path, "generator_othersector_$(nm1).jld2"))
# results_othersector = data["results"]
# bs0, bsp, bsm = data["bs0"], data["bsp"], data["bsm"]
results_othersector, bs0, bsp, bsm = PADsu3.for_generator_special(P, μc,coef[1], coef[2], coef[3], 400)
for (l2, c2) in [(0,3), (2,3), (6,3)]
    wanted_states_othersector[(l2, c2)] = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results_othersector)
    wanted_states[(l2, c2)] = []
    candidates_Z1 = []
    energies_Z2   = Float64[]
    tmps = filter(st -> abs(st[3]-l2) < TOL && abs(st[4]-c2) < TOL, results)
    for tmp in tmps
        z_val = tmp[6]
        if abs(z_val - 1.0) < TOL      # Z=+1 Sector
            push!(candidates_Z1, tmp)
        elseif abs(z_val + 1.0) < TOL  # Z=-1 Sector
            push!(energies_Z2, tmp[1])   # 我们只关心 Z=-1 的能量，不用存向量
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

add_state!(db, :O, "O", 0, 3, 1, false)
add_state!(db, :boxO, "□O", 0, 3, 2, false)
add_state!(db, :dO, "∂O", 2, 3, 5, false)
add_state!(db, :ddO, "∂∂O", 6, 3, 13, false) #找不着 这是暂时最好的了;

add_state!(db, :divJprime, "∂⋅J'", 0, 3, 3, false)
add_state!(db, :Jprime, "J'", 2, 3, 6, true) #ab是6; c是4
add_state!(db, :curlJprime, "ϵ∂J'", 2, 3, 10, true) #a是10; b是11; c找不着
add_state!(db, :dJprime, "∂J'", 6, 3, 12, false) #a是12; b是14且有14 15mixing; c找不着

add_state!(db, :J, "J", 2, 3, 1, true)
add_state!(db, :curlJ, "ϵ∂J", 2, 3, 2, true)
add_state!(db, :boxJ, "□J", 2, 3, 3, true) #有3 4mixing但是不知道怎么处理; b一样
add_state!(db, :dJ, "∂J", 6, 3, 1, true)
#add_state!(db, :dJ_mix, "∂J_mix", 6, 3, 2, true)
add_state!(db, :epsddJ, "ϵ∂∂J", 6, 3, 3, true) #根据前者有3 4或3 5mixing; b一样
#add_state!(db, :epsddJ_mix, "ϵ∂∂J_mix", 6, 3, 4, true)


tms_cand = make_lambda_candidates()
function calculate_coeffs(filename=coeffs_filename)
    vecs_op = []
    bs_S, st_S = db.by_label[:S].bs, db.by_label[:S].st
    bs_dS, st_dS = db.by_label[:dS].bs, db.by_label[:dS].st
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
    jldsave(filename; coeffs)
end

coeffs_filename = joinpath(file_path, "coeffs_$(nm1).jld2")
calculate_coeffs()
coeffs = load(coeffs_filename)["coeffs"]
#print_coefficients()

tms_Lambdaz = SimplifyTerms(coeffs' * tms_cand)
function remove_level_mixing(label_in, target, mix)
    in_info, target_info, mix_info = db.by_label[label_in], db.by_label[target], db.by_label[mix]
    st_in, bs_in = in_info.st, in_info.bs
    st_1, bs_1 = target_info.st, target_info.bs
    st_2, bs_2 = mix_info.st, mix_info.bs
    op = Operator(bs_in, bs_1, tms_Lambdaz)
    v = op * st_in

    c1 = dot(st_1, v)
    c2 = dot(st_2, v)

    target_info.st = normalize!(c1 .* st_1 .+ c2 .* st_2)
    mix_info.st = normalize!(c2 .* st_1 .- c1 .* st_2)
    
    if !isnothing(target_info.st_o)
        st_o_in = in_info.st_o
        st_o_1 = target_info.st_o
        st_o_2 = mix_info.st_o
        op = Operator(bs0, bs0, tms_Lambdaz)
        v_o = op * st_o_in

        c1_o = dot(st_o_1, v_o)
        c2_o = dot(st_o_2, v_o)
    
        target_info.st_o = normalize!(c1_o .* st_o_1 .+ c2_o .* st_o_2)
        mix_info.st_o = normalize!(c2_o .* st_o_1 .- c1_o .* st_o_2)
    end 
end
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
function calc_row(label_in, l_input, l_target, targets_list)
    info_in = db.by_label[label_in]
    st_in, bs_in, E_in = info_in.st, info_in.bs, info_in.E
    bs_tgt = db.by_label[targets_list[1]].bs # Use basis of first target
    op = Operator(bs_in, bs_tgt, tms_Lambdaz)
    v = op * st_in
    v_proj = project_state(v, Operator(bs_tgt, bs_tgt, P.tms_l2), l_target, get_possible_ls(l_input))
    denom_sq = Real(dot(v_proj, v_proj))
    @assert denom_sq>0.1 "denom_sq invalid"
    #println("denom_sq:$denom_sq")
    @printf "| %-6s | %-2d |" info_in.name l_target
    total_ovlp = 0.0
    
    for label_out in targets_list
        info_out = db.by_label[label_out]
        st_out, name_out, E_out = info_out.st, info_out.name, info_out.E
        val = abs(dot(st_out, v))^2 / denom_sq
        @printf " %-12s %.4f |" @sprintf("%s(%.2f)", name_out, (E_out-E_in)/factor) val
        total_ovlp += val
    end
    
    if length(targets_list) < 2
        print("          -          |")
    end
    @printf " %.4f |\n" total_ovlp
end
function calc_row_sameang(label_in, l_input, l_target, targets_list)
    info_in = db.by_label[label_in]
    st_in, E_in = info_in.st_o, info_in.E_o
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
    
    @printf "| %-6s | %-2d |" info_in.name l_target
    total_ovlp = 0.0
    
    for label_out in targets_list
        info_out = db.by_label[label_out]
        st_out, name_out, E_out = info_out.st_o, info_out.name, info_out.E_o
        val = abs(dot(st_out, v))^2 / denom_sq
        @printf " %-12s %.4f |" @sprintf("%s(%.2f)", name_out, (E_out-E_in)/factor) val
        total_ovlp += val
    end
    
    if length(targets_list) < 2
        print("          -          |")
    end
    @printf " %.4f |\n" total_ovlp
end

# remove level mixing
remove_level_mixing(:dS, :ddS, :T)
#remove_level_mixing(:J, :dJ, :dJ_mix)
#remove_level_mixing(:Sprime, :dSprime, :dSprime_mix)
#remove_level_mixing(:curlJ, :epsddJ, :epsddJ_mix)
#remove_level_mixing(:dJ, :boxJ, :Jprime)

println("| Input  | l' | Target 1 Ovlp       | Target 2 Ovlp       | Total  |")
println("|--------|----|---------------------|---------------------|--------|")

tasks = [
    # --- Row 1: S -> ∂S ---
    () -> calc_row(:S, 0, 1, [:dS]),

    # --- Row 2: ∂S -> S, □S ---
    () -> calc_row(:dS, 1, 0, [:S, :boxS]),

    # --- Row 3: ∂S -> ∂∂S(*) ---
    () -> calc_row(:dS, 1, 2, [:ddS]),

    # --- Row 4: □S -> ∂S, □∂S ---
    () -> calc_row(:boxS, 0, 1, [:dS, :boxdS]),

    # --- Row 5: ∂∂S(*) -> ∂S, □∂S ---
    () -> calc_row(:ddS, 2, 1, [:dS, :boxdS]),

    () -> println("|--------|----|---------------------|---------------------|--------|"),

    # --- Row 6: S' -> ∂S' ---
    () -> calc_row(:Sprime, 0, 1, [:dSprime]),

    () -> println("|--------|----|---------------------|---------------------|--------|"),
    
    # --- Row 7: J -> ϵ∂J (same ang!) ---
    () -> calc_row_sameang(:J, 1, 1, [:curlJ]),

    # --- Row 8: J -> ∂J ---
    () -> calc_row(:J, 1, 2, [:dJ]),
    #() -> calc_row(:J, 1, 0, [:O]),
        
    # --- Row 9: ∂J -> J, □J ---
    () -> calc_row(:dJ, 2, 1, [:J, :boxJ]),
        
    # --- Row 10: ∂J -> ϵ∂∂J (same ang!) ---
    () -> calc_row_sameang(:dJ, 2, 2, [:epsddJ]),
        
    # --- Row 11: ϵ∂J -> J, □J (same ang!) ---
    () -> calc_row_sameang(:curlJ, 1, 1, [:J, :boxJ]),
        
    # --- Row 12: ϵ∂J -> ϵ∂∂J ---
    () -> calc_row(:curlJ, 1, 2, [:epsddJ]),

    () -> println("|--------|----|---------------------|---------------------|--------|"),

    # --- Row 13: J' -> ∂⋅J' ---
    () -> calc_row(:Jprime, 1, 0, [:divJprime]),
        
    # --- Row 14: J' -> ϵ∂J' (same ang!) ---
    () -> calc_row_sameang(:Jprime, 1, 1, [:curlJprime]),
        
    # --- Row 15: J' -> ∂J' ---
    () -> calc_row(:Jprime, 1, 2, [:dJprime]),

    () -> println("|--------|----|---------------------|---------------------|--------|"),

    # --- Row 16: O -> ∂O ---
    () -> calc_row(:O, 0, 1, [:dO]),

    # --- Row 17: ∂O -> O, □O ---
    () -> calc_row(:dO, 1, 0, [:O, :boxO]),

    # --- Row 18: ∂O -> ∂∂O ---
    () -> calc_row(:dO, 1, 2, [:ddO]),
]

run_indices = []
indices_to_run = isempty(run_indices) ? (1:length(tasks)) : run_indices
for i in indices_to_run
    tasks[i]()
end
println("|--------|----|---------------------|---------------------|--------|")

function check_current_conservation()
    bs_J, st_J = db.by_label[:J].bs, db.by_label[:J].st
    bs_dJ = db.by_label[:dJ].bs
    op = Operator(bs_J, bs_dJ, tms_Lambdaz)
    v = op * st_J
    v_proj = project_state(v, Operator(bs_dJ, bs_dJ, P.tms_l2), 0, get_possible_ls(1))
    println("current conservation: $(Real(dot(v_proj, v_proj))/Real(dot(v, v)))")
end
function check_tensor_conservation()
    raw_info_T = wanted_states[(6, 0)][1]
    bs_after = bss[[raw_info_T[5], raw_info_T[6], -raw_info_T[7]]]
    bs_T, st_T = db.by_label[:T].bs, db.by_label[:T].st
    op = Operator(bs_T, bs_after, tms_Lambdaz)
    v = op * st_T
    v_proj = project_state(v, Operator(bs_after, bs_after, P.tms_l2), 1, get_possible_ls(2))
    println("tensor conservation: $(Real(dot(v_proj, v_proj))/Real(dot(v, v)))")
end
check_current_conservation()
check_tensor_conservation()
# print_full_spectrum()
# test_capacity()
#@info test_adjoint_scalar()
function check_current_conservation_withO()
    info_J = db.by_label[:J]
    info_O = db.by_label[:O]
    bs_J, st_J = info_J.bs, info_J.st
    bs_O, st_O = info_O.bs, info_O.st
    
    op = Operator(bs_J, bs_O, tms_Lambdaz)
    v = op * st_J
    v_proj = project_state(v, Operator(bs_O, bs_O, P.tms_l2), 0, get_possible_ls(1))
    
    norm_total = Real(dot(v, v))
    norm_l0 = Real(dot(v_proj, v_proj))
    
    println("=== 守恒流有限尺寸误差与 O 态 Mixing 分析 ===")
    @printf "1. 原始守恒律误差 (Norm of l=0 / Total Norm): %.6f\n" (norm_l0 / norm_total)
    
    overlap = dot(st_O, v_proj)
    norm_O = dot(st_O, st_O)
    weight_O = abs2(overlap) / (norm_O * norm_l0)
    @printf "2. 误差向量中 O 态的成分占比: %.2f%%\n" (weight_O * 100)
    
    v_clean = v_proj .- (overlap / norm_O) .* st_O
    norm_clean = Real(dot(v_clean, v_clean))
    
    @printf "3. 剔除 O 态混合后的纯净误差 (Clean Norm / Total Norm): %.6f\n" (norm_clean / norm_total)
    println("==================================================")
end
check_current_conservation_withO()
function check_generator_components()
    info_J = db.by_label[:J]
    st_J, bs_J = info_J.st, info_J.bs
    
    bs_tgt = db.by_label[:O].bs 
    op = Operator(bs_J, bs_tgt, tms_Lambdaz)
    op_L2 = Operator(bs_tgt, bs_tgt, P.tms_l2)
    
    v = op * st_J
    norm_total_sq = Real(dot(v, v))
    @printf "原始向量ΛJ总模长平方: %.6e\n" norm_total_sq
    
    all_ls = [0, 1, 2]
    v_l0 = project_state(v, op_L2, 0, all_ls)
    norm_l0_sq = Real(dot(v_l0, v_l0))
    ratio_l0 = norm_l0_sq / norm_total_sq
    @printf "l=0 分量模长平方: %.6e  占比: %.4f%%\n" norm_l0_sq (ratio_l0 * 100)
    
    # 4. 投影到 l=2 扇区 (合法的物理后代态部分)
    v_l2 = project_state(v, op_L2, 2, all_ls)
    norm_l2_sq = Real(dot(v_l2, v_l2))
    ratio_l2 = norm_l2_sq / norm_total_sq
    @printf "l=2 分量模长平方: %.6e  占比: %.4f%%\n" norm_l2_sq (ratio_l2 * 100)
end
check_generator_components()