module PADsu3

using FuzzifiED
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using WignerSymbols
FuzzifiED.ElementType = Float64 
FuzzifiED.SilentStd = true
import Base: ≈
≈(x, y) = abs(x - y) < √(eps(Float64))


Base.@kwdef mutable struct ModelParams
    name::Symbol
    nm1::Int
    s
    nf1::Int
    no1::Int
    nm0::Int
    nf0::Int
    no0::Int
    no::Int
    
    qnd
    qnf
    cfs

    tms_hop
    tms_int
    tms_f123
    tms_V1
    tms_l2
    tms_c2
    n0
    nf
end

PadQNDiag(qnd :: QNDiag, nol :: Int64, nor :: Int64) = QNDiag(qnd.name, [ fill(0, nol) ; qnd.charge ; fill(0, nor) ], qnd.modul)
PadQNOffd(qnf :: QNOffd, nol :: Int64, nor :: Int64) = QNOffd(
    [ collect(1 : nol) ; qnf.perm .+ nol ; collect(1 : nor) .+ (nol + length(qnf.perm)) ], 
    [ fill(0, nol) ; qnf.ph ; fill(0, nor) ], 
    [ fill(ComplexF64(1), nol) ; qnf.fac ; fill(ComplexF64(1), nor) ], qnf.cyc)
PadTerm(tm :: Term, nol :: Int64) = Term(tm.coeff, [ isodd(i) ? tm.cstr[i] : tm.cstr[i] + nol for i in eachindex(tm.cstr)])
PadTerm(tms :: Terms, nol :: Int64) = PadTerm.(tms, nol)
PadSphereObs(obs :: SphereObs, nol :: Int64) = SphereObs(obs.s2, obs.l2m, (l, m) -> PadTerm(obs.get_comp(l, m), nol))

function build_model(; nm1::Int)
    s = (nm1 - 1) / 2
    nf1 = 3
    no1 = nm1 * nf1
    nm0 = nm1 * 3 - 2 
    nf0 = 1
    no0 = nm0
    no = no1 + no0
    FuzzifiED.ObsNormRadSq = nm1

    qnd = [
        PadQNDiag(GetNeQNDiag(no1), 0, no0) + 3 * PadQNDiag(GetNeQNDiag(no0), no1, 0),
        PadQNDiag(GetLz2QNDiag(nm1, nf1), 0, no0) + PadQNDiag(GetLz2QNDiag(nm0, 1), no1, 0),
        PadQNDiag(GetFlavQNDiag(nm1, nf1, [1,-1, 0]), 0, no0),
        PadQNDiag(GetFlavQNDiag(nm1, nf1, [1, 1,-2]), 0, no0)
    ]
    qnf = [
        PadQNOffd(GetFlavPermQNOffd(nm1, nf1, [2, 1, 3], [1, -1, 1]), 0, no0),
        PadQNOffd(GetRotyQNOffd(nm1, nf1), 0, no0) * PadQNOffd(GetRotyQNOffd(nm0, 1), no1, 0)
    ]
    cfs = Dict{Int64, Confs}()
    for Sz = 0 : 1
        cfs[Sz] = Confs(no, [no1, 0, 0, 6Sz], qnd) 
    end

    f0 = PadSphereObs(GetElectronObs(nm0, 1, 1), no1)
    f = [GetElectronObs(nm1, nf1, f) for f = 1 : nf1]
    n0 = f0' * f0 
    nf = GetDensityObs(nm1, nf1)
    ne = nf + 3 * n0
    tms_hop = SimplifyTerms(GetIntegral(f0' * f[1] * f[2] * f[3]))
    tms_int = SimplifyTerms(GetIntegral(ne * ne))
    tms_V1 = SimplifyTerms(GetIntegral(n0 * Laplacian(n0)))
    tms_f123 = GetPolTerms(nm1, nf1)

    tms_lz1 = 
        [ begin m = div(o - 1, nf1)
            Term(m - s, [1, o, 0, o])
        end for o = 1 : no1 ]
    tms_lp1 = 
        [ begin m = div(o - 1, nf1)
            Term(sqrt(m * (nm1 - m)), [1, o, 0, o - nf1])
        end for o = nf1 + 1 : no1 ]
    tms_lz0 = [ Term(m - 3s - 1, [1, m + no1, 0, m + no1]) for m = 1 : nm0 ]
    tms_lp0 = [ Term(sqrt((m - 1) * (nm0 - m + 1)), [1, m + no1, 0, m - 1 + no1]) for m = 2 : nm0 ]
    tms_lz = tms_lz1 + tms_lz0 
    tms_lp = tms_lp1 + tms_lp0 
    tms_lm = tms_lp'
    tms_l2 = SimplifyTerms(tms_lz * tms_lz - tms_lz + tms_lp * tms_lm)
    tms_c2 = GetC2Terms(nm1, nf1, :SU)
    
    return ModelParams(; name=:PADsu3, nm1, s, nf1, no1, nm0, nf0, no0, no, qnd, qnf, cfs, 
    tms_hop, tms_int, tms_f123, tms_V1, tms_l2, tms_c2, n0, nf)
end

function make_tms_hmt(
    P::ModelParams,
    μ::Float64,
    U0::Union{Float64,AbstractVector{<:Real}} = 0.5,
    V1::Float64 = 1.0,
    t::Float64 = 0.8,)
    other_terms = V1 * P.tms_V1 - t * (P.tms_hop + P.tms_hop') + μ * P.tms_f123
    if U0 isa Float64
        return SimplifyTerms(U0 * P.tms_int + other_terms)
    else
        Uf, Uf0, U0 = Float64.(U0)
        return SimplifyTerms(
            Uf * SimplifyTerms(GetIntegral(P.nf * P.nf)) +
            Uf0 * SimplifyTerms(GetIntegral(P.nf * P.n0)) +
            U0 * SimplifyTerms(GetIntegral(P.n0 * P.n0)) +
            other_terms
        )
    end
end

function ground_state(P::ModelParams, μ::Float64, U0, V1::Float64=1.0, t::Float64=0.8, k::Int=1;
                            check_hermiticity::Bool=true,
                            tol_rel::Float64=1e-12,
                            symmetrize::Bool=true)
    tms_hmt = make_tms_hmt(P, μ, U0, V1, t)
    bestE = Inf; bestst = nothing; bestbs = nothing; bestR = 0; bestZ = 0
    for Z in (1,-1), R in (-1,1)
        bs = Basis(P.cfs[0], [Z, R], P.qnf)
        hmt_mat = OpMat(Operator(bs, tms_hmt))
        vals, vecs = GetEigensystem(hmt_mat, k)

        if vals[1] < bestE
            bestE, bestst, bestbs, bestR, bestZ = vals[1], vecs[:,1], bs, R, Z
        end
    end

    return bestst, bestbs, bestE, bestR, bestZ
end

function lowest_k_states(P::ModelParams, μ::Float64, U0, V1::Float64=1.0, t::Float64=0.8, k::Int=30)
    results = []
    #for Sz in 0:1, Z in (1, -1), R in (1, -1) 
    for Sz in (0), Z in (1, -1), R in (1, -1) 
        bs = Basis(P.cfs[Sz], [Z, R], P.qnf)
        hmt_mat = OpMat(Operator(bs, PADsu3.make_tms_hmt(P, μ, U0, V1, t)))
        n = hmt_mat.dimd
        
        if n == 0
            continue
        elseif n == 1
            hmatrix = Matrix(hmt_mat)
            enrg = [hmatrix[1,1]]
            st   = ones(eltype(hmatrix), 1, 1)
        elseif n ≤ 128
            hmatrix = Matrix(hmt_mat)
            vals, vecs = eigen(hmatrix)
            k_req =  min(k, n)
            idx   =  sortperm(vals)[1:k_req]
            enrg, st = vals[idx], vecs[:, idx]
        else
            enrg, st = GetEigensystem(hmt_mat, k)
        end
        l2_mat = OpMat(Operator(bs, P.tms_l2)) 
        c2_mat = OpMat(Operator(bs, P.tms_c2)) 
        l2_val = [real(st[:, i]' * l2_mat * st[:, i]) for i in eachindex(enrg)] 
        c2_val = [real(st[:, i]' * c2_mat * st[:, i]) for i in eachindex(enrg)]
        for i in eachindex(enrg) 
            #push!(results, (E=enrg[i], L2=l2_val[i], C2=c2_val[i], Sz=Sz, R=R, Z=Z, i=i))
            push!(results, vcat(round.([enrg[i], l2_val[i], c2_val[i]]; digits=7), Sz))
        end 
    end 
    sort!(results, by = st -> real(st[1]))
    return results
end

function for_generator(P::ModelParams, μ::Float64, U0, V1::Float64=1.0, t::Float64=0.8, k::Int=10)
    results = []
    bss = Dict{Vector{Int64}, Basis}()
    for Sz in (0), Z in (1, -1), R in (1, -1) 
        bs = Basis(P.cfs[Sz], [Z, R], P.qnf)
        hmt_mat = OpMat(Operator(bs, PADsu3.make_tms_hmt(P, μ, U0, V1, t)))
        n = hmt_mat.dimd
        
        if n == 0
            continue
        elseif n == 1
            hmatrix = Matrix(hmt_mat)
            enrg = [hmatrix[1,1]]
            st   = ones(eltype(hmatrix), 1, 1)
        elseif n ≤ 128
            hmatrix = Matrix(hmt_mat)
            vals, vecs = eigen(hmatrix)
            k_req =  min(k, n)
            idx   =  sortperm(vals)[1:k_req]
            enrg, st = vals[idx], vecs[:, idx]
        else
            enrg, st = GetEigensystem(hmt_mat, k)
        end
        l2_mat = OpMat(Operator(bs, P.tms_l2)) 
        c2_mat = OpMat(Operator(bs, P.tms_c2)) 
        l2_val = [real(st[:, i]' * l2_mat * st[:, i]) for i in eachindex(enrg)] 
        c2_val = [real(st[:, i]' * c2_mat * st[:, i]) for i in eachindex(enrg)]
        bss[[Sz, Z, R]] = bs
        for i in eachindex(enrg) 
            push!(results, [enrg[i], st[:,i], l2_val[i], c2_val[i], Sz, Z, R])
        end 
    end 
    sort!(results, by = st -> real(st[1]))
    return results, bss
end

using JLD2, Dates
function write_results(P::ModelParams, mus, U0, V1::Float64=1.0, t::Float64=0.8,
                        k::Int=30, path::AbstractString = "data/results_$(P.nm1).jld2")
    mkpath(dirname(path))
    mus_vec = collect(round.(Float64.(mus); digits=4))
    results_vec = Vector{Vector{Vector{Float64}}}(undef, length(mus_vec))
    for (i, μ) in enumerate(mus_vec)
        results_vec[i] = lowest_k_states(P, μ, U0, V1, t,k)
    end
    meta = Dict{String,Any}("nml"=>P.nm1, "k"=>k, "count_mu"=>length(mus_vec), "updated_at"=>string(Dates.now()))
    @save path mus=mus_vec results=results_vec meta
    println("✅ Saved $(length(mus_vec)) μ values (rounded to 4 decimals) to $path")
end

function read_results(nm1, path::AbstractString = "data/results_$(nm1).jld2")
    @assert isfile(path) "未找到文件"
    obj = JLD2.load(path)
    @assert haskey(obj, "mus") "文件缺少 'mus' 字段"
    @assert haskey(obj, "results") "文件缺少 'results' 字段"
    mus         = obj["mus"]::Vector{Float64}
    results_vec = obj["results"]::Vector{Vector{Vector{Float64}}}
    @assert length(mus) == length(results_vec) "文件损坏 mus 与 results 长度不一致"
    return mus, results_vec
end


function lowest_k_states_adjoint(P::ModelParams, μ::Float64, U0, V1::Float64=1.0, t::Float64=0.8, k::Int=30)
    results = []
    #for Sz in 0:1, Z in (1, -1), R in (1, -1) 
    for Sz in (1), Z in (1, -1)
        bs = Basis(Confs(P.no, [P.no1, 0, 2, 0], P.qnd), [Z], 
            [PadQNOffd(GetRotyQNOffd(P.nm1, P.nf1), 0, P.no0) * PadQNOffd(GetRotyQNOffd(P.nm0, 1), P.no1, 0)])
        hmt_mat = OpMat(Operator(bs, PADsu3.make_tms_hmt(P, μ, U0, V1, t)))
        n = hmt_mat.dimd
        
        if n == 0
            continue
        elseif n == 1
            hmatrix = Matrix(hmt_mat)
            enrg = [hmatrix[1,1]]
            st   = ones(eltype(hmatrix), 1, 1)
        elseif n ≤ 128
            hmatrix = Matrix(hmt_mat)
            vals, vecs = eigen(hmatrix)
            k_req =  min(k, n)
            idx   =  sortperm(vals)[1:k_req]
            enrg, st = vals[idx], vecs[:, idx]
        else
            enrg, st = GetEigensystem(hmt_mat, k)
        end
        l2_mat = OpMat(Operator(bs, P.tms_l2)) 
        c2_mat = OpMat(Operator(bs, P.tms_c2)) 
        l2_val = [real(st[:, i]' * l2_mat * st[:, i]) for i in eachindex(enrg)] 
        c2_val = [real(st[:, i]' * c2_mat * st[:, i]) for i in eachindex(enrg)]
        for i in eachindex(enrg) 
            #push!(results, (E=enrg[i], L2=l2_val[i], C2=c2_val[i], Sz=Sz, R=R, Z=Z, i=i))
            push!(results, vcat(round.([enrg[i], l2_val[i], c2_val[i]]; digits=7), Sz))
        end 
    end 
    sort!(results, by = st -> real(st[1]))
    return results
end
end