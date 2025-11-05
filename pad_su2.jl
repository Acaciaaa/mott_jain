module PADsu2

using FuzzifiED
using LinearAlgebra
using SpecialFunctions
using CairoMakie
using WignerSymbols
FuzzifiED.ElementType = Float64 
FuzzifiED.SilentStd = true
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
    tms_br_discrete
    tms_br_square
    tms_f123
    tms_V1
    tms_l2
    tms_c2
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
        PadQNDiag(GetFlavQNDiag(nm1, nf1, [1,-1, 0]), 0, no0)
    ]
    qnf = [
        PadQNOffd(GetFlavPermQNOffd(nm1, nf1, [2, 1, 3], [1, -1, 1]), 0, no0), 
        PadQNOffd(GetRotyQNOffd(nm1, nf1), 0, no0) * PadQNOffd(GetRotyQNOffd(nm0, 1), no1, 0)
    ]
    cfs = Confs(no, [no1, 0, 0], qnd)

    f0 = PadSphereObs(GetElectronObs(nm0, 1, 1), no1)
    f = [GetElectronObs(nm1, nf1, f) for f = 1 : nf1]
    n0 = f0' * f0 
    ne = GetDensityObs(nm1, nf1) + 3 * n0
    tms_hop = SimplifyTerms(GetIntegral(f0' * f[1] * f[2] * f[3]))
    tms_int = SimplifyTerms(GetIntegral(ne * ne))
    tms_V1 = SimplifyTerms(GetIntegral(n0 * Laplacian(n0)))
    tms_f123 = GetPolTerms(nm1, nf1)
    tms_br_discrete = GetPolTerms(nm1, nf1, diagm([1, 1, -2]))

    n1 = GetDensityObs(nm1, nf1, diagm([1, 0, 0]))
    n2 = GetDensityObs(nm1, nf1, diagm([0, 1, 0]))
    n3 = GetDensityObs(nm1, nf1, diagm([0, 0, 1]))
    tms_br_square = SimplifyTerms(GetIntegral(n1*n1 + n2*n2 - 2*n3*n3))

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

    return ModelParams(; name=:PADsu2, nm1, s, nf1, no1, nm0, nf0, no0, no, qnd, qnf, cfs, 
    tms_hop, tms_int, tms_br_discrete, tms_br_square, tms_f123, tms_V1, tms_l2, tms_c2)
end

make_tms_hmt(P::ModelParams, μ::Float64) = SimplifyTerms(
    0.5 * P.tms_int
    + 1.0 * P.tms_V1
    - 0.8 * (P.tms_hop + P.tms_hop')
    + μ * P.tms_f123
    #+ 0.05 * P.tms_br_discrete
    + 0.05 * P.tms_br_square
) 


function ground_state(P::ModelParams, μ::Float64, k::Int=1;
                            check_hermiticity::Bool=true,
                            tol_rel::Float64=1e-12,
                            symmetrize::Bool=true)
    tms_hmt = make_tms_hmt(P, μ)
    bestE = Inf; bestst = nothing; bestbs = nothing; bestR = 0; bestZ = 0
    for Z in (1,-1), R in (-1,1)
        bs = Basis(P.cfs, [Z, R], P.qnf)
        H  = Operator(bs, tms_hmt)

        A  = Matrix(OpMat(H))

        if check_hermiticity
            rel = opnorm(A - A') / max(opnorm(A), eps())
            @debug "μ=$μ (Z,R)=($Z,$R) rel_nonHerm=$rel"
            @assert rel ≤ max(tol_rel, 1e-14) "H not Hermitian enough: rel=$rel at (Z,R)=($Z,$R)"
        end

        A  = symmetrize ? (A + A')/2 : A
        vals, vecs = eigen(Hermitian(A))

        if vals[1] < bestE
            bestE, bestst, bestbs, bestR, bestZ = vals[1], vecs[:,1], bs, R, Z
        end
    end

    return bestst, bestbs, bestE, bestR, bestZ
end

function lowest_k_states(P::ModelParams, μ::Float64, k::Int=30)
    results = []
    for Z in (1,-1), R in (-1,1)
        bs = Basis(P.cfs, [Z, R], P.qnf)
        hmt_mat = OpMat(Operator(bs, make_tms_hmt(P, μ)))
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
            push!(results, round.([enrg[i], l2_val[i], c2_val[i]]; digits=7))
        end 
    end 
    sort!(results, by = st -> real(st[1]))
    return results
end

end