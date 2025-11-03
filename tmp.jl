using FuzzifiED
FuzzifiED.ElementType = Float64 
FuzzifiED.SilentStd = true
≈(x, y) = abs(x - y) < √(eps(Float64))

PadQNDiag(qnd :: QNDiag, nol :: Int64, nor :: Int64) = QNDiag(qnd.name, [ fill(0, nol) ; qnd.charge ; fill(0, nor) ], qnd.modul)
PadQNOffd(qnf :: QNOffd, nol :: Int64, nor :: Int64) = QNOffd(
    [ collect(1 : nol) ; qnf.perm .+ nol ; collect(1 : nor) .+ (nol + length(qnf.perm)) ], 
    [ fill(0, nol) ; qnf.ph ; fill(0, nor) ], 
    [ fill(ComplexF64(1), nol) ; qnf.fac ; fill(ComplexF64(1), nor) ], qnf.cyc)
PadTerm(tm :: Term, nol :: Int64) = Term(tm.coeff, [ isodd(i) ? tm.cstr[i] : tm.cstr[i] + nol for i in eachindex(tm.cstr)])
PadTerm(tms :: Terms, nol :: Int64) = PadTerm.(tms, nol)
PadSphereObs(obs :: SphereObs, nol :: Int64) = SphereObs(obs.s2, obs.l2m, (l, m) -> PadTerm(obs.get_comp(l, m), nol))

nm1 = 5
s = (nm1 - 1) / 2
nf1 = 3
no1 = nm1 * nf1 
nm0 = nm1 * 3 - 2 
no0 = nm0
no = no1 + no0

qnd = [
    PadQNDiag(GetNeQNDiag(no1), 0, no0) + 3 * PadQNDiag(GetNeQNDiag(no0), no1, 0),
    PadQNDiag(GetLz2QNDiag(nm1, nf1), 0, no0) + PadQNDiag(GetLz2QNDiag(nm0, 1), no1, 0),
    PadQNDiag(GetFlavQNDiag(nm1, nf1, [1,-1, 0]), 0, no0),
    PadQNDiag(GetFlavQNDiag(nm1, nf1, [1, 1,-2]), 0, no0)
]
qnf = [
    PadQNOffd(GetRotyQNOffd(nm1, nf1), 0, no0) * PadQNOffd(GetRotyQNOffd(nm0, 1), no1, 0),
    PadQNOffd(GetFlavPermQNOffd(nm1, nf1, [2, 1, 3], [1, -1, 1]), 0, no0)
]

FuzzifiED.ObsNormRadSq = nm1
f0 = PadSphereObs(GetElectronObs(nm0, 1, 1), no1)
f = [GetElectronObs(nm1, nf1, f) for f = 1 : nf1]
n0 = f0' * f0 
ne = GetDensityObs(nm1, nf1) + 3 * n0
tms_hop = SimplifyTerms(GetIntegral(f0' * f[1] * f[2] * f[3]))
tms_int_e = SimplifyTerms(GetIntegral(ne * ne))
tms_int_1 = SimplifyTerms(GetIntegral(n0 * Laplacian(n0)))
tms_hmt(U0, U1, t, μ) = SimplifyTerms(
    U0 * tms_int_e
    + U1 * tms_int_1
    - t * (tms_hop + tms_hop')
    + μ * GetPolTerms(nm1, nf1)
) 

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

cfs = Dict{Int64, Confs}() ; 
for Sz = 0 : 1
    cfs[Sz] = Confs(no, [no1, 0, 0, 6Sz], qnd) 
end

result = []
for Sz = 0 : 1, Z in [1, -1], R in [1, -1]
    bs = Basis(cfs[Sz], [R, Z], qnf)
    hmt = Operator(bs, tms_hmt(0.5, 1, 0.5, 0.085))
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, 10)

    l2 = Operator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ st[:, i]' * l2_mat * st[:, i] for i in eachindex(enrg)]

    c2 = Operator(bs, tms_c2)
    c2_mat = OpMat(c2)
    c2_val = [ st[:, i]' * c2_mat * st[:, i] for i in eachindex(enrg)]

    for i in eachindex(enrg)
        push!(result, vcat(round.([enrg[i], l2_val[i], c2_val[i]]; digits=7), Sz))
    end
end
sort!(result, by = st -> real(st[1])) 
enrg_0 = result[1][1]
enrg_1 = (filter(st -> st[2] ≈ 6 && st[3] ≈ 0, result)[1][1] - enrg_0) / 3
result_dim = [ [ (st[1] - enrg_0) / enrg_1 ; st] for st in result if st[2] < 12 ]
@info (permutedims(hcat(result_dim...)))