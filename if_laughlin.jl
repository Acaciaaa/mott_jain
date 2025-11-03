include(joinpath(@__DIR__, ".", "KLcommon.jl"))
include(joinpath(@__DIR__, ".", "JAINcommon.jl"))
include(joinpath(@__DIR__, ".", "pad_su3.jl"))
using .KLcommon
using .JAINcommon
using .PADsu3
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using SpecialFunctions 
using CairoMakie
using WignerSymbols

function make_V1_terms_pad(name, nm1::Int, nm0::Int; V::Float64=1.0, offset0::Int=0)
    base = offset0 + 3 * nm1
    nmf  = nm0

    twoS = nmf - 1
    if name == :V1
        L    = twoS - 1
    elseif name == :V3
        L = twoS - 3
    end

    two_m_of_k(k) = 2k - twoS

    buckets = Dict{Int, Vector{NTuple{3,Int}}}()
    coeffs  = Dict{Tuple{Int,Int}, Float64}()

    @inbounds for k1 in 0:nmf-1, k2 in 0:nmf-1
        tm1 = two_m_of_k(k1)
        tm2 = two_m_of_k(k2)
        twoM = tm1 + tm2
        if abs(twoM) > 2L
            continue
        end
        c = WignerSymbols.clebschgordan(Float64,
                twoS//2, tm1//2, twoS//2, tm2//2, L, twoM//2)
        if c == 0.0
            continue
        end
        push!(get!(buckets, twoM, NTuple{3,Int}[]), (k1, k2, twoM))
        coeffs[(k1,k2)] = c
    end

    tms = Term[]
    for (_twoM, pairs) in buckets
        @inbounds for a in pairs, b in pairs
            k1, k2 = a[1], a[2]
            k3, k4 = b[1], b[2]
            v = 0.5 * V * coeffs[(k1,k2)] * coeffs[(k3,k4)]
            if v != 0.0
                o1 = base + k1 + 1
                o2 = base + k2 + 1
                o3 = base + k3 + 1
                o4 = base + k4 + 1
                push!(tms, Term(v, [1,o1, 1,o2, 0,o4, 0,o3]))
            end
        end
    end

    return SimplifyTerms(tms)
end

function check_laughlin(name, bestst, bestbs, P; f_heavy::Int=4, tol0::Float64=1e-10)
    ψ = bestst ./ norm(bestst)
    # MV1 = Matrix(OpMat(Operator(bestbs, P.tms_V1))) 只有JAINcommon用
    tms_V = make_V1_terms_pad(name, P.nm1, P.nm0)
    MV = Matrix(OpMat(Operator(bestbs, tms_V)))
    
    E_V = real(dot(ψ, MV * ψ))

    vals, vecs = eigen(Hermitian(MV))
    Z = findall(v -> v < tol0, vals)  
    overlap2 = isempty(Z) ? 0.0 : sum(abs2, vecs[:,Z]' * ψ)

    return E_V, overlap2
end


#P = JAINcommon.build_model_su2u1(nml=4)
P = PADsu3.build_model(nm1=4)
#bestst, bestbs, bestE, bestR, bestZ = JAINcommon.ground_state_su2u1(P, 0.8, 0.05)
bestst, bestbs, bestE, bestR, bestZ = PADsu3.ground_state(P, 0.6)
E_V1, overlap2_1 = check_laughlin(:V1, bestst, bestbs, P)
E_V3, overlap2_3 = check_laughlin(:V3, bestst, bestbs, P)
println("⟨H_V1⟩ = ", E_V1)
println("⟨H_V3⟩ = ", E_V3)
println("overlap with ker(H_V1) = ", overlap2_1)
println("overlap with ker(H_V3) = ", overlap2_3)
