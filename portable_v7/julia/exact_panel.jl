# Exact reference for the external V7 panel, computed in QuantumOptics.jl -- a separate code path from the
# Python/QuTiP exact solve in pv7. OPTIONAL (Level 2 in docs/EXTERNAL_V7_REPRODUCTION_PROTOCOL.md).
#
# Input: external_output/panel_params_for_julia.json, written by run_external_v7_reproduction.py with the disorder
# that run drew. Output: external_output/julia_exact.csv (cell, M, seed, kappa, n_peak_exact, t_peak_exact).
# The Hamiltonian, dissipators, initial state and time grid are those of replicate_cumulant.jl (function exact_na,
# copied unchanged apart from reading the cell dict): H = wc a'a + sum_i w_i |e><e|_i + sum_i g_i (a s+_i + a' s-_i);
# kappa D[a] + gdown D[s-_i] + (gphi/2) D[sz_i]; vacuum x all excited; t in [0, 40], 200 points.
#
# Run (from the repository root, using its Julia environment):
#   julia --project=. portable_v7/julia/exact_panel.jl portable_v7/external_output/panel_params_for_julia.json
using QuantumOptics, JSON

const TMAX = 40.0
const NT   = 200
const NCUT = 16
const FOCK_GUARD = 0.45
tlist() = collect(range(0.0, TMAX; length = NT))

function exact_na(c)
    M = Int(c["M"]); w = Float64.(c["w"]); g = Float64.(c["g"])
    wc = c["wc"]; kappa = c["kappa"]; gdn = c["gdown"]; gph = c["gphi"]
    bc = FockBasis(NCUT)
    ba = SpinBasis(1//2)                         # spinup = EXCITED, spindown = ground
    b  = tensor(bc, fill(ba, M)...)
    a  = embed(b, 1, destroy(bc))
    sp = [embed(b, k + 1, sigmap(ba)) for k in 1:M]
    sm = [embed(b, k + 1, sigmam(ba)) for k in 1:M]
    sz = [embed(b, k + 1, sigmaz(ba)) for k in 1:M]
    pe = [0.5 * (identityoperator(b) + sz[k]) for k in 1:M]
    H = wc * (dagger(a) * a)
    for k in 1:M
        H += w[k] * pe[k] + g[k] * (a * sp[k] + dagger(a) * sm[k])
    end
    J = Any[a]; rates = Float64[kappa]
    for k in 1:M
        push!(J, sm[k]); push!(rates, gdn)
        push!(J, sz[k]); push!(rates, gph / 2)
    end
    psi0 = tensor(fockstate(bc, 0), fill(spinup(ba), M)...)
    nop  = dagger(a) * a
    ts   = tlist()
    _, n = timeevolution.master(ts, psi0, H, J; rates = rates, fout = (t, rho) -> real(expect(nop, rho)))
    n = Float64.(n)
    all(isfinite, n) || error("exact n(t) contains non-finite values.")
    maximum(n) < FOCK_GUARD * NCUT || error("Fock cutoff too small for this cell; raise NCUT.")
    i = argmax(n)
    return n[i], ts[i]
end

function main()
    inp = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "external_output", "panel_params_for_julia.json")
    d = JSON.parsefile(inp)
    outp = joinpath(dirname(inp), "julia_exact.csv")
    open(outp, "w") do io
        println(io, "cell,M,seed,kappa,n_peak_exact,t_peak_exact")
        for c in d["cells"]
            npk, tpk = exact_na(c)
            println(io, "$(c["regime"]),$(c["M"]),$(c["seed"]),$(c["kappa"]),$(npk),$(tpk)")
            println("$(c["regime"]): exact n_peak = $(round(npk, digits = 6)) at t = $(round(tpk, digits = 2))")
        end
    end
    println("wrote ", outp)
end

main()
