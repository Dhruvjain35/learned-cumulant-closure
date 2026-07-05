# =====================================================================================================
# CROSS-PLATFORM REPRODUCTION: 2nd-order cumulant vs EXACT, open disordered Tavis-Cummings superradiant burst.
#
# Purpose: reproduce -- in a separate codebase (QuantumCumulants.jl + QuantumOptics.jl) -- two quantities the
# rest of this project is measured against:
#   (1) the exact reference n(t) = <a'a>(t) for the open disordered TC model, fully-inverted initial state;
#   (2) that the standard 2nd-order cumulant closure MISSES the superradiant burst peak F_peak = kappa*max_t n(t).
# It does NOT reproduce the learned closure (that is the model under test); the learned closure's advantage is
# then measured against this independently-run baseline.
#
# STATUS (read this): this script was authored WITHOUT a Julia runtime available to its author. It is therefore
# NOT execution-verified and its output is NOT an independent validation until a Julia user runs it and checks
# the acceptance criterion. It is written to QuantumCumulants.jl >= 0.3 / QuantumOptics.jl APIs; expect some API
# drift across versions -- the cumulant initial-value mapping (product_state_value) is the most fragile part and
# is guarded by an explicit runtime check below. See REPLICATION_AUDIT.md for every assumption, and
# REPLICATION_QUICKSTART.md for how to run. Language note: this file avoids the words first/novel/necessary/proves
# on purpose -- it reports a reproduction check, nothing more.
#
# MODEL (units: everything in units of the cavity frequency; wc = 1). Convention: spin-up = EXCITED,
# spin-down = ground; sz = |e><e| - |g><g| = 2*pe - 1; pe = |e><e|. Per-emitter frequency w_i and coupling g_i.
#   H = wc a'a + sum_i w_i pe_i + sum_i g_i (a sp_i + a' sm_i)         (RWA)
#   dissipators: kappa*D[a]  +  sum_i gdown*D[sm_i]  +  sum_i (gphi/2)*D[sz_i]
#   initial state: cavity vacuum, every emitter EXCITED (fully inverted).
#   observable: emitted flux kappa*<a'a>(t); burst peak F_peak = kappa*max_t<a'a>, delay t_peak = argmax_t.
# =====================================================================================================

using QuantumCumulants
using QuantumOptics
using OrdinaryDiffEq
using ModelingToolkit
using JSON
using SHA
using Pkg

const TMAX = 40.0
const NT   = 200
const NCUT = 16          # Fock cutoff for the exact solve; the Fock-occupation guard below checks it is enough
const FOCK_GUARD = 0.45  # fail if the exact burst reaches >= this fraction of the Fock cutoff (unreliable)

tlist() = collect(range(0.0, TMAX; length = NT))

# ---- environment / version report (recorded in REPLICATION_SIGNOFF.md; NOT asserted -- see AUDIT) ------------
function report_versions()
    println("# ENVIRONMENT (record these in REPLICATION_SIGNOFF.md)")
    println("Julia version: ", VERSION)
    for name in ("QuantumCumulants", "QuantumOptics", "OrdinaryDiffEq", "ModelingToolkit")
        try
            v = Pkg.dependencies()[Base.identify_package(name).uuid].version
            println(rpad(name, 18), ": ", v)
        catch
            println(rpad(name, 18), ": <version query failed; record manually via `] status`>")
        end
    end
    println("NOTE: these package versions were NOT tested by the script's author. If the script errors on an")
    println("API call, that is expected drift -- see REPLICATION_AUDIT.md, do not assume the physics is wrong.\n")
end

# ---- parameter loading with hashes + validation (cannot silently run on wrong/missing params) ----------------
function load_params()
    path = joinpath(@__DIR__, "results", "replication_params.json")
    isfile(path) || error("MISSING results/replication_params.json at $path -- clone the full repo.")
    raw = read(path)
    phash = bytes2hex(sha256(raw))
    d = JSON.parse(String(raw))
    haskey(d, "cells") && !isempty(d["cells"]) || error("replication_params.json has no 'cells'.")
    println("parameter-file sha256 : ", phash)
    return d, phash
end

function validate_cell(c)
    for k in ("M", "seed", "wc", "kappa", "gdown", "gphi", "w", "g", "regime")
        haskey(c, k) || error("cell missing key '$k' -- params file malformed.")
    end
    M = c["M"]
    w = Float64.(c["w"]); g = Float64.(c["g"])
    length(w) == M || error("disorder draw w has length $(length(w)) != M=$M (malformed).")
    length(g) == M || error("coupling g has length $(length(g)) != M=$M (malformed).")
    all(isfinite, w) && all(isfinite, g) || error("non-finite entry in w or g (malformed).")
    isapprox(c["wc"], 1.0; atol = 1e-9) || @warn "wc != 1: the units convention in this script assumes wc=1."
    # disorder-draw hash so the collaborator can confirm they used the identical draw
    dhash = bytes2hex(sha256(string(round.(w; digits = 12))))
    return M, w, g, dhash
end

# ---------------------------------------------------------------------------------------------------
# EXACT master equation (QuantumOptics.jl)
# ---------------------------------------------------------------------------------------------------
function exact_na(c, M, w, g)
    wc = c["wc"]; kappa = c["kappa"]; gdn = c["gdown"]; gph = c["gphi"]
    bc = FockBasis(NCUT)
    ba = SpinBasis(1//2)                         # spinup = EXCITED, spindown = ground
    b  = tensor(bc, fill(ba, M)...)
    a  = embed(b, 1, destroy(bc))
    sp = [embed(b, k + 1, sigmap(ba)) for k in 1:M]   # |e><g|
    sm = [embed(b, k + 1, sigmam(ba)) for k in 1:M]   # |g><e|
    sz = [embed(b, k + 1, sigmaz(ba)) for k in 1:M]
    pe = [0.5 * (identityoperator(b) + sz[k]) for k in 1:M]

    H = wc * (dagger(a) * a)
    for k in 1:M
        H += w[k] * pe[k] + g[k] * (a * sp[k] + dagger(a) * sm[k])
    end
    J = Any[a]; rates = Float64[kappa]
    for k in 1:M
        push!(J, sm[k]); push!(rates, gdn)           # gdown * D[sm]
        push!(J, sz[k]); push!(rates, gph / 2)       # (gphi/2) * D[sz]
    end
    psi0 = tensor(fockstate(bc, 0), fill(spinup(ba), M)...)   # vacuum + all excited
    nop  = dagger(a) * a
    ts   = tlist()
    _, n = timeevolution.master(ts, psi0, H, J; rates = rates, fout = (t, rho) -> real(expect(nop, rho)))
    n = Float64.(n)
    # physical-validity + Fock-cutoff guards (cannot silently return garbage)
    all(isfinite, n) || error("exact n(t) contains non-finite values.")
    all(x -> x > -1e-8, n) || error("exact n(t) went negative -- non-physical.")
    maximum(n) < FOCK_GUARD * NCUT || error("exact burst reaches $(round(maximum(n),digits=2)) photons, >= "
        * "$(FOCK_GUARD)*NCUT=$(FOCK_GUARD*NCUT); raise NCUT and rerun (result would be Fock-truncation-biased).")
    return ts, n
end

# ---------------------------------------------------------------------------------------------------
# 2nd-order CUMULANT (QuantumCumulants.jl)
# ---------------------------------------------------------------------------------------------------
function cumulant_na(c, M, w, g)
    wc = c["wc"]; kappa = c["kappa"]; gdn = c["gdown"]; gph = c["gphi"]
    hc  = FockSpace(:cavity)
    has = [NLevelSpace(Symbol("atom", i), 2) for i in 1:M]   # level 1 = ground, 2 = excited
    h   = tensor(hc, has...)
    a    = Destroy(h, :a)
    sm(k) = Transition(h, Symbol("σ", k), 1, 2, k + 1)       # |g><e|
    sp(k) = Transition(h, Symbol("σ", k), 2, 1, k + 1)       # |e><g|
    pe(k) = Transition(h, Symbol("σ", k), 2, 2, k + 1)       # |e><e|

    H = wc * (a' * a)
    for k in 1:M
        H += w[k] * pe(k) + g[k] * (a * sp(k) + a' * sm(k))
    end
    J = Any[a]; rates = Float64[kappa]
    for k in 1:M
        push!(J, sm(k));         push!(rates, gdn)
        push!(J, 2 * pe(k) - 1); push!(rates, gph / 2)       # sz = 2*pe - 1
    end
    eqs = complete(meanfield([a' * a], H, J; rates = rates, order = 2))
    @named sys = ODESystem(eqs)

    # initial condition: cavity vacuum, ALL atoms excited (factorized product state).
    # product_state_value maps each tracked average to its product-state value. THIS IS THE FRAGILE STEP
    # (string-based; API-version dependent). It is guarded immediately below by physical checks on u0.
    states = eqs.states
    u0 = ComplexF64[product_state_value(s, M) for s in states]

    ia = findfirst(s -> _is_adaga(s), states)
    ia === nothing && error("could not locate <a'a> among cumulant states -- API/string mismatch, see AUDIT.")
    # GUARD the fragile IC mapping: at t=0, <a'a>=0 and total excited population = M (all excited).
    abs(u0[ia]) < 1e-9 || error("initial <a'a> = $(u0[ia]) != 0 -- product_state_value IC mapping is WRONG for "
        * "your QuantumCumulants version; fix it (see REPLICATION_AUDIT.md) before trusting output.")
    pe_idx = [i for (i, s) in enumerate(states) if _is_single_pe(s)]
    if !isempty(pe_idx)
        tot = sum(real(u0[pe_idx]))
        isapprox(tot, float(M); atol = 1e-6) || error("initial total excited pop = $tot != M=$M -- IC mapping "
            * "WRONG; fix product_state_value (see REPLICATION_AUDIT.md).")
    end

    prob = ODEProblem(sys, u0, (0.0, TMAX))
    sol  = solve(prob, Tsit5(); saveat = tlist(), reltol = 1e-8, abstol = 1e-10)
    sol.retcode == :Success || error("cumulant ODE solve failed: retcode=$(sol.retcode).")
    n = Float64.(real.(getindex.(sol.u, ia)))
    all(isfinite, n) || error("cumulant n(t) contains non-finite values (closure may have diverged).")
    return collect(sol.t), n
end

# product-state expectation of a tracked average (all atoms excited, cavity vacuum). Returns 1 only for a
# product of excited-projectors; 0 for anything containing a cavity op or a raising/lowering transition.
function product_state_value(avg, M)
    ex = string(avg)
    (occursin("a", ex) && !occursin("atom", ex)) && return 0.0 + 0im   # bare cavity a / a'
    occursin("21", ex) && return 0.0 + 0im                              # raising  σ21
    occursin("12", ex) && return 0.0 + 0im                              # lowering σ12
    (occursin("22", ex) && !occursin("Destroy", ex)) && return 1.0 + 0im
    return 0.0 + 0im
end
_is_adaga(avg) = occursin("a′*a", string(avg)) || occursin("a'*a", string(avg)) || string(avg) == "⟨a′*a⟩"
_is_single_pe(avg) = (n22 = count(==("22"), split(string(avg), r"[^0-9]+")); n22 == 1 &&
                      !occursin("a", replace(string(avg), "atom" => "")) )

# ---------------------------------------------------------------------------------------------------
function burst(ts, n, kappa)
    ipk = argmax(n)
    return (Fpeak = kappa * n[ipk], tpeak = ts[ipk], npeak = n[ipk])
end

function main()
    println("# CROSS-PLATFORM REPRODUCTION (QuantumCumulants.jl + QuantumOptics.jl) -- see REPLICATION_QUICKSTART.md\n")
    report_versions()
    d, phash = load_params()
    cells = [c for c in d["cells"] if c["seed"] == 50 && c["M"] in (2, 3, 4)]   # exact-feasible; one draw
    isempty(cells) && error("no matching cells (seed 50, M in 2..4) in params file.")
    println("\nregime   M | F_peak_exact  F_peak_cumulant  signed_relerr  t_peak_ex  t_peak_cum | disorder_hash")
    rows = String[]
    for c in cells
        kappa = c["kappa"]
        M, w, g, dhash = validate_cell(c)
        tse, ne = exact_na(c, M, w, g)
        tsc, nc = cumulant_na(c, M, w, g)
        # time-grid consistency (cannot compare peaks on mismatched grids)
        (length(tse) == length(tsc) && maximum(abs.(tse .- tsc)) < 1e-6) ||
            error("exact and cumulant time grids differ -- cannot compare peaks.")
        be = burst(tse, ne, kappa); bc = burst(tsc, nc, kappa)
        srel = (bc.Fpeak - be.Fpeak) / max(abs(be.Fpeak), 1e-12)          # SIGNED (expect POSITIVE overshoot)
        println("$(rpad(c["regime"],7)) $M | $(rpad(round(be.Fpeak,digits=4),12)) "
            * "$(rpad(round(bc.Fpeak,digits=4),15)) $(rpad(round(srel,digits=3),13)) "
            * "$(rpad(round(be.tpeak,digits=2),9)) $(round(bc.tpeak,digits=2)) | $(dhash[1:12])")
        push!(rows, "$(c["regime"]),$M,$(be.Fpeak),$(bc.Fpeak),$srel,$(be.tpeak),$(bc.tpeak),$phash,$dhash")
    end
    outp = joinpath(@__DIR__, "results", "replication_cumulant.csv")
    open(outp, "w") do io
        println(io, "regime,M,Fpeak_exact,Fpeak_cumulant,signed_relerr,tpeak_exact,tpeak_cumulant,params_sha256,disorder_hash")
        foreach(r -> println(io, r), rows)
    end
    println("\nwrote ", outp)
    println("\nACCEPTANCE (see REPLICATION_QUICKSTART.md): the 2nd-order cumulant should OVERSHOOT the burst peak")
    println("by a POSITIVE signed relative error of order +0.25..+0.35, roughly M-flat, in BOTH regimes.")
    println("(The overshoot does NOT grow with M -- that specific hypothesis was tested and rejected upstream.)")
end

main()
