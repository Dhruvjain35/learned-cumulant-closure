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
# STATUS (read this, the distinction matters):
#   EXECUTION-VERIFIED: this script now runs end to end against the versions pinned in Manifest.toml
#     (Julia 1.12.6, QuantumCumulants 0.7.0, QuantumOptics 1.2.7, OrdinaryDiffEq 7.6.0,
#     ModelingToolkitBase 1.65.0, JSON 1.7.1) and produces the acceptance numbers. The earlier
#     "authored without a Julia runtime" caveat no longer applies to the CODE.
#   STILL NOT AN INDEPENDENT VALIDATION: it was run by the project's own author on the author's machine.
#     Per docs/EXTERNAL_VALIDATION_STATUS.md, "cross-platform reproduction" requires a person who did not
#     write this code to run it and record the result. That row remains NOT DONE. Do not cite this script's
#     own output as external validation of anything.
# Written against the pinned versions above; the cumulant initial-value mapping (product_state_value) remains the
# most fragile part and now FAILS CLOSED (it throws on any operator rendering it does not recognise) rather than
# silently returning a wrong initial condition. See REPLICATION_AUDIT.md for every assumption, and
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
# QuantumCumulants 0.7 builds its systems on ModelingToolkitBase (see its Project.toml [compat] and test suite);
# importing the ModelingToolkit wrapper instead makes ODEProblem fail with "im is present in the system but im is
# not an unknown". Import the names explicitly so nothing shadows QuantumOptics.
using ModelingToolkitBase: @named, mtkcompile, ODEProblem, unknowns
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
    # Match by name over the whole dependency graph: SecondQuantizedAlgebra is a TRANSITIVE dependency (pulled
    # in by QuantumCumulants 0.7), so identify_package cannot resolve it from this project's direct deps.
    byname = Dict{String,Any}()
    for (_, info) in Pkg.dependencies()
        byname[info.name] = info.version
    end
    for name in ("QuantumCumulants", "QuantumOptics", "OrdinaryDiffEq", "ModelingToolkitBase",
                 "SecondQuantizedAlgebra", "JSON")
        v = get(byname, name, nothing)
        println(rpad(name, 24), ": ", v === nothing ? "<not installed>" : v)
    end
    println("NOTE: this script HAS been executed against the versions pinned in Manifest.toml (see the STATUS")
    println("block at the top of this file). If your resolved versions differ from the pinned ones and the")
    println("script errors on an API call, that is version drift, not a physics problem -- see REPLICATION_AUDIT.md.\n")
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

    # `dagger` is exported by BOTH QuantumOptics and SecondQuantizedAlgebra (via QuantumCumulants), so a bare
    # call is ambiguous and errors. Qualify it: this block is the QuantumOptics (numeric) side.
    H = wc * (QuantumOptics.dagger(a) * a)
    for k in 1:M
        H += w[k] * pe[k] + g[k] * (a * sp[k] + QuantumOptics.dagger(a) * sm[k])
    end
    J = Any[a]; rates = Float64[kappa]
    for k in 1:M
        push!(J, sm[k]); push!(rates, gdn)           # gdown * D[sm]
        push!(J, sz[k]); push!(rates, gph / 2)       # (gphi/2) * D[sz]
    end
    psi0 = tensor(fockstate(bc, 0), fill(spinup(ba), M)...)   # vacuum + all excited
    nop  = QuantumOptics.dagger(a) * a
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
    eqs = QuantumCumulants.complete(meanfield([a' * a], H, J; rates = rates, order = 2))
    @named sys = System(eqs)
    sysc = mtkcompile(sys)
    uk = unknowns(sysc)

    # initial condition: cavity vacuum, ALL atoms excited (factorized product state).
    # product_state_value parses each tracked average STRUCTURALLY (factor by factor) and throws on any factor
    # it does not recognise, so a future rendering change halts instead of silently returning zeros.
    vals = [product_state_value(u) for u in uk]
    u0 = Dict(uk[i] => vals[i][1] for i in eachindex(uk))

    # GUARD the IC mapping: at t=0, <a'a>=0 and the M single-emitter populations are each 1 (all excited).
    # These now FAIL CLOSED -- an empty match is an error, not a skipped check.
    ia = findfirst(i -> vals[i][2] == :adaga, eachindex(uk))
    ia === nothing && error("could not locate <a'a> among cumulant unknowns -- API mismatch, see AUDIT.")
    abs(vals[ia][1]) < 1e-9 || error("initial <a'a> = $(vals[ia][1]) != 0 -- IC mapping is WRONG.")
    pe_idx = [i for i in eachindex(uk) if vals[i][2] == :single_pe]
    length(pe_idx) == M || error("found $(length(pe_idx)) single-emitter population states, expected M=$M -- "
        * "IC mapping is WRONG for your QuantumCumulants version (see REPLICATION_AUDIT.md).")
    tot = sum(real(vals[i][1]) for i in pe_idx)
    isapprox(tot, float(M); atol = 1e-6) || error("initial total excited pop = $tot != M=$M -- IC mapping WRONG.")

    prob = ODEProblem(sysc, u0, (0.0, TMAX))
    sol  = solve(prob, Tsit5(); saveat = tlist(), reltol = 1e-8, abstol = 1e-10)
    sol.retcode == ReturnCode.Success || error("cumulant ODE solve failed: retcode=$(sol.retcode).")

    # Extract <a'a>(t) symbolically (QuantumCumulants' own accessor) rather than by positional index, then
    # cross-check against the directly-indexed unknown -- if these disagree the extraction is untrustworthy.
    nfun = get_solution(sol, a' * a, eqs)
    n = Float64[real(nfun(t)) for t in sol.t]
    n_direct = Float64.(real.(getindex.(sol.u, ia)))
    maximum(abs.(n .- n_direct)) < 1e-8 || error("get_solution and direct indexing of <a'a> disagree by "
        * "$(maximum(abs.(n .- n_direct))) -- observable extraction is unreliable, halting.")
    all(isfinite, n) || error("cumulant n(t) contains non-finite values (closure may have diverged).")
    return collect(sol.t), n
end

# Product-state expectation of a tracked average for |vacuum> ⊗ |e,e,...,e>.
#
# This is parsed STRUCTURALLY, factor by factor, rather than by substring search on the whole average. That
# matters: QuantumCumulants 0.7 renders transitions with UNICODE SUBSCRIPT level indices (σ1₂₂, not σ122), so
# the previous `occursin("22", ...)` test matched nothing and returned 0.0 for EVERY state -- a silently wrong
# (un-inverted) initial condition rather than a crash. Any factor this does not recognise is an ERROR, so a
# future rendering change halts loudly instead of quietly producing a plausible wrong number.
#
# Returns (value, tag) where tag is :adaga for <a'a>, :single_pe for a single <σk₂₂>, else :other.
const _SUBSCRIPTS = Dict('₀'=>'0','₁'=>'1','₂'=>'2','₃'=>'3','₄'=>'4','₅'=>'5','₆'=>'6','₇'=>'7','₈'=>'8','₉'=>'9')

function _avg_factors(x)
    s = replace(string(x), r"\(t\)$" => "")      # unknowns carry a (t); eqs.states do not
    s = replace(s, "⟨" => "", "⟩" => "")
    return strip.(split(s, "*"))
end

_is_cavity_factor(f) = (f == "a" || f == "a'" || f == "a′")

function product_state_value(x)
    fs = _avg_factors(x)
    ncav = count(_is_cavity_factor, fs)
    n22  = 0
    for f in fs
        _is_cavity_factor(f) && continue
        m = match(r"^σ(\d+)([₀-₉])([₀-₉])$", f)
        m === nothing && error("UNRECOGNIZED FACTOR '$f' in average '$(string(x))' -- the operator rendering of "
            * "your QuantumCumulants version is not one this script knows how to map to a product state. "
            * "Fix product_state_value before trusting any output (see REPLICATION_AUDIT.md).")
        i = _SUBSCRIPTS[first(m.captures[2])]; j = _SUBSCRIPTS[first(m.captures[3])]
        (i == '2' && j == '2') || return (0.0 + 0im, :other)   # any raising/lowering factor -> 0 in |e...e>
        n22 += 1
    end
    # <a'a> is the tracked photon number: exactly the two cavity factors a' and a, nothing else.
    if ncav == 2 && n22 == 0 && length(fs) == 2 && any(f -> f == "a'" || f == "a′", fs) && any(==("a"), fs)
        return (0.0 + 0im, :adaga)                              # vacuum: <a'a> = 0
    end
    ncav > 0 && return (0.0 + 0im, :other)                      # any cavity factor -> 0 in the vacuum
    return (1.0 + 0im, n22 == 1 ? :single_pe : :other)          # pure product of excited projectors
end

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
