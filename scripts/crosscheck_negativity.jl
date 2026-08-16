# Julia half of the negativity loose-end check (see scripts/crosscheck_negativity.py).
# Emits the second-order cumulant <a'a>(t) on a FIXED shared grid with tight tolerances, so the Python
# comparison is done at identical times rather than at whichever points each solver happened to save.
#
# Run:  julia --project=. scripts/crosscheck_negativity.jl
using QuantumCumulants
using ModelingToolkitBase: @named, mtkcompile, ODEProblem, unknowns
using OrdinaryDiffEq

const M     = 2
const WC    = 1.0
const W     = [1.0, 1.02]
const G     = [0.5, 0.5]
const KAPPA = 0.1
const GDOWN = 0.01
const GPHI  = 0.0
const TGRID = collect(0.0:0.1:20.0)

const SUBS = Dict('₀'=>'0','₁'=>'1','₂'=>'2','₃'=>'3','₄'=>'4','₅'=>'5','₆'=>'6','₇'=>'7','₈'=>'8','₉'=>'9')
function factors(x)
    s = replace(string(x), r"\(t\)$" => "")
    s = replace(s, "⟨" => "", "⟩" => "")
    return strip.(split(s, "*"))
end
iscav(f) = (f == "a" || f == "a'" || f == "a′")
function psv(x)
    fs = factors(x)
    for f in fs
        iscav(f) && return 0.0 + 0im
        m = match(r"^σ(\d+)([₀-₉])([₀-₉])$", f)
        m === nothing && error("UNRECOGNIZED FACTOR '$f'")
        (SUBS[first(m.captures[2])] == '2' && SUBS[first(m.captures[3])] == '2') || return 0.0 + 0im
    end
    return 1.0 + 0im
end

hc  = FockSpace(:cavity)
has = [NLevelSpace(Symbol("atom", i), 2) for i in 1:M]
h   = tensor(hc, has...)
a    = Destroy(h, :a)
sm(k) = Transition(h, Symbol("σ", k), 1, 2, k + 1)
sp(k) = Transition(h, Symbol("σ", k), 2, 1, k + 1)
pe(k) = Transition(h, Symbol("σ", k), 2, 2, k + 1)

H = WC * (a' * a)
for k in 1:M
    global H
    H += W[k] * pe(k) + G[k] * (a * sp(k) + a' * sm(k))
end
J = Any[a]; rates = Float64[KAPPA]
for k in 1:M
    push!(J, sm(k));         push!(rates, GDOWN)
    push!(J, 2 * pe(k) - 1); push!(rates, GPHI / 2)
end

eqs = QuantumCumulants.complete(meanfield([a' * a], H, J; rates = rates, order = 2))
@named sys = System(eqs)
sysc = mtkcompile(sys)
uk = unknowns(sysc)
u0 = Dict(u => psv(u) for u in uk)

prob = ODEProblem(sysc, u0, (0.0, TGRID[end]))
sol  = solve(prob, Tsit5(); saveat = TGRID, reltol = 1e-10, abstol = 1e-12)
sol.retcode == ReturnCode.Success || error("solve failed: $(sol.retcode)")

nfun = get_solution(sol, a' * a, eqs)
open(joinpath(@__DIR__, "..", "results", "crosscheck_negativity_julia.csv"), "w") do io
    println(io, "t,n")
    for t in TGRID
        println(io, t, ",", real(nfun(t)))
    end
end
println("wrote results/crosscheck_negativity_julia.csv  (min n = ",
        minimum(real(nfun(t)) for t in TGRID), ")")
