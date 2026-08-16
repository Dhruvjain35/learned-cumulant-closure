# One-time dependency install for the cross-platform reproduction (see REPLICATION_QUICKSTART.md).
# Run from the repo root:  julia --project=. julia_setup.jl
#
# This now instantiates the EXACT dependency set recorded in Manifest.toml, which is the set the script was
# executed against. It no longer resolves "whatever the registry has today" -- that unpinned resolve is what
# produced the API-drift failures reported in the first external run (QuantumCumulants 0.7 renders operators
# differently, and the ModelingToolkit wrapper is not the package QuantumCumulants 0.7 builds on).

using Pkg

Pkg.activate(@__DIR__)

if isfile(joinpath(@__DIR__, "Manifest.toml"))
    println("Manifest.toml found -- instantiating the pinned dependency set.")
    try
        Pkg.instantiate()
    catch e
        println("\n!! Pkg.instantiate() from the pinned Manifest.toml FAILED:")
        println(sprint(showerror, e))
        println("\nThis is most likely a Julia-version mismatch. The Manifest was generated on Julia 1.12.6.")
        println("Options, in order of preference:")
        println("  1. Run this on Julia 1.12.6 (juliaup: `juliaup add 1.12.6; julia +1.12.6 --project=. ...`).")
        println("  2. Delete Manifest.toml and re-run this script; it will then resolve from the [compat]")
        println("     bounds in Project.toml, which pin the same direct-dependency versions.")
        println("  3. Report the error -- a resolution failure here is itself useful diagnostic information.")
        rethrow()
    end
else
    println("No Manifest.toml -- resolving from the [compat] pins in Project.toml.")
    Pkg.resolve()
    Pkg.instantiate()
end

Pkg.precompile()

println("\n# installed versions (copy into REPLICATION_SIGNOFF.md):")
try
    Pkg.status()
catch e
    println("Pkg.status() failed: ", e)
end
println("\nsetup complete. Next:  julia --project=. replicate_cumulant.jl")
