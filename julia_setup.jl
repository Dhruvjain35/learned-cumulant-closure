# One-time dependency install for the cross-platform reproduction (see REPLICATION_QUICKSTART.md).
# Run from the repo root:  julia --project=. julia_setup.jl
# This resolves package versions from the Julia General registry (no versions are hard-pinned by the author,
# who had no Julia runtime). Record the versions printed at the end in REPLICATION_SIGNOFF.md.

using Pkg

Pkg.activate(@__DIR__)

# SHA is a standard library (no add needed). The rest come from the registry.
for pkg in ("QuantumCumulants", "QuantumOptics", "OrdinaryDiffEq", "ModelingToolkit", "JSON")
    println("installing ", pkg, " ...")
    Pkg.add(pkg)
end

Pkg.instantiate()
Pkg.precompile()

println("\n# installed versions (copy into REPLICATION_SIGNOFF.md):")
try
    Pkg.status()
catch e
    println("Pkg.status() failed: ", e)
end
println("\nsetup complete. Next:  julia --project=. replicate_cumulant.jl")
