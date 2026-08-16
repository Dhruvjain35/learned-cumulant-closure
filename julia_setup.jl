# One-time dependency install for the cross-platform reproduction (see REPLICATION_QUICKSTART.md).
# Run from the repo root:  julia --project=. julia_setup.jl
#
# This instantiates a PINNED dependency set rather than resolving "whatever the registry has today". The
# unpinned resolve is what produced the API-drift failures in the first external run.
#
# Two manifests are committed, and Julia picks the right one automatically:
#   Manifest-v1.11.toml  -> used by Julia 1.11.x
#   Manifest.toml        -> used by everything else (generated on 1.12.6)
# Both pin the same direct-dependency versions. A version-specific manifest is required because the 1.12
# manifest cannot be instantiated on 1.11: it pins a PrecompileTools that needs Julia 1.12 internals
# (`UndefVarError: StaticData not defined in Base`), which fails 7 direct dependencies.

using Pkg

Pkg.activate(@__DIR__)

const VER_MANIFEST = joinpath(@__DIR__, "Manifest-v$(VERSION.major).$(VERSION.minor).toml")
const STD_MANIFEST = joinpath(@__DIR__, "Manifest.toml")

if isfile(VER_MANIFEST)
    println("Julia $(VERSION): using the version-specific pinned manifest ",
            basename(VER_MANIFEST), ".")
elseif isfile(STD_MANIFEST)
    println("Julia $(VERSION): using Manifest.toml (generated on Julia 1.12.6).")
    println("If this errors below, your Julia minor version is incompatible with that pinned set -- see the")
    println("recovery instructions this script prints on failure.")
else
    println("No manifest found -- resolving from the exact [compat] pins in Project.toml.")
    Pkg.resolve()
end

# Both instantiate AND precompile must be guarded: a Julia-version-incompatible manifest fails inside
# Pkg.precompile(), not inside Pkg.instantiate(), so guarding only the former lets a raw stacktrace through.
try
    Pkg.instantiate()
    Pkg.precompile()
catch e
    println("\n!! Dependency setup FAILED:")
    println(sprint(showerror, e))
    println("\nThis is almost certainly a Julia-version mismatch with the pinned manifest.")
    println("Recovery, in order of preference:")
    println("  1. Use a Julia version that has a committed manifest: 1.11.x or 1.12.6.")
    println("  2. Resolve from the [compat] pins instead. From the repo root:")
    println("       mv Manifest.toml Manifest.toml.bak")
    println("       julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'")
    println("     This still pins every DIRECT dependency to the exact tested version; only the transitive")
    println("     closure is re-resolved. It has been verified to give identical results on Julia 1.11.2.")
    println("  3. Report the error. A resolution failure here is useful diagnostic information, and is the")
    println("     script author's problem to fix rather than yours.")
    rethrow()
end

println("\n# installed versions (copy into REPLICATION_SIGNOFF.md):")
try
    Pkg.status()
catch e
    println("Pkg.status() failed: ", e)
end
println("\nsetup complete. Next:  julia --project=. replicate_cumulant.jl")
