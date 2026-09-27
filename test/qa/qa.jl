# Default QA (admiral decisions 0008, 0015, 0018): Aqua, ExplicitImports, docstring doctests and
# missing docstrings, unchanged in every package except for opt-outs written with their reason
# (the package name is read from Project.toml). Aqua, ExplicitImports and Documenter go in the test env.
using Test, TOML, Aqua, ExplicitImports, Documenter

const PKGNAME = Symbol(TOML.parsefile(joinpath(@__DIR__, "..", "..", "Project.toml"))["name"])
@eval using $PKGNAME
const PKG = getfield(@__MODULE__, PKGNAME)

@testset "Aqua" begin
    # Opt-out, persistent_tasks: Aqua 0.8.18 builds its wrapper manifest from each dependency's
    # Project.toml, which misses CUDA_Runtime_jll's platform-selected CUDA_Compiler_jll, so the
    # package fails to load in the wrapper before the check runs (a loading error, not a task).
    Aqua.test_all(PKG; persistent_tasks = false)
    # Opt-out: switch off one check and state the reason beside it, e.g.
    #   Aqua.test_all(PKG; ambiguities=false)  # reason: ambiguities come from ForwardDiff's Dual methods
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(PKG) === nothing
    @test check_no_stale_explicit_imports(PKG) === nothing
    # Opt-out: ignore named items and state the reason, e.g.
    #   check_no_implicit_imports(PKG; ignore=(:Foo,))  # reason: Foo is re-exported on purpose
end

@testset "Docstrings" begin
    # Every public name, the module included, has a docstring (Docs.undocumented_names: Julia 1.11+).
    if isdefined(Base.Docs, :undocumented_names)
        @test isempty(Base.Docs.undocumented_names(PKG))
    end
    # Every jldoctest in a docstring runs. Examples that need a GPU, data or network are plain julia blocks.
    DocMeta.setdocmeta!(PKG, :DocTestSetup, :(using $PKGNAME); recursive = true)
    doctest(PKG; manual = false)
end
