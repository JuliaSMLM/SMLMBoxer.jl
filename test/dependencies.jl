# Dependency guarantees that downstream environments rely on.
using Test, SMLMBoxer

@testset "cuDNN is not loaded" begin
    # SMLMBoxer must never load cuDNN: its CUDNN_jll collides with Reactant's bundled
    # cuDNN (undefined symbol in libcudnn_cnn.so) when both load in one process.
    cudnn = Base.PkgId(Base.UUID("02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"), "cuDNN")
    @test !haskey(Base.loaded_modules, cudnn)
end
