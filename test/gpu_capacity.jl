using Test, SMLMBoxer

@testset "One frame fits a visible GPU" begin
    # Required bytes already include polling's 1.5x safety margin.
    @test SMLMBoxer.frame_fits_some_gpu(150, [100, 150])
    @test !SMLMBoxer.frame_fits_some_gpu(150, [100, 149])
    @test !SMLMBoxer.frame_fits_some_gpu(150, Int[])
end
