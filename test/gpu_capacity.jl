using Test, SMLMBoxer

@testset "One frame fits memory a visible GPU can free" begin
    # Required bytes already include polling's 1.5x safety margin.
    @test SMLMBoxer.frame_fits_some_gpu(150, [100, 150], 0)
    @test !SMLMBoxer.frame_fits_some_gpu(150, [100, 149], 0)
    @test !SMLMBoxer.frame_fits_some_gpu(150, Int[], 0)

    reserve = SMLMBoxer.GPU_CONTEXT_RESERVE
    totals = [reserve + 100, reserve + 150]
    @test SMLMBoxer.frame_fits_some_gpu(150, totals, reserve)
    # Fits total memory, but cannot fit even after all reclaimable memory is freed.
    @test SMLMBoxer.frame_fits_some_gpu(151, totals, 0)
    @test !SMLMBoxer.frame_fits_some_gpu(151, totals, reserve)
    @test !SMLMBoxer.frame_fits_some_gpu(150, [reserve - 1], reserve)
    @test !SMLMBoxer.frame_fits_some_gpu(150, UInt64[reserve - 1], reserve)
end
