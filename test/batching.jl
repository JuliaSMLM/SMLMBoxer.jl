using Test, SMLMBoxer

@testset "A frame cap preserves detections and global frame indices" begin
    image = zeros(Float32, 32, 32, 5)
    for f in axes(image, 3)
        image[10 + f, 16, f] = 1000f0
    end
    args = SMLMBoxer.GetBoxesArgs(; imagestack=image,
        sigma_small=1.0, sigma_large=2.0, minval=1.0, backend=:cpu)
    stack = SMLMBoxer.reshape_for_flux(image)
    reference, _, _, _ = SMLMBoxer._process_with_batching(
        stack, args, 5, typemax(Int); use_gpu=false)
    coords, batch_size, n_batches, memory = SMLMBoxer._process_with_batching(
        stack, args, 5, typemax(Int); use_gpu=false, max_batch_frames=2)
    @test all(!isempty, reference)
    @test coords == reference
    @test batch_size == 2
    @test n_batches == 3
    @test memory == 2 * SMLMBoxer.estimate_gpu_memory_per_frame(32, 32, nothing)
end
