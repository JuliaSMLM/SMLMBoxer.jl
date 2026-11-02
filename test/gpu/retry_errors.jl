using Test, SMLMBoxer, SMLMData, CUDA

@testset "GPU processing raises non-resource errors immediately" begin
    # The camera and its readnoise map agree, but not with the 32x32 image below.
    # GetBoxesArgs does not check this: _process_with_batching -> dog_filter ->
    # dog_filter_variance_weighted -> get_variance_map asserts inside the GPU attempt.
    camera = SCMOSCamera(16, 16, 0.1f0, fill(5.0f0, 16, 16))
    image = zeros(Float32, 32, 32)

    # Exercise the GPU path on a matching image before checking error propagation.
    _, info = getboxes(zeros(Float32, 16, 16), camera;
        backend=:gpu, gpu_timeout=20.0)
    @test info.backend == :gpu

    for backend in (:gpu, :auto)
        @testset "$backend" begin
            logs, _ = Test.collect_test_logs() do
                @test_throws AssertionError getboxes(image, camera;
                    backend=backend, gpu_timeout=20.0, auto_timeout=20.0)
            end
            @test !any(log -> log.message ==
                "GPU failed, releasing memory and retrying", logs)
        end
    end
end
