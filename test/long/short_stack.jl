using Test, SMLMBoxer, CUDA, Statistics

@testset "Short GPU calls avoid full-GC release overhead" begin
    if CUDA.functional()
        image = zeros(Float32, 256, 256, 10)
        image[64, 64, :] .= 2000f0
        image[160, 180, :] .= 1500f0
        kw = (minval=1.0,)
        reference, _ = getboxes(image; backend=:cpu, kw...)
        @test !isempty(reference.data)
        for _ in 1:2
            getboxes(image; backend=:gpu, gpu_timeout=120.0, kw...)
        end

        times = Float64[]
        for _ in 1:7
            elapsed = @elapsed roi, info = getboxes(image;
                backend=:gpu, gpu_timeout=120.0, kw...)
            push!(times, elapsed)
            @test info.backend == :gpu
            @test roi.data == reference.data
            @test roi.x_corners == reference.x_corners
            @test roi.y_corners == reference.y_corners
            @test roi.frame_indices == reference.frame_indices
        end
        @info "256x256x10 GPU call median" milliseconds=1000 * median(times)
        @test median(times) < 0.020
    else
        @test_skip CUDA.functional()
    end
end
