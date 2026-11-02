using Test, SMLMBoxer, SMLMData, CUDA

@testset "auto CPU fallback does not use the GPU" begin
    # auto_timeout = 0 forces the :auto fallback without waiting; the fallback must then
    # filter on the CPU, so it allocates no GPU memory. Only meaningful with a GPU.
    image = rand(Float32, 128, 128, 10)
    kw = (sigma_small=1.5, sigma_large=3.0, minval=0.1)
    cameras = (
        IdealCamera(1:129, 1:129, 0.1f0),
        SCMOSCamera(128, 128, 0.1f0, 5.0f0, offset=100.0f0, gain=2.0f0, qe=0.9f0),
    )
    for camera in cameras
        (roi_cpu, _) = getboxes(image, camera; backend=:cpu, kw...)
        getboxes(image, camera; backend=:auto, auto_timeout=0.0, kw...)  # compile first
        local roi, info
        gpu_bytes = CUDA.@allocated begin
            (roi, info) = getboxes(image, camera; backend=:auto, auto_timeout=0.0, kw...)
        end
        @test info.backend == :cpu
        @test gpu_bytes == 0
        @test length(roi) == length(roi_cpu)
        @test roi.x_corners == roi_cpu.x_corners && roi.y_corners == roi_cpu.y_corners
    end
end
