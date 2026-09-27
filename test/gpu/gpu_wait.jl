# Backend selection and the GPU wait/fallback contract (getboxes docstring):
#   :cpu  never touches the GPU;
#   :gpu  runs on a GPU, or errors once gpu_timeout passes without one;
#   :auto runs on a GPU when one has room, else waits up to auto_timeout (calling on_wait
#         each poll) and then falls back to the CPU.
# The unavailable-GPU cases fill every device with a blocker, leaving less free memory than
# one frame needs, so the fallback is forced rather than hoped for.
using Test, SMLMBoxer, SMLMData, CUDA

img = rand(Float32, 128, 128, 10)
camera = IdealCamera(1:129, 1:129, 0.1f0)
kw = (sigma_small = 1.5, sigma_large = 3.0, minval = 0.1)

@testset "GPU available" begin
    (roi_cpu, info_cpu) = getboxes(img, camera; backend = :cpu, kw...)
    @test info_cpu.backend == :cpu
    @test info_cpu.device_id == -1

    (roi_auto, info_auto) = getboxes(img, camera; backend = :auto, auto_timeout = 60.0, kw...)
    @test info_auto.backend == :gpu
    @test info_auto.device_id >= 0
    @test length(roi_auto) == length(roi_cpu)

    (roi_gpu, info_gpu) = getboxes(img, camera; backend = :gpu, gpu_timeout = 60.0, kw...)
    @test info_gpu.backend == :gpu
    @test info_gpu.device_id >= 0
    @test length(roi_gpu) == length(roi_cpu)
end

@testset "GPU unavailable" begin
    # One 4096x4096 frame needs ~400 MB (~600 MB with the 1.5x margin); leave ~200 MB free.
    big = rand(Float32, 4096, 4096, 1)
    big_camera = IdealCamera(1:4097, 1:4097, 0.1f0)
    need = SMLMBoxer.estimate_gpu_memory_per_frame(4096, 4096, big_camera)
    leave = 200_000_000

    blockers = CuArray{UInt8}[]
    try
        for dev in CUDA.devices()
            CUDA.device!(dev)
            push!(blockers, CUDA.zeros(UInt8, max(0, CUDA.free_memory() - leave)))
            CUDA.synchronize()
        end
        # Precondition: no device has room for one frame, as NVML (the poll) sees it.
        maxfree = maximum(
            CUDA.NVML.memory_info(CUDA.NVML.Device(i)).free for i in 0:(length(CUDA.devices()) - 1)
        )
        @test maxfree < 1.5 * need

        waits = Ref(0)
        on_wait = (elapsed, available, required) -> begin
            waits[] += 1
            @test available < 1.5 * required
            return nothing
        end
        (roi, info) = @test_logs (:warn, r"GPU unavailable") match_mode = :any getboxes(
            big, big_camera; backend = :auto, auto_timeout = 2.0, on_wait = on_wait, kw...
        )
        @test info.backend == :cpu
        @test info.device_id == -1
        @test waits[] >= 1

        @test_throws ErrorException getboxes(big, big_camera; backend = :gpu, gpu_timeout = 2.0, kw...)
    finally
        empty!(blockers)
        GC.gc()
        for dev in CUDA.devices()
            CUDA.device!(dev)
            CUDA.reclaim()
        end
    end
end
