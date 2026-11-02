# Backend selection and the GPU wait/fallback contract (getboxes docstring):
#   :cpu  never touches the GPU;
#   :gpu  runs on a GPU, or errors once gpu_timeout passes without one;
#   :auto runs on a GPU when one has room, else waits up to auto_timeout (calling on_wait
#         each poll) and then falls back to the CPU.
# auto_cpu_fallback.jl forces the fallback with auto_timeout = 0. Here the unavailable-GPU
# cases fill every device in a child process so waiting and fallback are forced; its exit
# releases the blockers even when CUDA's memory pool retains freed allocations.
using Test, SMLMBoxer, SMLMData, CUDA

img = rand(Float32, 128, 128, 10)
camera = IdealCamera(1:128, 1:128, 0.1f0)
kw = (sigma_small = 1.5, sigma_large = 3.0, minval = 0.1)

@testset "GPU available" begin
    (roi_cpu, info_cpu) = getboxes(img, camera; backend = :cpu, kw...)
    @test info_cpu.backend == :cpu
    @test info_cpu.device_id == -1

    (roi_auto, info_auto) = getboxes(
        img, camera; backend = :auto, auto_timeout = 60.0, kw...,
    )
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
    big_camera = IdealCamera(1:4096, 1:4096, 0.1f0)
    need = SMLMBoxer.estimate_gpu_memory_per_frame(4096, 4096, big_camera)
    leave = 200_000_000

    # Release what earlier tests left in the memory pool first; memory freed after the
    # blockers are sized would make room for a frame and the :gpu case would not time out.
    GC.gc()
    for dev in CUDA.devices()
        CUDA.device!(dev)
        CUDA.reclaim()
    end
    script = raw"""
    using CUDA
    leave = parse(Int, ARGS[1])
    blockers = CuArray{UInt8}[]
    for dev in CUDA.devices()
        CUDA.device!(dev)
        push!(blockers, CUDA.zeros(UInt8, max(0, CUDA.free_memory() - leave)))
        CUDA.synchronize()
    end
    println("READY")
    flush(stdout)
    read(stdin)
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no
           --project=$(Base.active_project()) -e $script $leave`
    depot_path = join(DEPOT_PATH, Sys.iswindows() ? ';' : ':')
    cmd = addenv(cmd, "JULIA_DEPOT_PATH" => depot_path)
    mktemp() do stderr_path, blocker_stderr
        blocker = open(pipeline(cmd; stderr = blocker_stderr), "r+")
        ready = @async readline(blocker)
        try
            startup = timedwait(
                () -> istaskdone(ready) || process_exited(blocker), 120.0,
            )
            if startup == :timed_out
                error("GPU blocker timed out waiting for READY.\n" *
                    read(stderr_path, String))
            elseif process_exited(blocker) || fetch(ready) != "READY"
                error("GPU blocker exited or did not report READY.\n" *
                    read(stderr_path, String))
            end
            # Precondition: no device has room for one frame, as the poll sees it.
            maxfree = maximum(
                SMLMBoxer.gpu_free_memory(i)
                    for i in 0:(length(CUDA.devices()) - 1)
            )
            @test maxfree < 1.5 * need

            waits = Ref(0)
            on_wait = (elapsed, available, required) -> begin
                waits[] += 1
                @test available < 1.5 * required
                return nothing
            end
            (roi, info) = @test_logs (:warn, r"GPU unavailable") match_mode = :any getboxes(
                big, big_camera; backend = :auto, auto_timeout = 2.0,
                on_wait = on_wait, kw...,
            )
            @test info.backend == :cpu
            @test info.device_id == -1
            @test waits[] >= 1
            (roi_cpu, _) = getboxes(big, big_camera; backend = :cpu, kw...)
            @test roi.x_corners == roi_cpu.x_corners && roi.y_corners == roi_cpu.y_corners

            @test_throws ErrorException getboxes(
                big, big_camera; backend = :gpu, gpu_timeout = 2.0, kw...,
            )
        finally
            # Closing only the write end sends stdin EOF; wait for driver memory release.
            close(blocker.in)
            stopped = timedwait(() -> process_exited(blocker), 10.0) == :ok
            stopped || kill(blocker, Base.SIGKILL)
            wait(blocker)
            close(blocker)
            stopped || error("GPU blocker did not exit within 10 s of stdin EOF.\n" *
                read(stderr_path, String))
            success(blocker) || error("GPU blocker exited unsuccessfully.\n" *
                read(stderr_path, String))
        end
    end
end
