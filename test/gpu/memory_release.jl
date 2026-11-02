using Test, SMLMBoxer, CUDA

@testset "Successful calls return GPU memory to the driver" begin
    script = raw"""
    using Test, SMLMBoxer, CUDA

    CUDA.NVML.has_nvml() || error("The memory release test requires NVML")
    dev = SMLMBoxer.nvml_device(0)
    function process_memory(dev)
        processes = CUDA.NVML.compute_processes(dev)
        memory = processes[getpid()].used_gpu_memory
        ismissing(memory) && error("NVML cannot report this process's GPU memory")
        return memory
    end

    function check_release(camera)
        kw = (minval=1.0, gpu_timeout=120.0, backend=:gpu)
        warm = zeros(Float32, 256, 256, 2)
        warm[128, 128, :] .= 2000f0
        getboxes(warm, camera; kw...)
        before = process_memory(dev)

        # 64 MiB input: about 384 MiB with the standard-path memory estimate.
        image = zeros(Float32, 256, 256, 256)
        image[128, 128, :] .= 2000f0
        roi, info = getboxes(image, camera; kw...)
        after = process_memory(dev)
        @test info.backend == :gpu
        @test length(roi) == 256
        println("release|camera=", typeof(camera), "|before=", before, "|after=", after)
        if v"6.5" <= pkgversion(CUDA) < v"6.6"
            # CHANGELOG known issue: CUDA.jl 6.5.x retains freed pool memory.
            @test_broken after <= before + 64 * 1024^2
        else
            @test after <= before + 64 * 1024^2
        end
    end

    camera = only(ARGS) == "scmos" ?
        SMLMBoxer.SCMOSCamera(256, 256, 0.1f0, 3f0) : nothing
    check_release(camera)
    """
    depot_path = join(DEPOT_PATH, Sys.iswindows() ? ';' : ':')
    # Pin both baseline and measured calls to the same device, even on a shared host.
    # CUDA_VISIBLE_DEVICES requires the GPU- prefix for a UUID.
    device_uuid = "GPU-" * string(CUDA.uuid(first(CUDA.devices())))
    # Separate processes keep retained memory from one path out of the next baseline.
    for camera in ("ideal", "scmos")
        @testset "$camera" begin
            cmd = `$(Base.julia_cmd()) --startup-file=no
                   --project=$(Base.active_project()) -e $script $camera`
            cmd = addenv(cmd, "JULIA_DEPOT_PATH" => depot_path,
                "CUDA_VISIBLE_DEVICES" => device_uuid)
            process = run(ignorestatus(cmd))
            @test success(process)
        end
    end
end

@testset "GPU inputs remain usable after temporary cleanup" begin
    image = zeros(Float32, 32, 32, 3)
    image[16, 16, :] .= 2000f0
    input = CuArray(image)
    stack = SMLMBoxer.reshape_for_flux(input)
    variance = CUDA.ones(Float32, 32, 32)
    args = SMLMBoxer.GetBoxesArgs(; imagestack=image, minval=1.0, backend=:cpu)
    try
        reference, _ = getboxes(image; backend=:cpu, minval=1.0)
        roi, info = getboxes(input; backend=:gpu, minval=1.0, gpu_timeout=120.0)
        @test info.backend == :gpu
        @test roi.data == reference.data
        @test roi.x_corners == reference.x_corners
        @test roi.y_corners == reference.y_corners
        @test roi.frame_indices == reference.frame_indices
        @test Array(input) == image

        coords, _, n_batches, _ = SMLMBoxer._process_with_batching(
            stack, args, 5, typemax(Int); use_gpu=true, max_batch_frames=1)
        @test n_batches == 3
        @test all(c -> size(c, 1) == 1, coords)
        @test Array(stack) == reshape(image, 32, 32, 1, 3)

        filtered = SMLMBoxer.convolve_variance_weighted(stack, variance, 1f0, true)
        CUDA.unsafe_free!(filtered)
        @test Array(variance) == ones(Float32, 32, 32)
        @test Array(input) == image

        # BigInt cannot be passed to a GPU kernel; pooling has already allocated.
        @test_throws Exception SMLMBoxer.genlocalmaximage(
            stack, 5; minval=big(1), use_gpu=true)
        @test Array(input) == image
    finally
        CUDA.unsafe_free!(variance)
        stack === input || CUDA.unsafe_free!(stack)
        CUDA.unsafe_free!(input)
        SMLMBoxer.release_gpu_memory()
    end
end
