using Test, SMLMBoxer, SMLMData, CUDA

@testset "NVML memory query fallback" begin
    # Isolate NVML overrides so they cannot affect the rest of the suite.
    script = raw"""
    using SMLMBoxer
    using CUDA

    const nvml_error_code = Ref(CUDA.NVML.NVML_ERROR_NOT_SUPPORTED)
    CUDA.NVML.memory_info(dev::CUDA.NVML.Device) =
        throw(CUDA.NVML.NVMLError(nvml_error_code[]))
    CUDA.NVML.compute_processes(dev::CUDA.NVML.Device) =
        error("Unexpected NVML contention query")
    CUDA.NVML.utilization_rates(dev::CUDA.NVML.Device) =
        error("Unexpected NVML utilization query")

    CUDA.device!(0)
    j = length(CUDA.devices()) - 1
    free_mem = SMLMBoxer.gpu_free_memory(j)
    valid_memory = free_mem isa Integer &&
        0 < free_mem <= CUDA.totalmem(CuDevice(j))
    println("helper|memory=", valid_memory,
        "|restored=", CUDA.device() == CuDevice(0))

    function same_boxes(roi, reference)
        return roi.data == reference.data &&
            roi.x_corners == reference.x_corners &&
            roi.y_corners == reference.y_corners &&
            roi.frame_indices == reference.frame_indices
    end

    function run_backend(backend, image, reference, kw)
        try
            roi, info = getboxes(image; backend=backend,
                auto_timeout=10.0, gpu_timeout=10.0, kw...)
            return "$(info.backend),$(same_boxes(roi, reference))"
        catch e
            if e isa CUDA.NVML.NVMLError &&
                    e.code == CUDA.NVML.NVML_ERROR_UNKNOWN
                return "nvml_unknown"
            end
            showerror(stderr, e)
            println(stderr)
            return "error"
        end
    end

    image = zeros(Float32, 64, 64, 20)
    for frame in 1:20
        image[20, 20, frame] = 100
        image[45, 40, frame] = 75
    end
    kw = (boxsize=7, sigma_small=1.0, sigma_large=2.0, minval=0.5)
    reference, _ = getboxes(image; backend=:cpu, kw...)
    isempty(reference.data) && error("CPU reference contains no boxes")

    for (label, code) in (
            ("not_supported", CUDA.NVML.NVML_ERROR_NOT_SUPPORTED),
            ("other_failure", CUDA.NVML.NVML_ERROR_UNKNOWN))
        nvml_error_code[] = code
        auto_result = run_backend(:auto, image, reference, kw)
        gpu_result = run_backend(:gpu, image, reference, kw)
        println(label, "|auto=", auto_result, "|gpu=", gpu_result)
    end
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no
           --project=$(Base.active_project()) -e $script`
    depot_path = join(DEPOT_PATH, Sys.iswindows() ? ';' : ':')
    cmd = addenv(cmd, "JULIA_DEPOT_PATH" => depot_path)
    output = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=output))
    @test success(process)
    lines = split(String(take!(output)), '\n')
    @test "helper|memory=true|restored=true" in lines
    @test "not_supported|auto=gpu,true|gpu=gpu,true" in lines
    # Without a usable NVML (Tegra, no library) the NVML error is never raised
    other_failure = CUDA.NVML.has_nvml() ? "auto=cpu,true|gpu=nvml_unknown" :
        "auto=gpu,true|gpu=gpu,true"
    @test "other_failure|$other_failure" in lines
end
