using Test, SMLMBoxer, CUDA

@testset "GPU faults fall back immediately only under :auto" begin
    # Keep the injected fault and cleanup override out of the parent process.
    script = raw"""
    using SMLMBoxer, CUDA, Test

    function same_boxes(roi, reference)
        return roi.data == reference.data &&
            roi.x_corners == reference.x_corners &&
            roi.y_corners == reference.y_corners &&
            roi.frame_indices == reference.frame_indices
    end

    image = zeros(Float32, 64, 64, 2)
    image[24, 24, :] .= 1000f0
    image[44, 40, :] .= 800f0
    kw = (sigma_small=1.0, sigma_large=2.0, minval=1.0)
    reference, _ = getboxes(image; backend=:cpu, kw...)
    isempty(reference.data) && error("CPU reference contains no boxes")

    # Code 719 is CUDA_ERROR_LAUNCH_FAILED in CUDA 5.11 and 6.x.
    const fault = CUDA.CuError(typeof(CUDA.ERROR_OUT_OF_MEMORY)(719))
    @eval SMLMBoxer function convolve(imagestack::AbstractArray{<:Real},
            kernel::Matrix{Float32}; use_gpu=false)
        use_gpu && throw($fault)
        weights = reshape(kernel, size(kernel)..., 1, 1)
        p = size(kernel, 1) ÷ 2
        return NNlib.conv(imagestack, weights; pad=(p, p, p, p), stride=1)
    end

    # A sticky CUDA fault can make reclaim throw as well. It must not mask `fault`.
    @eval CUDA.reclaim() = error("injected cleanup failure")

    function check_auto(image, reference, kw)
        elapsed = @elapsed logs, _ = Test.collect_test_logs() do
            roi, info = getboxes(image; backend=:auto, auto_timeout=60.0, kw...)
            println("auto|backend=", info.backend, "|same=", same_boxes(roi, reference))
        end
        messages = [log.message for log in logs]
        println("auto|fast=", elapsed < 20,
            "|warning=", messages == ["GPU processing failed; using the CPU"])
    end

    function check_gpu(image, kw)
        caught = nothing
        trace = nothing
        elapsed = @elapsed try
            getboxes(image; backend=:gpu, gpu_timeout=60.0, kw...)
        catch e
            caught = e
            trace = catch_backtrace()
        end
        original = caught === fault
        from_convolve = trace !== nothing &&
            any(frame -> frame.func == :convolve, stacktrace(trace))
        println("gpu|original=", original, "|trace=", from_convolve,
            "|fast=", elapsed < 20)
    end

    check_auto(image, reference, kw)
    check_gpu(image, kw)
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no
           --project=$(Base.active_project()) -e $script`
    depot_path = join(DEPOT_PATH, Sys.iswindows() ? ';' : ':')
    output = IOBuffer()
    process = run(pipeline(ignorestatus(addenv(cmd, "JULIA_DEPOT_PATH" => depot_path));
        stdout=output))
    @test success(process)
    lines = split(String(take!(output)), '\n')
    @test "auto|backend=cpu|same=true" in lines
    @test "auto|fast=true|warning=true" in lines
    @test "gpu|original=true|trace=true|fast=true" in lines
end
