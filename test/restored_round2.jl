using Test, SMLMBoxer

# Exercise the real interface/retry source with host-backed device stubs. This
# isolates overrides and covers cleanup/fallback even on a machine without CUDA.
module RetryHarness
import SMLMBoxer
using SMLMData: AbstractCamera, IdealCamera, SCMOSCamera, ROIBatch
const BoxerConfig = SMLMBoxer.BoxerConfig
const BoxesInfo = SMLMBoxer.BoxesInfo
const GetBoxesArgs = SMLMBoxer.GetBoxesArgs
const CuArray = Array
const GPU_CONTEXT_RESERVE = SMLMBoxer.GPU_CONTEXT_RESERVE
const fault = Ref{Union{Exception,Nothing}}(nothing)
const normal_releases = Ref(0)
const failed_reclaims = Ref(0)
const cleanup_failed = Bool[]
module CUDA
devices() = [0]
totalmem(device) = 8 * 1024^3
device!(device) = nothing
free_memory() = 4 * 1024^3
end
has_cuda() = true
poll_gpu_nvml(required; kwargs...) = (true, 0)
release_gpu_memory() = (normal_releases[] += 1; nothing)
reclaim_failed_gpu_attempt() = (failed_reclaims[] += 1; nothing)
free_gpu_arrays(arrays...; failed=false) = (push!(cleanup_failed, failed); nothing)
reshape_for_flux(image) = SMLMBoxer.reshape_for_flux(image)
estimate_gpu_memory_per_frame(args...; kwargs...) = SMLMBoxer.estimate_gpu_memory_per_frame(args...; kwargs...)
frame_fits_some_gpu(args...) = SMLMBoxer.frame_fits_some_gpu(args...)
is_gpu_query_error(e) = SMLMBoxer.is_gpu_query_error(e)
is_gpu_resource_error(e) = SMLMBoxer.is_gpu_resource_error(e)
removeoverlap(args...) = SMLMBoxer.removeoverlap(args...)
getboxstack(args...) = SMLMBoxer.getboxstack(args...)
function dog_filter(image, args; use_gpu=false, kwargs...)
    use_gpu && fault[] !== nothing && throw(fault[])
    return SMLMBoxer.dog_filter(image, args; use_gpu=false, kwargs...)
end
findlocalmax(image, kernel; minval, use_gpu) =
    SMLMBoxer.findlocalmax(image, kernel; minval, use_gpu=false)
include(joinpath(dirname(pathof(SMLMBoxer)), "interface.jl"))
function reset!()
    fault[] = nothing
    normal_releases[] = failed_reclaims[] = 0
    empty!(cleanup_failed)
end
end

@testset "Round 2: input markers and GPU ArgumentError fallback" begin
    image = zeros(Float32, 32, 32, 3)
    image[16, 16, :] .= 2000f0
    reference, _ = getboxes(image; minval=1., backend=:cpu)
    RetryHarness.reset!()
    RetryHarness.fault[] = ArgumentError("injected GPU filter reason")
    args = SMLMBoxer.GetBoxesArgs(; imagestack=image, minval=1., backend=:auto)
    logs, result = Test.collect_test_logs() do
        try RetryHarness._getboxes_impl(args) catch e e end
    end
    @test result isa Tuple
    if result isa Tuple
        roi, info = result
        @test info.backend == :cpu
        @test roi.data == reference.data
        @test roi.x_corners == reference.x_corners
        @test roi.y_corners == reference.y_corners
        @test roi.frame_indices == reference.frame_indices
    end
    @test RetryHarness.failed_reclaims[] == 1
    @test any(log -> log.message == "GPU processing failed; using the CPU", logs)

    for backend in (:cpu, :auto, :gpu)
        RetryHarness.reset!()
        masked = copy(image)
        masked[4, 5, 2] = NaN32
        masked[6, 7, 3] = Inf32
        args = SMLMBoxer.GetBoxesArgs(; imagestack=masked, minval=1., backend)
        logs, err = Test.collect_test_logs() do
            try RetryHarness._getboxes_impl(args) catch e e end
        end
        @test err isa ArgumentError
        @test occursin("2 non-finite pixels", sprint(showerror, err))
        @test occursin("(4, 5, 2)", sprint(showerror, err))
        @test !any(log -> log.message == "GPU processing failed; using the CPU", logs)
        @test RetryHarness.normal_releases[] == (backend == :cpu ? 0 : 1)
        @test RetryHarness.failed_reclaims[] == 0
        @test all(!, RetryHarness.cleanup_failed)
        @test isequal(masked, args.imagestack)

        # Batch validation returns normally, including when a later batch is bad.
        result = try
            RetryHarness._process_with_batching(SMLMBoxer.reshape_for_flux(masked),
                args, 5, typemax(Int); use_gpu=false, max_batch_frames=1)
        catch e
            e
        end
        @test result isa Tuple && result[1] === nothing
    end
end

@testset "Round 2: keyword construction converts before validation" begin
    for boxsize in (9.0, Int32(9), Int16(9), 9//1, big(9))
        c = BoxerConfig(; boxsize)
        @test c.boxsize === 9
    end
    c = BoxerConfig(psf_sigma=Int32(1), min_photons=Int32(500),
        sigma_small=Int32(1), sigma_large=Int32(2), minval=Int32(0),
        boxsize=Int32(9), overlap=Int32(2), auto_timeout=Int32(0), gpu_timeout=Int32(0))
    @test c.psf_sigma === 1.0
    @test c.min_photons === 500.0
    @test c.sigma_small === 1.0 && c.sigma_large === 2.0
    @test c.minval === 0.0 && c.overlap === 2.0
    @test c.auto_timeout === c.gpu_timeout === 0.0
    @test_throws ArgumentError BoxerConfig(boxsize=0.0)
    @test_throws ArgumentError BoxerConfig(boxsize=Int32(-1))
end
