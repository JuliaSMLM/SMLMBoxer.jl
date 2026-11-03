using Test, SMLMBoxer, CUDA

@testset "Device finite check covers shapes, strided tails and non-finite values" begin
    for T in (Float16, Float32, Float64), dims in ((0,), (1,), (257,), (32, 33, 1, 2), (131089,))
        host = zeros(T, dims)
        if !isempty(host)
            host[begin] = floatmax(T)
            host[end] = -floatmax(T)
        end
        input = CuArray(host)
        try
            @test SMLMBoxer.pixels_are_finite(input)
            for bad in (T(NaN), T(Inf), T(-Inf))
                isempty(host) && continue
                # The last case passes the worker count twice, with a partial tail.
                for index in unique((1, min(length(host), 65537), length(host)))
                    previous = host[index]
                    host[index] = bad
                    copyto!(input, host)
                    @test !SMLMBoxer.pixels_are_finite(input)
                    @test isequal(Array(input), host)
                    host[index] = previous
                end
                copyto!(input, host)
                @test SMLMBoxer.pixels_are_finite(input)
            end
        finally
            CUDA.unsafe_free!(input)
        end
    end
end

@testset "Device finite check frees its workspace without a GC" begin
    # A fresh process isolates the pool baseline and the injected kernel error.
    script = raw"""
    using Test, SMLMBoxer, CUDA

    @testset "owned finite-check flag" begin
        input = CUDA.zeros(Float32, 256, 256, 1, 256)
        warm = CUDA.zeros(Float32, 256, 256, 1, 1)
        try
            # Same element type, dimensionality and worker count as the large check.
            @test SMLMBoxer.pixels_are_finite(warm)
            CUDA.unsafe_free!(warm)
            warm = nothing
            SMLMBoxer.release_gpu_memory()
            before_used = CUDA.used_memory()
            before_reserved = CUDA.cached_memory()
            before_used isa Integer && before_reserved isa Integer ||
                error("Finite-check pool test requires stream-ordered pool metrics")

            @test SMLMBoxer.pixels_are_finite(input)
            SMLMBoxer.release_gpu_memory()  # deliberately no GC
            @test CUDA.used_memory() <= before_used
            @test CUDA.cached_memory() <= before_reserved

            CUDA.@allowscalar input[end] = NaN32
            @test !SMLMBoxer.pixels_are_finite(input)
            SMLMBoxer.release_gpu_memory()
            @test CUDA.used_memory() <= before_used
            @test CUDA.cached_memory() <= before_reserved

            # Reporting an invalid caller-owned CuArray must leave no GPU diagnostic
            # reductions alive either; count and first coordinate remain correct.
            err = try SMLMBoxer.validate_finite_pixels(input) catch e e end
            @test err isa ArgumentError
            @test occursin("1 non-finite pixels", sprint(showerror, err))
            @test occursin("(256, 256, 256)", sprint(showerror, err))
            SMLMBoxer.release_gpu_memory()
            @test CUDA.used_memory() <= before_used
            @test CUDA.cached_memory() <= before_reserved

            # Fail after the flag allocation, preserving the original error and
            # releasing the owned flag even if the kernel cannot be launched.
            fault = ArgumentError("injected finite-check kernel failure")
            @eval SMLMBoxer finite_pixels_kernel!(::CUDABackend, ::Int) = throw($fault)
            err = try Base.invokelatest(SMLMBoxer.pixels_are_finite, input) catch e e end
            @test err === fault
            SMLMBoxer.release_gpu_memory()
            @test CUDA.used_memory() <= before_used
            @test CUDA.cached_memory() <= before_reserved
            println("finite-check|used_before=", before_used,
                "|used_after=", CUDA.used_memory(),
                "|reserved_before=", before_reserved,
                "|reserved_after=", CUDA.cached_memory())
        finally
            SMLMBoxer.free_gpu_arrays(warm, input)
            SMLMBoxer.release_gpu_memory()
        end
    end
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
    depot_path = join(DEPOT_PATH, Sys.iswindows() ? ';' : ':')
    device_uuid = "GPU-" * string(CUDA.uuid(first(CUDA.devices())))
    process = run(ignorestatus(addenv(cmd, "JULIA_DEPOT_PATH" => depot_path,
        "CUDA_VISIBLE_DEVICES" => device_uuid)))
    @test success(process)
end
