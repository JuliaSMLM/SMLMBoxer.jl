using SMLMBoxer
using SMLMData
using CUDA
using Statistics
using Printf
using Test

@testset "SMLMBoxer.jl" begin

    @testset "API without camera" begin
        # Test image with two bright peaks
        image = zeros(Float32, 100, 100)
        image[20, 50] = 10
        image[30, 60] = 10

        # Get boxes without camera (positional interface)
        result = getboxes(image;
            boxsize=5,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.1,
            use_gpu=false
        )

        # Test NamedTuple structure
        @test haskey(result, :boxes)
        @test haskey(result, :coords_pixels)
        @test haskey(result, :metadata)

        # Should detect two peaks
        @test size(result.boxes) == (5, 5, 2)

        # Verify correct box location
        @test result.coords_pixels[1, :] ≈ [20, 50, 1]
        @test result.coords_pixels[2, :] ≈ [30, 60, 1]

        # Camera-specific fields should be nothing
        @test result.coords_microns === nothing
        @test result.camera_rois === nothing
    end

    @testset "Overlap removal" begin
        # Test image with two close bright peaks
        image = zeros(Float32, 100, 100)
        image[20, 50] = 20
        image[21, 51] = 10

        result = getboxes(image;
            boxsize=5,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.1,
            use_gpu=false
        )

        # Should detect only one peak (overlap removed)
        @test size(result.boxes) == (5, 5, 1)

        # Verify correct box location (should keep brighter peak)
        @test result.coords_pixels[1, :] ≈ [20, 50, 1]
    end

    @testset "New API with IdealCamera" begin
        # Test image with two bright peaks
        image = zeros(Float32, 100, 100)
        image[20, 50] = 10
        image[30, 60] = 10

        # Create an IdealCamera
        pixel_size = 0.1f0  # microns
        camera = IdealCamera(
            1:100,  # pixel range x
            1:100,  # pixel range y
            pixel_size  # pixel size
        )

        # Get boxes with camera
        result = getboxes(image, camera;
            boxsize=5,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.1,
            use_gpu=false
        )

        # Should detect two peaks
        @test size(result.boxes) == (5, 5, 2)

        # Check pixel coordinates
        @test result.coords_pixels[1, :] ≈ [20, 50, 1]
        @test result.coords_pixels[2, :] ≈ [30, 60, 1]

        # Check micron coordinates exist
        @test result.coords_microns !== nothing
        @test size(result.coords_microns) == (2, 2)

        # Verify micron conversion (center of pixel)
        expected_x1 = (camera.pixel_edges_x[50] + camera.pixel_edges_x[51]) / 2
        expected_y1 = (camera.pixel_edges_y[20] + camera.pixel_edges_y[21]) / 2
        @test result.coords_microns[1, 1] ≈ expected_x1
        @test result.coords_microns[1, 2] ≈ expected_y1

        # Check camera ROIs exist
        @test result.camera_rois !== nothing
        @test length(result.camera_rois) == 2
        @test result.camera_rois[1] isa IdealCamera
    end

    @testset "New API with SCMOSCamera (scalar params)" begin
        # Test image
        image = zeros(Float32, 100, 100)
        image[20, 50] = 10

        # Create SCMOSCamera with scalar parameters
        pixel_size = 0.1f0
        camera = SCMOSCamera(
            100,  # npixels_x
            100,  # npixels_y
            pixel_size,  # pixel size
            5.0f0,  # readnoise
            offset = 100.0f0,
            gain = 2.0f0,
            qe = 0.9f0
        )

        result = getboxes(image, camera;
            boxsize=5,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.1,
            use_gpu=false
        )

        # Should detect the peak
        @test size(result.boxes, 3) >= 1

        # Camera ROI should also be SCMOSCamera
        @test result.camera_rois[1] isa SCMOSCamera
        @test result.camera_rois[1].offset == 100.0f0
        @test result.camera_rois[1].gain == 2.0f0
    end

    @testset "sCMOS variance-weighted detection (per-pixel)" begin
        # Create image with two spots of equal intensity
        image = zeros(Float32, 100, 100)
        image[30, 30] = 100.0  # Spot in low-noise region
        image[70, 70] = 100.0  # Spot in high-noise region

        # Create per-pixel readnoise map
        readnoise_map = 2.0f0 .* ones(Float32, 100, 100)
        # Make one region very noisy
        readnoise_map[60:80, 60:80] .= 20.0f0  # 10x more noise

        pixel_size = 0.1f0
        camera = SCMOSCamera(
            100,  # npixels_x
            100,  # npixels_y
            pixel_size,  # pixel size
            readnoise_map,  # per-pixel readnoise
            offset = 100.0f0,
            gain = 2.0f0,
            qe = 0.9f0
        )

        result = getboxes(image, camera;
            boxsize=7,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.5,  # Threshold to potentially reject noisy spot
            use_gpu=false
        )

        # With variance weighting, the low-noise spot should be detected
        # The high-noise spot may or may not be detected depending on threshold
        @test size(result.boxes, 3) >= 1

        # Verify camera ROIs have correct per-pixel calibration
        if length(result.camera_rois) > 0
            @test result.camera_rois[1] isa SCMOSCamera
            @test result.camera_rois[1].readnoise isa AbstractArray
            @test size(result.camera_rois[1].readnoise) == (7, 7)
        end
    end

end

# Local performance benchmark (only runs in local environment, not on CI)
println()
println("="^70)
if get(ENV, "CI", "false") == "false"
    println("Local environment detected - running performance benchmark")
    println("="^70)
    include("local_performance_benchmark.jl")

    # Run the benchmark
    @testset "Local Performance Benchmark" begin
        results = run_comprehensive_benchmark()
        @test results !== nothing
        @test !isempty(results)

        # Validate that we got results for each configuration
        @test length(results) >= 5  # At least 5 test configs

        # Check that throughput values are positive
        @test all(r -> r.cpu_throughput > 0, results)

        # Check detection accuracy is reasonable (>50% of expected spots)
        for r in results
            accuracy = r.cpu_found / r.expected_spots
            @test accuracy > 0.5
        end

        # If GPU available, check GPU results
        if CUDA.functional()
            @test all(r -> r.gpu_throughput > 0, results)
            @test all(r -> r.speedup > 0, results)

            # GPU should generally be faster (speedup > 1.0) for larger images
            large_results = filter(r -> r.config.nx >= 256, results)
            if !isempty(large_results)
                @test any(r -> r.speedup > 1.0, large_results)
            end
        end
    end
else
    println("CI environment detected - skipping performance benchmark")
    println("To run performance benchmarks, execute tests locally:")
    println("  julia> using Pkg; Pkg.test(\"SMLMBoxer\")")
    println("="^70)
end