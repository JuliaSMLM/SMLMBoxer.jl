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
        roi_batch = getboxes(image;
            boxsize=5,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.1,
            use_gpu=false
        )

        # Test ROIBatch structure
        @test roi_batch isa ROIBatch
        @test hasfield(typeof(roi_batch), :data)
        @test hasfield(typeof(roi_batch), :x_corners)
        @test hasfield(typeof(roi_batch), :y_corners)
        @test hasfield(typeof(roi_batch), :frame_indices)
        @test hasfield(typeof(roi_batch), :camera)

        # Should detect two peaks
        @test size(roi_batch.data) == (5, 5, 2)
        @test length(roi_batch) == 2

        # Verify correct box locations (x_corners/y_corners are col/row of top-left corner)
        # For boxsize=5 and center at (row=20, col=50): corner = (50 - 5÷2, 20 - 5÷2) = (48, 18)
        @test roi_batch.x_corners[1] == 48  # x (col) of first ROI
        @test roi_batch.y_corners[1] == 18  # y (row) of first ROI
        @test roi_batch.x_corners[2] == 58  # x (col) of second ROI
        @test roi_batch.y_corners[2] == 28  # y (row) of second ROI
        @test roi_batch.frame_indices[1] == 1
        @test roi_batch.frame_indices[2] == 1

        # Default camera should be created
        @test roi_batch.camera isa IdealCamera
    end

    @testset "Overlap removal" begin
        # Test image with two close bright peaks
        image = zeros(Float32, 100, 100)
        image[20, 50] = 20
        image[21, 51] = 10

        roi_batch = getboxes(image;
            boxsize=5,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.1,
            use_gpu=false
        )

        # Should detect only one peak (overlap removed)
        @test size(roi_batch.data) == (5, 5, 1)
        @test length(roi_batch) == 1

        # Verify correct box location (should keep brighter peak)
        # For boxsize=5 and center at (row=20, col=50): corner = (50 - 5÷2, 20 - 5÷2) = (48, 18)
        @test roi_batch.x_corners[1] == 48  # x (col)
        @test roi_batch.y_corners[1] == 18  # y (row)
        @test roi_batch.frame_indices[1] == 1
    end

    @testset "New API with IdealCamera" begin
        # Test image with two bright peaks
        image = zeros(Float32, 100, 100)
        image[20, 50] = 10
        image[30, 60] = 10

        # Create an IdealCamera
        pixel_size = 0.1f0  # microns
        camera = IdealCamera(
            1:101,  # pixel range x (need 101 edges for 100 pixels)
            1:101,  # pixel range y
            pixel_size  # pixel size
        )

        # Get boxes with camera
        roi_batch = getboxes(image, camera;
            boxsize=5,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.1,
            use_gpu=false
        )

        # Should detect two peaks
        @test size(roi_batch.data) == (5, 5, 2)
        @test length(roi_batch) == 2

        # Check corner positions (top-left corner of ROI)
        # For boxsize=5 and center at (row=20, col=50): corner = (48, 18)
        @test roi_batch.x_corners[1] == 48  # x (col) of first ROI
        @test roi_batch.y_corners[1] == 18  # y (row) of first ROI
        @test roi_batch.x_corners[2] == 58  # x (col) of second ROI
        @test roi_batch.y_corners[2] == 28  # y (row) of second ROI

        # Check frame indices
        @test roi_batch.frame_indices[1] == 1
        @test roi_batch.frame_indices[2] == 1

        # Check camera is present and correct type
        @test roi_batch.camera isa IdealCamera
        @test roi_batch.camera === camera
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

        roi_batch = getboxes(image, camera;
            boxsize=5,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.1,
            use_gpu=false
        )

        # Should detect the peak
        @test size(roi_batch.data, 3) >= 1
        @test length(roi_batch) >= 1

        # Camera should be SCMOSCamera with correct parameters
        @test roi_batch.camera isa SCMOSCamera
        @test roi_batch.camera === camera
        @test roi_batch.camera.offset == 100.0f0
        @test roi_batch.camera.gain == 2.0f0
    end

    @testset "PSF-aware interface (physical units)" begin
        # Test image with a bright peak representing an emitter
        image = zeros(Float32, 100, 100)
        image[50, 50] = 1000.0  # ~1000 photon peak

        # Create camera
        pixel_size = 0.1f0  # 100nm pixels
        camera = IdealCamera(
            1:101,
            1:101,
            pixel_size
        )

        # Use PSF-aware interface with physical units (microns)
        psf_sigma_microns = 0.13f0  # 130nm PSF
        roi_batch = getboxes(image, camera;
            psf_sigma = psf_sigma_microns,  # In microns (auto-converts to pixels)
            min_photons = 500.0,             # Should detect our 1000 photon peak
            boxsize = 11,
            use_gpu = false
        )

        # Should detect the peak
        @test length(roi_batch) >= 1
        @test size(roi_batch.data, 3) >= 1

        # Verify the corner is correct
        # For boxsize=11 and center at (row=50, col=50): corner = (50 - 11÷2, 50 - 11÷2) = (45, 45)
        @test roi_batch.x_corners[1] == 45  # x (col)
        @test roi_batch.y_corners[1] == 45  # y (row)

        # Test with higher threshold - should not detect
        roi_batch_high = getboxes(image, camera;
            psf_sigma = psf_sigma_microns,
            min_photons = 5000.0,  # Way above our peak
            boxsize = 11,
            use_gpu = false
        )
        @test length(roi_batch_high) == 0
    end

    @testset "Backward compatibility (old interface)" begin
        # Verify old interface still works
        image = zeros(Float32, 100, 100)
        image[20, 50] = 10

        roi_batch = getboxes(image;
            boxsize = 5,
            sigma_small = 1.0,
            sigma_large = 2.0,
            minval = 0.1,
            use_gpu = false
        )

        @test length(roi_batch) >= 1
        @test size(roi_batch.data, 3) >= 1
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

        roi_batch = getboxes(image, camera;
            boxsize=7,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=0.5,  # Threshold to potentially reject noisy spot
            use_gpu=false
        )

        # With variance weighting, the low-noise spot should be detected
        # The high-noise spot may or may not be detected depending on threshold
        @test size(roi_batch.data, 3) >= 1
        @test length(roi_batch) >= 1

        # Verify camera has correct per-pixel calibration
        @test roi_batch.camera isa SCMOSCamera
        @test roi_batch.camera.readnoise isa AbstractArray
        @test size(roi_batch.camera.readnoise) == (100, 100)  # Full image readnoise map
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

    # GPU wait/timeout tests
    println()
    println("="^70)
    println("Running GPU wait/timeout tests")
    println("="^70)
    include("local_gpu_wait_test.jl")

    @testset "GPU Wait/Timeout Tests" begin
        @test run_gpu_wait_tests() == true

        # Memory pressure test (optional - may not trigger on high-memory GPUs)
        if CUDA.functional()
            @test test_memory_pressure_wait() == true
        end
    end
else
    println("CI environment detected - skipping performance benchmark")
    println("To run performance benchmarks, execute tests locally:")
    println("  julia> using Pkg; Pkg.test(\"SMLMBoxer\")")
    println("="^70)
end