using SMLMBoxer
using SMLMData
using Test

@testset "SMLMBoxer.jl" begin

    @testset "Legacy API (keyword-only, no camera)" begin
        # Test image with two bright peaks
        image = zeros(Float32, 100, 100)
        image[20, 50] = 10
        image[30, 60] = 10

        # Get boxes using legacy keyword interface
        result = getboxes(
            imagestack=image,
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

        result = getboxes(
            imagestack=image,
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
        pixel_size = 0.1  # microns
        camera = IdealCamera(
            pixel_edges_x = Float32.(0:pixel_size:100*pixel_size),
            pixel_edges_y = Float32.(0:pixel_size:100*pixel_size)
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
        pixel_size = 0.1
        camera = SCMOSCamera(
            pixel_edges_x = Float32.(0:pixel_size:100*pixel_size),
            pixel_edges_y = Float32.(0:pixel_size:100*pixel_size),
            offset = 100.0f0,
            gain = 2.0f0,
            readnoise = 5.0f0,
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

end