# sCMOS cameras: scalar and per-pixel calibration and the
# variance-weighted DoG path, on the CPU backend.
using Test, SMLMBoxer, SMLMData

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

    (roi_batch, info) = getboxes(
        image, camera;
        boxsize = 5,
        overlap = 3.0,
        sigma_small = 1.0,
        sigma_large = 2.0,
        minval = 0.1,
        backend = :cpu
    )

    # Should detect the peak
    @test size(roi_batch.data, 3) >= 1
    @test length(roi_batch) >= 1

    # Camera should be SCMOSCamera with correct parameters
    @test roi_batch.camera isa SCMOSCamera
    @test roi_batch.camera === camera
    @test roi_batch.camera.offset == 100.0f0
    @test roi_batch.camera.gain == 2.0f0

    # BoxesInfo should be valid
    @test info isa BoxesInfo
    @test info.elapsed_s > 0
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

    (roi_batch, info) = getboxes(
        image, camera;
        boxsize = 7,
        overlap = 3.0,
        sigma_small = 1.0,
        sigma_large = 2.0,
        minval = 0.5,  # Threshold to potentially reject noisy spot
        backend = :cpu
    )

    # With variance weighting, the low-noise spot should be detected
    # The high-noise spot may or may not be detected depending on threshold
    @test size(roi_batch.data, 3) >= 1
    @test length(roi_batch) >= 1

    # Verify camera has correct per-pixel calibration
    @test roi_batch.camera isa SCMOSCamera
    @test roi_batch.camera.readnoise isa AbstractArray
    @test size(roi_batch.camera.readnoise) == (100, 100)  # Full image readnoise map

    # BoxesInfo should be valid
    @test info isa BoxesInfo
    @test info.elapsed_s > 0
end
