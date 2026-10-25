# sCMOS cameras: scalar and per-pixel calibration, per-ROI calibration extraction, and the
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

@testset "Rectangular SCMOSCamera with per-pixel calibration" begin
    # Test extract_camera_roi directly with rectangular camera + per-pixel arrays
    # This verifies the (ny, nx) = (rows, cols) indexing convention in SMLMData 0.6+
    nrows, ncols = 80, 120

    # Create per-pixel readnoise array matching image convention (ny, nx) = (rows, cols)
    # Use spatially varying values to verify correct indexing
    readnoise_map = zeros(Float32, nrows, ncols)
    for r in 1:nrows, c in 1:ncols
        readnoise_map[r, c] = 1.0f0 + 0.01f0 * r + 0.001f0 * c  # Unique per pixel
    end

    pixel_size = 0.1f0
    camera = SCMOSCamera(
        ncols,  # npixels_x
        nrows,  # npixels_y
        pixel_size,
        readnoise_map,  # per-pixel readnoise
        offset = 100.0f0,
        gain = 2.0f0,
        qe = 0.9f0
    )

    # Extract a 7x7 ROI centered around row 40, col 60
    # ROI spans rows 37:43 (7 pixels), cols 57:63 (7 pixels)
    # For camera extraction, ranges include the +1 for pixel edges
    row_range = 37:44  # 8 elements for 7 pixels (edges)
    col_range = 57:64  # 8 elements for 7 pixels (edges)

    roi_camera = SMLMBoxer.extract_camera_roi(camera, row_range, col_range)

    @test roi_camera isa SCMOSCamera
    @test roi_camera.readnoise isa AbstractArray
    @test size(roi_camera.readnoise) == (7, 7)

    # Verify values are from correct region by checking the pattern
    # Original: readnoise[r,c] = 1.0 + 0.01*r + 0.001*c
    # ROI starts at row 37, col 57
    # So roi_readnoise[1,1] should be readnoise_map[37, 57] = 1.0 + 0.37 + 0.057 = 1.427
    expected_corner = 1.0f0 + 0.01f0 * 37 + 0.001f0 * 57
    @test roi_camera.readnoise[1, 1] ≈ expected_corner

    # Check center: roi_readnoise[4,4] should be readnoise_map[40, 60]
    # = 1.0 + 0.40 + 0.060 = 1.46
    expected_center = 1.0f0 + 0.01f0 * 40 + 0.001f0 * 60
    @test roi_camera.readnoise[4, 4] ≈ expected_center

    # Verify NOT transposed: if wrongly indexed, we'd get readnoise_map[57, 37] which
    # doesn't exist
    # (would error) or readnoise_map[col, row] giving wrong values
    # Check opposite corner: roi_readnoise[7,7] should be readnoise_map[43, 63]
    expected_opposite = 1.0f0 + 0.01f0 * 43 + 0.001f0 * 63
    @test roi_camera.readnoise[7, 7] ≈ expected_opposite
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
