# getboxes calling conventions, ROIBatch/BoxesInfo contents, overlap removal and the
# PSF-aware (physical units) interface, on the CPU backend.
using Test, SMLMBoxer, SMLMData

@testset "API without camera" begin
    # Test image with two bright peaks
    image = zeros(Float32, 100, 100)
    image[20, 50] = 10
    image[30, 60] = 10

    # Get boxes without camera (positional interface)
    (roi_batch, info) = getboxes(image;
        boxsize=5,
        overlap=3.0,
        sigma_small=1.0,
        sigma_large=2.0,
        minval=0.1,
        backend=:cpu
    )

    # Test ROIBatch structure
    @test roi_batch isa ROIBatch
    @test hasfield(typeof(roi_batch), :data)
    @test hasfield(typeof(roi_batch), :x_corners)
    @test hasfield(typeof(roi_batch), :y_corners)
    @test hasfield(typeof(roi_batch), :frame_indices)
    @test hasfield(typeof(roi_batch), :camera)

    # Test BoxesInfo structure
    @test info isa BoxesInfo
    @test info.backend == :cpu
    @test info.elapsed_s > 0
    @test info.device_id == -1  # CPU

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

    (roi_batch, info) = getboxes(image;
        boxsize=5,
        overlap=3.0,
        sigma_small=1.0,
        sigma_large=2.0,
        minval=0.1,
        backend=:cpu
    )

    # Should detect only one peak (overlap removed)
    @test size(roi_batch.data) == (5, 5, 1)
    @test length(roi_batch) == 1

    # Verify correct box location (should keep brighter peak)
    # For boxsize=5 and center at (row=20, col=50): corner = (50 - 5÷2, 20 - 5÷2) = (48, 18)
    @test roi_batch.x_corners[1] == 48  # x (col)
    @test roi_batch.y_corners[1] == 18  # y (row)
    @test roi_batch.frame_indices[1] == 1

    # BoxesInfo should be valid
    @test info isa BoxesInfo
    @test info.elapsed_s > 0
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
    (roi_batch, info) = getboxes(image, camera;
        boxsize=5,
        overlap=3.0,
        sigma_small=1.0,
        sigma_large=2.0,
        minval=0.1,
        backend=:cpu
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

    # BoxesInfo should be valid
    @test info isa BoxesInfo
    @test info.elapsed_s > 0
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
    (roi_batch, info) = getboxes(image, camera;
        psf_sigma = psf_sigma_microns,  # In microns (auto-converts to pixels)
        min_photons = 500.0,             # Should detect our 1000 photon peak
        boxsize = 11,
        backend = :cpu
    )

    # Should detect the peak
    @test length(roi_batch) >= 1
    @test size(roi_batch.data, 3) >= 1

    # Verify the corner is correct
    # For boxsize=11 and center at (row=50, col=50): corner = (50 - 11÷2, 50 - 11÷2) = (45, 45)
    @test roi_batch.x_corners[1] == 45  # x (col)
    @test roi_batch.y_corners[1] == 45  # y (row)

    # BoxesInfo should be valid
    @test info isa BoxesInfo
    @test info.elapsed_s > 0

    # Test with higher threshold - should not detect
    (roi_batch_high, _) = getboxes(image, camera;
        psf_sigma = psf_sigma_microns,
        min_photons = 5000.0,  # Way above our peak
        boxsize = 11,
        backend = :cpu
    )
    @test length(roi_batch_high) == 0
end
@testset "BoxerConfig calling convention" begin
    # Test config-based calling
    image = zeros(Float32, 100, 100)
    image[50, 50] = 1000.0

    camera = IdealCamera(1:101, 1:101, 0.1f0)

    # PSF-aware config
    config_psf = BoxerConfig(psf_sigma=0.13, min_photons=500.0, boxsize=11)
    @test config_psf isa BoxerConfig
    @test config_psf.psf_sigma == 0.13
    @test config_psf.boxsize == 11

    (roi_batch, info) = getboxes(image, camera, config_psf)
    @test length(roi_batch) >= 1
    @test info isa BoxesInfo

    # Advanced config (sigma_small/sigma_large)
    config_adv = BoxerConfig(sigma_small=1.5, sigma_large=3.0, minval=0.1, boxsize=7, backend=:cpu)
    @test config_adv.psf_sigma === nothing
    @test config_adv.sigma_small == 1.5
    @test config_adv.backend == :cpu

    image2 = zeros(Float32, 100, 100)
    image2[20, 50] = 10

    (roi_batch2, info2) = getboxes(image2, nothing, config_adv)
    @test info2.backend == :cpu

    # Kwargs should produce same result as config
    (roi_batch3, info3) = getboxes(image2;
        sigma_small=1.5, sigma_large=3.0, minval=0.1, boxsize=7, backend=:cpu)
    @test length(roi_batch2) == length(roi_batch3)
end
