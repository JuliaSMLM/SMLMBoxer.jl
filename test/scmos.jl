# sCMOS cameras: scalar and per-pixel calibration and the
# variance-weighted DoG path, on the CPU backend.
using Test, SMLMBoxer, SMLMData
using SMLMSim
using MicroscopePSFs: GaussianPSF

function threshold_spots(camera; sigma=0.13, photons=500.0)
    emitters = [Emitter2DFit{Float64}((col-0.5)*0.1, 1.65, photons,
        0., 0., 0., 0., 0.; frame=1) for col in (17, 49)]
    image, _ = gen_images(BasicSMLD(emitters, camera, 1, 1), GaussianPSF(sigma);
        sampling=32, threaded=false)
    return Float32.(image .* camera.qe ./ camera.gain .+ camera.offset)
end

@testset "Item 7: photon threshold on both halves of a gain map" begin
    gain = fill(0.4, 33, 65); gain[:, 33:end] .= 0.5
    camera = SCMOSCamera(65, 33, 0.1, 1.6; gain)
    image = threshold_spots(camera)
    original = copy(image)
    for (threshold, expected) in ((495., 2), (505., 0))
        roi, _ = getboxes(image, camera; psf_sigma=0.13, min_photons=threshold, backend=:cpu)
        @test length(roi) == expected
        if expected == 2
            @test roi.x_corners == [14, 46]
            @test roi.y_corners == [14, 14]
        end
        for i in eachindex(roi.frame_indices)
            row, col = roi.y_corners[i], roi.x_corners[i]
            @test roi.data[:, :, i] == original[row:row+6, col:col+6, 1]
        end
        @test roi.camera === camera
        @test image == original
    end
    stack = repeat(image, 1, 1, 3)
    original_stack = copy(stack)
    args = SMLMBoxer.GetBoxesArgs(; imagestack=stack, camera, psf_sigma=0.13,
        min_photons=495., backend=:cpu)
    for cap in (1, 3)
        coords, batch_size, _, memory = SMLMBoxer._process_with_batching(
            SMLMBoxer.reshape_for_flux(stack), args, 5, typemax(Int);
            use_gpu=false, max_batch_frames=cap)
        @test [c[:, 1:3] for c in coords] ==
            [Float32[17 17 f; 17 49 f] for f in 1:3]
        @test stack == original_stack
        @test memory == batch_size * 33 * 65 * sizeof(Float32) * 9
    end
end

@testset "Item 7: scalar/map offset, gain and QE; raw ADU boxes" begin
    halves = fill(0.4, 33, 65); halves[:, 33:end] .= 0.5
    qe_map = fill(0.6, 33, 65); qe_map[:, 33:end] .= 0.9
    offset_map = fill(-25., 33, 65); offset_map[:, 33:end] .= 100.
    for gain in (0.46, halves), qe in (0.72, qe_map), offset in (-10., offset_map)
        camera = SCMOSCamera(65, 33, 0.1, 1.6; gain, qe, offset)
        image = threshold_spots(camera)
        original = copy(image)
        for (threshold, expected) in ((495., 2), (505., 0))
            roi, _ = getboxes(image, camera; psf_sigma=0.13, min_photons=threshold, backend=:cpu)
            @test length(roi) == expected
            @test roi.camera === camera
            @test image == original
            for i in eachindex(roi.frame_indices)
                row, col = roi.y_corners[i], roi.x_corners[i]
                @test roi.data[:, :, i] == original[row:row+6, col:col+6, 1]
            end
        end
        # Direct parameters still filter the supplied ADU with ADU² variance.
        args = SMLMBoxer.GetBoxesArgs(; imagestack=image, camera, backend=:cpu)
        stack = SMLMBoxer.reshape_for_flux(image)
        variance = Float32.(ones(33, 65) .* (camera.readnoise ./ camera.gain).^2)
        small = SMLMBoxer.convolve_variance_weighted(stack, variance, 1f0, false)
        large = SMLMBoxer.convolve_variance_weighted(stack, variance, 2f0, false)
        @test SMLMBoxer.dog_filter(stack, args; use_gpu=false) == small .- large
        photon_variance = SMLMBoxer.get_variance_map(camera, (33, 65); photons=true)
        @test photon_variance ≈ ones(33, 65) .* (camera.readnoise ./ camera.qe).^2
        @test eltype(photon_variance) == Float32
    end
end

@testset "Item 1 extension: finite offset when photon conversion uses it" begin
    image = ones(Float32, 33, 65)
    for bad in (NaN, Inf, -Inf), mapped in (false, true)
        offset = mapped ? zeros(33, 65) : bad
        mapped && (offset[17, 17] = bad)
        camera = SCMOSCamera(65, 33, 0.1, 1.6; offset)
        err = try
            getboxes(image, camera; psf_sigma=0.13, backend=:cpu)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("offset", sprint(showerror, err))
        @test occursin(string(mapped ? 1 : 33*65), sprint(showerror, err))
        @test occursin("finite", sprint(showerror, err))
        # The direct interface does not consume the offset.
        @test first(getboxes(image, camera; minval=1., backend=:cpu)) isa ROIBatch
    end
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
