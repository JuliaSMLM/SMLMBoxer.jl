using Test, SMLMBoxer, SMLMData, SMLMSim
using Markdown
using MicroscopePSFs: GaussianPSF

function simulated_spots(camera, spots; sigma=0.135, nframes=1, sampling=32)
    emitters = [
        Emitter2DFit{Float64}(
            (col - 0.5) * 0.1, (row - 0.5) * 0.1, photons,
            0.0, 0.0, 0.0, 0.0, 0.0; frame=frame
        ) for (row, col, photons, frame) in spots
    ]
    smld = BasicSMLD(emitters, camera, nframes, 1)
    images, _ = gen_images(smld, GaussianPSF(sigma); sampling, threaded=false)
    # Noise-free SMLMSim images are photons. Every sCMOS camera here has unit gain and QE and
    # zero offset, so photons are also counts; the counts convention itself is tested against
    # SMLMSim's camera noise model in test/scmos.jl.
    return Float32.(images)
end

@testset "0.4.0 regressions" begin
    @testset "C4: filter-derived photon threshold" begin
        cameras = (
            IdealCamera(45, 41, 0.1),
            SCMOSCamera(45, 41, 0.1, 1.6),
        )
        for camera in cameras, sigma in (0.02, 0.05, 0.135, 0.22)
            @testset "$(nameof(typeof(camera))) sigma=$sigma" begin
                image = simulated_spots(camera, [(21, 23, 500.0, 1)];
                    sigma, sampling=128)
                args = SMLMBoxer.GetBoxesArgs(; imagestack=image, camera,
                    psf_sigma=sigma, min_photons=500.0, backend=:cpu)
                filtered = SMLMBoxer.dog_filter(
                    SMLMBoxer.reshape_for_flux(image), args; use_gpu=false)
                # Also verify the numerical integration, beyond the 0.5% requirement.
                @test filtered[21, 23, 1, 1] ≈ args.minval rtol=1e-4
                if sigma == 0.135
                    factor = camera isa IdealCamera ? 0.547 : 0.581
                    expected = 500 * factor / (2π * (sqrt(2) * 1.35)^2)
                    # PSF-aware filtering and its threshold are now both in photons.
                    @test args.minval ≈ expected rtol=0.01
                end
                for (scale, expected) in ((1.02, 1), (0.98, 0))
                    rb, _ = getboxes(image .* scale, camera;
                        psf_sigma=sigma, min_photons=500, backend=:cpu)
                    @test length(rb) == expected
                end
            end
        end
        for camera in cameras
            @testset "$(nameof(typeof(camera))) pixel corner at sigma=1.35 px" begin
                # Half-integer pixel centers put the emitter on a corner of four pixels.
                photons = 500.0
                image = simulated_spots(camera, [(21.5, 23.5, photons, 1)];
                    sigma=0.135, sampling=128)
                args = SMLMBoxer.GetBoxesArgs(; imagestack=image, camera,
                    psf_sigma=0.135, min_photons=photons, backend=:cpu)
                filtered = SMLMBoxer.dog_filter(
                    SMLMBoxer.reshape_for_flux(image), args; use_gpu=false)
                peak_per_photon = maximum(filtered) / photons
                threshold_photons = args.minval / peak_per_photon
                @test 1.08 <= threshold_photons / photons <= 1.13
            end
        end
    end

    @testset "M1: larger overlap preserves resolvable neighbors on every frame" begin
        camera = IdealCamera(45, 41, 0.1)
        spots = [(21, col, 2000.0, f) for f in 1:3 for col in (20, 23)]
        image = simulated_spots(camera, spots; sigma=0.05, nframes=3)
        for overlap in (2.0, 3.0, 4.0, 10.0)
            rb, _ = getboxes(image, camera; sigma_small=0.8, sigma_large=1.6,
                minval=5.0, boxsize=7, overlap, backend=:cpu)
            @test [count(==(f), rb.frame_indices) for f in 1:3] == [2, 2, 2]
        end
    end

    @testset "M3: odd and even boxes at interior and edges" begin
        camera = IdealCamera(45, 41, 0.1)
        for (row, col) in ((21, 23), (1, 1), (41, 45))
            image = simulated_spots(camera, [(row, col, 2000.0, 1)]; sigma=0.05)
            for boxsize in (1, 2, 7, 8, 9, 10)
                @testset "center=($row, $col) boxsize=$boxsize" begin
                    rb, _ = getboxes(image, camera; sigma_small=0.5,
                        sigma_large=1.0, minval=5, boxsize, backend=:cpu)
                    r = clamp(row - (boxsize - 1) ÷ 2, 1, 41 - boxsize + 1)
                    c = clamp(col - (boxsize - 1) ÷ 2, 1, 45 - boxsize + 1)
                    @test length(rb) == 1
                    @test rb.x_corners == [c]
                    @test rb.y_corners == [r]
                    @test size(rb.data) == (boxsize, boxsize, 1)
                    @test rb.data[:, :, 1] == image[r:r+boxsize-1, c:c+boxsize-1, 1]
                    @test maximum(rb.data) == maximum(image)
                end
            end
        end
    end

    @testset "S2: camera dimensions checked before detection" begin
        for camera in (
            IdealCamera(45, 41, 0.1),
            SCMOSCamera(45, 41, 0.1, 1.6),
            SCMOSCamera(45, 41, 0.1, fill(1.6, 41, 45)),
        )
            for dims in ((20, 20), (45, 41)), backend in (:cpu, :auto, :gpu)
                image = zeros(Float32, dims)
                config = BoxerConfig(; backend)
                for call in (() -> getboxes(image, camera; backend),
                             () -> getboxes(image, camera, config))
                    error = try
                        call()
                        nothing
                    catch e
                        e
                    end
                    @test error isa DimensionMismatch
                    message = sprint(showerror, error)
                    @test occursin(string(dims), message)
                    @test occursin("(41, 45)", message)
                end
            end
        end
    end

    @testset "S3: one box per clamped corner and frame, brightest wins" begin
        camera = IdealCamera(45, 41, 0.1)
        spots = [(row, 1, photons, f)
            for f in 1:2 for (row, photons) in ((1, 1000.0), (4, 2000.0))]
        image = simulated_spots(camera, spots; sigma=0.05, nframes=2)
        rb, info = getboxes(image, camera; sigma_small=0.5, sigma_large=1.0,
            minval=5, boxsize=9, overlap=6, backend=:cpu)
        @test length(rb) == info.n_rois == 2
        @test rb.x_corners == rb.y_corners == [1, 1]
        @test rb.frame_indices == [1, 2]

        # The winner's place in detection order makes its identity observable.
        coords = Float32[1 1 1 10; 21 23 1 15; 4 1 1 20; 1 1 2 10]
        args = SMLMBoxer.GetBoxesArgs(; imagestack=image, camera, boxsize=9,
            backend=:cpu)
        data, corners = SMLMBoxer.getboxstack(
            SMLMBoxer.reshape_for_flux(image), coords, args)[1:2]
        @test corners == [17 19 1; 1 1 1; 1 1 2]
        @test size(data, 3) == 3
    end

    @testset "Note 5: variance in ADU squared for scalar and map calibration" begin
        gains = reshape(collect(range(0.4, 0.5; length=30)), 5, 6)
        noises = reshape(collect(range(1.0, 2.0; length=30)), 5, 6)
        for gain in (0.46, gains), readnoise in (1.6, noises)
            camera = SCMOSCamera(6, 5, 0.1, readnoise; gain)
            variance = SMLMBoxer.get_variance_map(camera, (5, 6))
            expected = ones(5, 6) .* (readnoise ./ gain).^2
            @test variance ≈ expected
            @test eltype(variance) == Float32
        end
    end

    @testset "S6: documented chunks retain global frame indices" begin
        camera = IdealCamera(45, 41, 0.1)
        image = simulated_spots(camera, [(21, 23, 2000.0, f) for f in 1:4];
            nframes=4)
        doc = string(@doc recommend_batch_size)
        example = match(r"for chunk_start in .*?\nend"s, doc).match
        example = replace(example, "# ... process results" =>
            "append!(frames, roi_batch.frame_indices)")
        Core.eval(@__MODULE__, quote
            function documented_chunks(image, camera)
                max_frames, total_frames = 2, size(image, 3)
                load_frames(indices) = image[:, :, indices]
                frames = Int32[]
                $(Meta.parse(example))
                return frames
            end
        end)
        @test Base.invokelatest(documented_chunks, image, camera) == [1, 2, 3, 4]
    end
end
