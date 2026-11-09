using Test, SMLMBoxer, SMLMData, SMLMSim, Random
using MicroscopePSFs: GaussianPSF

function restored_image(camera; camera_noise=false)
    emitter = Emitter2DFit{Float64}(1.65, 1.65, 2000., 0., 0., 0., 0., 0.; frame=1)
    image, _ = gen_images(BasicSMLD([emitter], camera, 1, 1), GaussianPSF(0.13);
        camera_noise, threaded=false, sampling=32)
    return Float32.(image)
end

function argument_message(f, parts...)
    err = try
        f()
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    message = sprint(showerror, err)
    for part in parts
        @test occursin(part, message)
    end
end

@testset "0.4.1 restored bugs" begin
    camera = IdealCamera(33, 33, 0.1)
    image = restored_image(camera)

    @testset "M2: positive finite sCMOS calibration" begin
        Random.seed!(41)
        for field in (:readnoise, :gain, :qe), bad in (0., -1., NaN, Inf), mapped in (false, true)
            value = mapped ? ones(33, 33) : bad
            mapped && (value[17, 17] = bad)
            kw = merge((gain=1., qe=1.), NamedTuple{(field,)}((value,)))
            readnoise = get(kw, :readnoise, 1.)
            cam = SCMOSCamera(33, 33, 0.1, readnoise; gain=kw.gain, qe=kw.qe)
            affected = mapped ? 1 : 33 * 33
            for psf_sigma in (nothing, 0.13)
                argument_message(() -> getboxes(image, cam; psf_sigma, backend=:cpu),
                    string(field), string(affected), "IdealCamera", "photons")
            end
        end
        zero_cam = SCMOSCamera(33, 33, 0.1, 0.)
        noisy = restored_image(zero_cam; camera_noise=true)
        @test all(isfinite, noisy)
        argument_message(() -> getboxes(noisy, zero_cam; backend=:cpu), "readnoise")
        # Noise-free readout remains valid through the photon interface.
        @test length(first(getboxes(image, camera; psf_sigma=0.13, backend=:cpu))) == 1
    end

    @testset "Note 2: non-finite pixels report count and first coordinate" begin
        for backend in (:cpu, :auto), bad in (NaN32, Inf32, -Inf32)
            masked = repeat(image, 1, 1, 3)
            masked[4, 5, 2] = bad
            masked[6, 7, 3] = bad
            argument_message(() -> getboxes(masked, camera; backend, auto_timeout=0, gpu_timeout=0),
                "2", "(4, 5, 2)")
        end
        masked = copy(image[:, :, 1]); masked[3, 7] = NaN32
        argument_message(() -> getboxes(masked; backend=:cpu), "1", "(3, 7, 1)")
        # Includes conversion overflow rather than accepting Inf generated at entry.
        overflowed = Float64.(image); overflowed[2, 3, 1] = 1e300
        argument_message(() -> getboxes(overflowed; backend=:cpu), "(2, 3, 1)")
        integer_image = round.(UInt16, image)
        rb, _ = getboxes(integer_image, camera; psf_sigma=0.13, backend=:cpu)
        @test length(rb) == 1
        @test rb.data[:, :, 1] == Float32.(integer_image[14:20, 14:20, 1])
    end

    @testset "S1: photon threshold requires PSF sigma" begin
        argument_message(() -> BoxerConfig(min_photons=1e9), "min_photons", "psf_sigma")
        argument_message(() -> getboxes(image, camera; min_photons=1e9, backend=:cpu),
            "min_photons", "psf_sigma")
        argument_message(() -> SMLMBoxer.GetBoxesArgs(; imagestack=image, min_photons=500.),
            "min_photons", "psf_sigma")
        @test BoxerConfig().min_photons === nothing
        config = BoxerConfig(psf_sigma=0.13, backend=:cpu)
        @test config.min_photons === 500.0
        @test occursin("min_photons=500.0", sprint(show, config))
        default, _ = getboxes(image, camera, config)
        explicit, _ = getboxes(image, camera; psf_sigma=0.13, min_photons=500., backend=:cpu)
        @test default.data == explicit.data
        @test default.x_corners == explicit.x_corners
        @test default.y_corners == explicit.y_corners
    end

    @testset "S4: small, unsupported and empty stacks" begin
        for dims in ((3, 33), (33, 3), (0, 33), (33, 0),
                     (33, 33, 1, 2), (33, 33, 2, 1), (33, 33, 1, 1, 1))
            @test_throws ArgumentError getboxes(zeros(Float32, dims); backend=:cpu)
        end
        for camera_arg in (nothing, camera, SCMOSCamera(33, 33, 0.1, 1.)), backend in (:cpu, :auto, :gpu)
            rb, info = getboxes(zeros(Float32, 33, 33, 0), camera_arg; backend)
            @test rb isa ROIBatch
            @test info isa BoxesInfo
            @test size(rb.data) == (7, 7, 0)
            @test isempty(rb.x_corners) && isempty(rb.y_corners) && isempty(rb.frame_indices)
            @test info.n_rois == info.n_batches == info.batch_size == info.memory_per_batch == 0
            @test info.backend == :cpu && info.device_id == -1
            @test info.elapsed_s >= 0
            camera_arg === nothing || @test rb.camera === camera_arg
        end
    end

    @testset "S5: shared keyword parameter validation" begin
        cases = [
            (sigma_small=0.,), (sigma_small=-1.,), (sigma_small=NaN,), (sigma_small=Inf,),
            (sigma_large=0.,), (sigma_large=NaN,), (sigma_large=Inf,),
            (sigma_small=2.,sigma_large=2.), (sigma_small=3.,sigma_large=2.),
            (psf_sigma=0.,), (psf_sigma=-1.,), (psf_sigma=NaN,), (psf_sigma=Inf,),
            (psf_sigma=0.13,min_photons=0.), (psf_sigma=0.13,min_photons=-1.),
            (psf_sigma=0.13,min_photons=NaN,), (psf_sigma=0.13,min_photons=Inf,),
            (minval=NaN,), (minval=Inf,), (minval=-Inf,), (boxsize=0,), (boxsize=-1,),
            (overlap=-1.,), (overlap=NaN,), (overlap=Inf,), (backend=:wrong,),
            (auto_timeout=-1.,), (auto_timeout=NaN,), (gpu_timeout=-1.,), (gpu_timeout=NaN,),
            (on_wait=42,),
        ]
        for kw in cases
            field = haskey(kw, :min_photons) ? :min_photons : first(keys(kw))
            for call in (() -> BoxerConfig(;kw...),
                         () -> getboxes(image, camera; kw...),
                         () -> SMLMBoxer.GetBoxesArgs(; imagestack=image, camera, kw...))
                argument_message(call, string(field), string(kw[field]))
            end
        end
        for timeout in (0., Inf)
            @test BoxerConfig(auto_timeout=timeout, gpu_timeout=timeout) isa BoxerConfig
        end
        @test BoxerConfig(minval=-10., overlap=1e30) isa BoxerConfig
        @test length(first(getboxes(image; overlap=1e30, minval=1, backend=:cpu))) == 1
        argument_message(() -> getboxes(image; psf_sigma=0.13, backend=:cpu), "camera", "psf_sigma")
    end

    @testset "Note 5: PSF conversion requires uniform square pixels" begin
        edges = collect(0.:0.1:3.3)
        rectangular = IdealCamera(edges, 2 .* edges)
        unequal_edges = copy(edges); unequal_edges[18:end] .+= 0.01
        unequal = IdealCamera(unequal_edges, edges)
        # Widths close to the first x pixel can still differ from each other by > tolerance.
        opposite_errors = IdealCamera(vcat(0., cumsum(vcat(0.1, fill(0.100009, 32)))),
            collect(0:33) .* 0.099991)
        for cam in (rectangular, unequal,
                    opposite_errors,
                    SCMOSCamera(edges, 2 .* edges; readnoise=1.))
            argument_message(() -> getboxes(image, cam; psf_sigma=0.13, backend=:cpu),
                "square", "sigma_small", "sigma_large", "pixels")
            # Direct parameters remain available on the same camera and input.
            @test length(first(getboxes(image, cam; minval=1, backend=:cpu))) == 1
        end
        for factor in (1., 1. + 5e-5)
            cam = IdealCamera(edges, factor .* edges)
            @test length(first(getboxes(image, cam; psf_sigma=0.13, backend=:cpu))) == 1
        end
        # Uniform cameras stored in Float32: rounding of the stored edges alone makes widths of a
        # 2048-pixel camera differ by more than 1e-4 relative; they are uniform and are accepted.
        for p in (0.1f0, 0.065f0), cam in (IdealCamera(2048, 2048, p), SCMOSCamera(2048, 2048, p, 1.6f0))
            @test SMLMBoxer.get_pixel_size(cam) ≈ p rtol=1e-6
        end
        # Far from the origin, Float32 rounding moves each width by up to 6e-4 of a pixel; the
        # camera is uniform within that rounding, is accepted, and a spot simulated there is found.
        far = Float32.(1000 .+ (0:33) .* 0.1)
        far_camera = IdealCamera(far, far)
        @test SMLMBoxer.get_pixel_size(far_camera) ≈ 0.1 rtol=1e-3
        far_spot = Emitter2DFit{Float64}(1001.65, 1001.65, 2000., 0., 0., 0., 0., 0.; frame=1)
        far_image, _ = gen_images(BasicSMLD([far_spot], far_camera, 1, 1), GaussianPSF(0.13);
            threaded=false, sampling=32)
        rois, _ = getboxes(Float32.(far_image), far_camera; psf_sigma=0.13, backend=:cpu)
        @test length(rois) == 1
        # Cameras built from a pixel count and size keep 0.4.0's pixel size (the first width), so
        # their filters and boxes do not change; that includes unequal counts in Float16.
        for cam in (IdealCamera(33, 33, 0.16f0), IdealCamera(2048, 2048, 0.1f0),
                    IdealCamera(32, 33, Float16(0.1)), SCMOSCamera(32, 33, Float16(0.1), Float16(1.6)))
            @test SMLMBoxer.get_pixel_size(cam) == Float64(cam.pixel_edges_x[2] - cam.pixel_edges_x[1])
        end
        # Edge rounding is allowed per pixel and per axis only: pixels that differ between x and
        # y, a width more than 1e-4 from its axis's pitch, and a backward edge are refused.
        for cam in (IdealCamera(2048, 2048, (0.1f0, 0.10002f0)),
                    SCMOSCamera(2048, 2048, (0.1f0, 0.10002f0), 1.6f0),
                    IdealCamera(1024, 1024, (Float16(0.0625), Float16(0.125))),
                    IdealCamera(vcat(0., cumsum(repeat([0.099985, 0.100015], 4))), collect(0:8) .* 0.1),
                    IdealCamera(Float16[1024, 1025, 1024, 1025, 1026, 1027, 1028, 1029],
                                Float16[1024, 1025, 1024, 1025, 1026, 1027, 1028, 1029]))
            argument_message(() -> SMLMBoxer.get_pixel_size(cam), "square")
        end
        # The rounding allowance is each width's own edges, not the largest edge's: a first width
        # 2e-4 too wide near the origin is refused although the far edges round by more.
        wide_first = Float32.(0:2048) .* 0.1f0
        wide_first[2] = 0.10002f0
        argument_message(() -> SMLMBoxer.get_pixel_size(IdealCamera(wide_first, wide_first)), "square")
    end

    @testset "Low-precision sCMOS calibration does not overflow the variance" begin
        # (readnoise / qe)^2 and (readnoise / gain)^2 = 160,000 are above Float16's largest value.
        for mapped in (false, true)
            noise = mapped ? fill(Float16(20), 9, 9) : Float16(20)
            divisor = mapped ? fill(Float16(0.05), 9, 9) : Float16(0.05)
            cam = SCMOSCamera(9, 9, Float16(0.125), noise; gain=divisor, qe=divisor)
            for photons in (true, false)
                variance = SMLMBoxer.get_variance_map(cam, (9, 9); photons)
                @test all(isfinite, variance)
                @test variance[5, 5] ≈ 160_000 rtol=1e-3
            end
        end
        # Float32 calibration keeps 0.4.0's Float32 arithmetic bit for bit (Float64 would round
        # this one differently), so its boxes do not change.
        cam32 = SCMOSCamera(9, 9, 0.125f0, 0.3f0; gain=0.46f0)
        @test SMLMBoxer.get_variance_map(cam32, (9, 9))[5, 5] === (0.3f0 / 0.46f0)^2
    end
end
