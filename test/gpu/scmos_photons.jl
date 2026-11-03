using Test, SMLMBoxer, SMLMData, SMLMSim, CUDA
using MicroscopePSFs: GaussianPSF

@testset "PSF-aware sCMOS photon detection agrees across CPU/GPU and batches" begin
    ideal = IdealCamera(65, 33, 0.1)
    emitters = [Emitter2DFit{Float64}((col-0.5)*0.1,1.65,500.,0.,0.,0.,0.,0.;frame=1)
        for col in (17,49)]
    photons, _ = gen_images(BasicSMLD(emitters, ideal, 1, 1), GaussianPSF(0.13);
        sampling=32, threaded=false)
    gain_map = fill(0.4,33,65); gain_map[:,33:end] .= 0.5
    qe_map = fill(0.6,33,65); qe_map[:,33:end] .= 0.9
    offset_map = fill(-25.,33,65); offset_map[:,33:end] .= 100.
    for gain in (0.46, gain_map), qe in (0.72, qe_map), offset in (-10., offset_map)
        camera = SCMOSCamera(65,33,0.1,1.6;gain,qe,offset)
        image = repeat(Float32.(photons .* qe ./ gain .+ offset),1,1,3)
        original = copy(image)
        for (threshold, expected) in ((495.,6),(505.,0))
            kw = (psf_sigma=0.13,min_photons=threshold)
            cpu, _ = getboxes(image,camera;backend=:cpu,kw...)
            @test length(cpu) == expected
            @test image == original
            for storage in (identity,CuArray)
                input = storage(image)
                try
                    gpu, info = getboxes(input,camera;backend=:gpu,kw...)
                    @test info.backend == :gpu
                    @test gpu.data == cpu.data
                    @test gpu.x_corners == cpu.x_corners
                    @test gpu.y_corners == cpu.y_corners
                    @test gpu.frame_indices == cpu.frame_indices
                    @test Array(input) == original
                finally
                    input isa CuArray && CUDA.unsafe_free!(input)
                end
            end
        end
        args = SMLMBoxer.GetBoxesArgs(;imagestack=image,camera,psf_sigma=0.13,
            min_photons=495.,backend=:gpu)
        input = CuArray(SMLMBoxer.reshape_for_flux(image))
        try
            reference = first(SMLMBoxer._process_with_batching(
                SMLMBoxer.reshape_for_flux(image),args,5,typemax(Int);
                use_gpu=false,max_batch_frames=1))
            for cap in (1,3)
                coords,_,_,_ = SMLMBoxer._process_with_batching(input,args,5,typemax(Int);
                    use_gpu=true,max_batch_frames=cap)
                @test all(size(c,1)==2 for c in coords)
                @test [c[:,1:3] for c in coords] == [c[:,1:3] for c in reference]
                @test Array(input) == reshape(image,33,65,1,3)
            end
            # Low-level calls also copy borrowed input before photon conversion.
            filtered = SMLMBoxer.dog_filter(input,args;use_gpu=true)
            CUDA.unsafe_free!(filtered)
            @test Array(input) == reshape(image,33,65,1,3)
        finally
            CUDA.unsafe_free!(input)
            SMLMBoxer.release_gpu_memory()
        end
    end
end
