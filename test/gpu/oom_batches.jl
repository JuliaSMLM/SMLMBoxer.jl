using Test, SMLMBoxer, CUDA

@testset "OOM retries reduce the batch under a hard memory limit" begin
    script = raw"""
    using SMLMBoxer, CUDA

    function same_boxes(roi, reference)
        return roi.data == reference.data &&
            roi.x_corners == reference.x_corners &&
            roi.y_corners == reference.y_corners &&
            roi.frame_indices == reference.frame_indices
    end

    camera = SMLMBoxer.IdealCamera(1:256, 1:256, 0.1f0)
    kw = (psf_sigma=0.135, min_photons=50.0)
    # Compile the small GPU path before measuring the large stack's retries.
    getboxes(zeros(Float32, 256, 256, 2), camera;
        backend=:gpu, gpu_timeout=120.0, kw...)

    image = zeros(Float32, 256, 256, 2000)
    for frame in axes(image, 3)
        image[64, 64, frame] = 2000f0
        image[160, 180, frame] = 1500f0
    end
    elapsed = @elapsed roi, info = getboxes(image, camera;
        backend=:gpu, gpu_timeout=120.0, kw...)
    reference, _ = getboxes(image, camera; backend=:cpu, kw...)
    isempty(reference.data) && error("CPU reference contains no boxes")
    println("limited|backend=", info.backend, "|batched=", info.n_batches > 1,
        "|same=", same_boxes(roi, reference), "|fast=", elapsed < 60)
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no
           --project=$(Base.active_project()) -e $script`
    depot_path = join(DEPOT_PATH, Sys.iswindows() ? ';' : ':')
    cmd = addenv(cmd, "JULIA_DEPOT_PATH" => depot_path,
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "1GiB")
    output = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=output))
    @test success(process)
    lines = split(String(take!(output)), '\n')
    @test "limited|backend=gpu|batched=true|same=true|fast=true" in lines
end
