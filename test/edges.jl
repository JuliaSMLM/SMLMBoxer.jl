# Standard DoG normalizes each Gaussian by its in-bounds weights at image edges.
using Test, SMLMBoxer, SMLMData, Random

@testset "Flat backgrounds produce no boxes" begin
    image = fill(100f0, 128, 128, 3)
    camera = IdealCamera(128, 128, 0.1f0)

    roi_camera, _ = getboxes(image, camera;
        psf_sigma = 0.13, min_photons = 500, backend = :cpu)
    @test length(roi_camera) == 0

    roi_default, _ = getboxes(image;
        sigma_small = 1.3, sigma_large = 2.6, minval = 1.0, backend = :cpu)
    @test length(roi_default) == 0
end

@testset "Spots near an edge and in the interior" begin
    image = fill(100f0, 128, 128, 3)
    sigma = 1.3f0
    # Column 4 is three pixels from the left edge. Use different frames to check indexing.
    for (row, col, frame) in ((32, 4, 1), (64, 64, 3))
        spot = [exp(-((r - row)^2 + (c - col)^2) / (2 * sigma^2))
                for r in 1:128, c in 1:128]
        image[:, :, frame] .+= (3000f0 / sum(spot)) .* spot
    end

    for camera in (IdealCamera(128, 128, 0.1f0), nothing)
        roi_batch, _ = if camera === nothing
            getboxes(image; sigma_small = 1.3, sigma_large = 2.6, minval = 1.0,
                boxsize = 7, backend = :cpu)
        else
            getboxes(image, camera; psf_sigma = 0.13, min_photons = 500,
                boxsize = 7, backend = :cpu)
        end
        @test length(roi_batch) == 2
        @test Set(zip(roi_batch.x_corners, roi_batch.y_corners, roi_batch.frame_indices)) ==
            Set(((1, 29, 1), (61, 61, 3)))
    end
end

@testset "Standard DoG preserves interior convolution exactly" begin
    image = 50f0 .+ rand(MersenneTwister(42), Float32, 64, 64, 4)
    x = SMLMBoxer.reshape_for_flux(image)
    dog = SMLMBoxer.dog_kernel(1.3f0, 2.6f0)
    original = SMLMBoxer.convolve(x, dog; use_gpu = false)
    p = size(dog, 1) ÷ 2
    interior = (p + 1):(64 - p)
    border = [i <= p || i > 64 - p || j <= p || j > 64 - p
              for i in 1:64, j in 1:64, _ in 1:1, _ in 1:4]

    for camera in (IdealCamera(64, 64, 0.1f0), nothing)
        args = SMLMBoxer.GetBoxesArgs(; imagestack = image, camera,
            sigma_small = 1.3, sigma_large = 2.6, backend = :cpu)
        filtered = SMLMBoxer.dog_filter(x, args; use_gpu = false)
        @test filtered[interior, interior, :, :] == original[interior, interior, :, :]
        @test filtered[border] != original[border]
    end
end

@testset "Standard DoG on images smaller than its window" begin
    rng = MersenneTwister(43)
    ks, kl = SMLMBoxer.dog_kernel_pair(1.3f0, 2.6f0)
    p = size(ks, 1) ÷ 2
    for (nrows, ncols) in ((1, 1), (1, 20), (20, 1), (3, 4), (10, 17), (17, 10))
        image = 50f0 .+ rand(rng, Float32, nrows, ncols, 2)
        args = SMLMBoxer.GetBoxesArgs(; imagestack = image,
            sigma_small = 1.3, sigma_large = 2.6, backend = :cpu)
        filtered = SMLMBoxer.dog_filter(SMLMBoxer.reshape_for_flux(image), args)
        # Independent reference: crop each Gaussian to the available image, then normalize.
        expected = similar(filtered)
        for f in 1:2, j in 1:ncols, i in 1:nrows
            rows = max(1, i - p):min(nrows, i + p)
            cols = max(1, j - p):min(ncols, j + p)
            weights_small = ks[p + 1 .+ i .- rows, p + 1 .+ j .- cols]
            weights_large = kl[p + 1 .+ i .- rows, p + 1 .+ j .- cols]
            pixels = image[rows, cols, f]
            expected[i, j, 1, f] = sum(weights_small .* pixels) / sum(weights_small) -
                sum(weights_large .* pixels) / sum(weights_large)
        end
        @test all(isapprox.(filtered, expected; atol = 1f-4, rtol = 0))
    end
end
