using Test, SMLMBoxer, CUDA, Random, NNlib, KernelAbstractions

@testset "GPU kernels match the CPU" begin
    # The GPU DoG conv and maxpool are KernelAbstractions kernels (no cuDNN); compare
    # them against the NNlib CPU path they replaced.
    Random.seed!(42)
    x = SMLMBoxer.reshape_for_flux(rand(Float32, 64, 64, 5))
    dog = SMLMBoxer.dog_kernel(1.3f0, 2.6f0)

    # Convolution: KA kernel (GPU) vs NNlib (CPU)
    conv_cpu = SMLMBoxer.convolve(x, dog; use_gpu = false)
    conv_gpu = Array(SMLMBoxer.convolve(x, dog; use_gpu = true))
    @test isapprox(conv_cpu, conv_gpu; atol = 1.0f-4)  # observed max |diff| is 6.7e-6

    # Symmetric DoG kernels cannot detect a missing kernel flip.
    asymmetric = rand(MersenneTwister(17), Float32, 5, 5)
    asymmetric_cpu = SMLMBoxer.convolve(x, asymmetric; use_gpu = false)
    asymmetric_gpu = Array(SMLMBoxer.convolve(x, asymmetric; use_gpu = true))
    # Elementwise (values near 5; a missing flip differs by order 1)
    @test maximum(abs.(asymmetric_cpu .- asymmetric_gpu)) < 1.0f-4

    # Max pooling: KA kernel (GPU) vs NNlib (CPU), odd (5) and even (4) windows,
    # on the same CPU-filtered input.
    y = copy(conv_cpu)
    y[32, 32, 1, 1] = NaN32
    y_gpu = CuArray(y)
    for k in (5, 4)
        max_cpu = SMLMBoxer.genlocalmaximage(y, k; minval = 0.0, use_gpu = false)
        max_gpu = Array(SMLMBoxer.genlocalmaximage(y, k; minval = 0.0, use_gpu = true))
        @test isequal(max_cpu, max_gpu)

        # Also compare the raw (unmasked) max window, not only the thresholded image
        plo = (k - 1) ÷ 2
        pad = isodd(k) ? (k ÷ 2, k ÷ 2, k ÷ 2, k ÷ 2) : (plo, k ÷ 2, plo, k ÷ 2)
        raw_cpu = NNlib.maxpool(y, (k, k); pad = pad, stride = 1)
        raw_gpu = CUDA.zeros(Float32, size(y))
        kernel! = SMLMBoxer.localmax_kernel!(CUDABackend())
        kernel!(raw_gpu, y_gpu, plo, k; ndrange = size(y)[[1, 2, 4]])
        KernelAbstractions.synchronize(CUDABackend())
        @test isequal(raw_cpu, Array(raw_gpu))
    end
end

@testset "IdealCamera boxes near a NaN pixel match the CPU" begin
    image = 20f0 .+ 3f0 .* randn(MersenneTwister(21), Float32, 96, 96, 2)
    sigma = 1.35f0
    spots = [(30, 30), (40, 52), (60, 60), (47, 41), (80, 20)]
    for f in 1:2, (r, c) in spots, i in (r - 5):(r + 5), j in (c - 5):(c + 5)
        image[i, j, f] += 2000f0 *
            exp(-((i - r)^2 + (j - c)^2) / (2sigma^2)) / (2π * sigma^2)
    end
    image[45, 45, 1] = NaN32
    camera = SMLMBoxer.IdealCamera(96, 96, 0.1f0)
    kw = (psf_sigma = 0.135, min_photons = 500.0, boxsize = 7)
    cpu, _ = getboxes(image, camera; backend = :cpu, kw...)
    gpu, info = getboxes(image, camera; backend = :gpu, kw...)
    @test info.backend == :gpu
    @test !isempty(cpu.data)
    @test isequal(gpu.data, cpu.data)
    @test gpu.x_corners == cpu.x_corners
    @test gpu.y_corners == cpu.y_corners
    @test gpu.frame_indices == cpu.frame_indices
end

@testset "Standard DoG borders match the CPU and preserve GPU interiors" begin
    image = 50f0 .+ rand(MersenneTwister(42), Float32, 64, 64, 4)
    x = SMLMBoxer.reshape_for_flux(image)
    args = SMLMBoxer.GetBoxesArgs(; imagestack = image,
        sigma_small = 1.3, sigma_large = 2.6, backend = :cpu)
    dog = SMLMBoxer.dog_kernel(1.3f0, 2.6f0)
    filtered_cpu = SMLMBoxer.dog_filter(x, args; use_gpu = false)
    filtered_gpu = Array(SMLMBoxer.dog_filter(x, args; use_gpu = true))
    original_gpu = Array(SMLMBoxer.convolve(CuArray(x), dog; use_gpu = true))

    p = size(dog, 1) ÷ 2
    interior = (p + 1):(64 - p)
    border = [i <= p || i > 64 - p || j <= p || j > 64 - p
              for i in 1:64, j in 1:64, _ in 1:1, _ in 1:4]
    @test all(isapprox.(filtered_gpu[interior, interior, :, :],
        filtered_cpu[interior, interior, :, :]; atol = 1f-4, rtol = 0))
    @test all(isapprox.(filtered_gpu[border], filtered_cpu[border]; atol = 1f-4, rtol = 0))
    @test filtered_gpu[interior, interior, :, :] == original_gpu[interior, interior, :, :]
end
