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

    # Max pooling: KA kernel (GPU) vs NNlib (CPU), odd (5) and even (4) windows,
    # on the same CPU-filtered input.
    y = conv_cpu
    y_gpu = CuArray(y)
    for k in (5, 4)
        max_cpu = SMLMBoxer.genlocalmaximage(y, k; minval = 0.0, use_gpu = false)
        max_gpu = Array(SMLMBoxer.genlocalmaximage(y, k; minval = 0.0, use_gpu = true))
        @test max_cpu == max_gpu

        # Also compare the raw (unmasked) max window, not only the thresholded image
        plo = (k - 1) ÷ 2
        pad = isodd(k) ? (k ÷ 2, k ÷ 2, k ÷ 2, k ÷ 2) : (plo, k ÷ 2, plo, k ÷ 2)
        raw_cpu = NNlib.maxpool(y, (k, k); pad = pad, stride = 1)
        raw_gpu = CUDA.zeros(Float32, size(y))
        kernel! = SMLMBoxer.localmax_kernel!(CUDABackend())
        kernel!(raw_gpu, y_gpu, plo, k; ndrange = size(y)[[1, 2, 4]])
        KernelAbstractions.synchronize(CUDABackend())
        @test raw_cpu == Array(raw_gpu)
    end
end
