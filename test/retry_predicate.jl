using Test, SMLMBoxer, CUDA

@testset "Only recoverable GPU resource errors are retried" begin
    @test SMLMBoxer.is_gpu_resource_error(CUDA.OutOfGPUMemoryError())
    @test SMLMBoxer.is_gpu_resource_error(CUDA.CuError(CUDA.ERROR_OUT_OF_MEMORY))
    @test SMLMBoxer.is_gpu_resource_error(CUDA.CuError(CUDA.ERROR_DEVICE_UNAVAILABLE))
    @test !SMLMBoxer.is_gpu_resource_error(CUDA.CuError(CUDA.ERROR_NO_BINARY_FOR_GPU))
    @test !SMLMBoxer.is_gpu_resource_error(CUDA.CuError(CUDA.ERROR_INVALID_IMAGE))
    @test !SMLMBoxer.is_gpu_resource_error(AssertionError("x"))
    @test !SMLMBoxer.is_gpu_resource_error(ErrorException("x"))
end
