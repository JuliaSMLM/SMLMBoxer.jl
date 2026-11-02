using Test, SMLMBoxer, CUDA

@testset "NVML devices match CUDA ordinals" begin
    if CUDA.NVML.has_nvml()
        for i in 0:(length(CUDA.devices()) - 1)
            @test CUDA.NVML.uuid(SMLMBoxer.nvml_device(i)) == CUDA.uuid(CUDA.CuDevice(i))
        end
    else
        @test_skip CUDA.NVML.has_nvml()
    end
end
