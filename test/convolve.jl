using Test, SMLMBoxer

@testset "Convolution requires an odd square kernel" begin
    image = zeros(Float32, 8, 8, 1, 1)
    for use_gpu in (false, true)
        @test_throws ArgumentError SMLMBoxer.convolve(image, ones(Float32, 2, 2); use_gpu)
        @test_throws ArgumentError SMLMBoxer.convolve(image, ones(Float32, 3, 5); use_gpu)
    end
end
