@kwdef mutable struct GetBoxesArgs
    imagestack = rand(Float32, 256, 256, 50) .> 0.999
    boxsize = 7
    overlap = 2.0
    sigma_small = 1.0
    sigma_large = 2.0
    minval = 0.0
    use_gpu = true
end