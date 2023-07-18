using Revise
using SMLMBoxer
using Flux
using Images
using CUDA

include("visutil.jl")

# Create Dataset
sigma = 1.3f0
kernelsize = 7
kernel = SMLMBoxer.gaussian_2d(sigma, kernelsize)
weights = reshape(kernel, size(kernel)..., 1, 1)
bias = zeros(Float32, 1)
conv_layer = Conv(weights, bias, identity, pad=SamePad())
data = rand(Float32, 256, 512, 1, 2000) .> 0.999
imagestack = conv_layer(Float32.(data));


#This is the main function call
@time boxes, boxcoords, maxcoords = SMLMBoxer.getboxes(imagestack=imagestack[:, :, 1, :], use_gpu=true);
@time boxes, boxcoords, maxcoords = SMLMBoxer.getboxes(imagestack=imagestack[:, :, 1, :], use_gpu=false);


display(showscaled(imagestack))
display(showscaled(boxes; nz = 15))

