using Revise
using SMLMBoxer
using Flux
using Images

include("visutil.jl")

# setup Convolution
sigma = 1.3f0
kernelsize = 7
kernel = SMLMBoxer.gaussian_2d(sigma, kernelsize)
showscaled(kernel)
weights = reshape(kernel, size(kernel)..., 1, 1)
bias = zeros(Float32, 1)
conv_layer = Conv(weights, bias, identity, pad=SamePad())
data = rand(256, 256, 1, 5000) .> 0.999
imagestack = conv_layer(Float32.(data))
showscaled(imagestack)



sigma_small, sigma_large, minval = 1, 2, 0.00

filtered_stack = SMLMBoxer.dog_filter(imagestack, SMLMBoxer.GetBoxesArgs(;sigma_small, sigma_large, minval, use_gpu=true))
showscaled(filtered_stack |> cpu)

break

localmaximage = SMLMBoxer.genlocalmaximage(filtered_stack, 7; minval)
showscaled(localmaximage |> cpu)

break

showscaled(data)

# Convert the local max image to coordinates
coords_true = SMLMBoxer.maxima2coords(data)
coords = SMLMBoxer.maxima2coords(localmaximage)
coords_filtered = SMLMBoxer.removeoverlap(coords, 2)
coords_found = SMLMBoxer.findlocalmax(imagestack, 1, 2, 0)

SMLMBoxer.getboxes(use_gpu=false)

boxes, coords = SMLMBoxer.getboxes(imagestack = imagestack[:,:,1,:], use_gpu=false )


showscaled(boxes, nz = 4, zoom = 16)


SMLMBoxer.getboxes(use_gpu=true)

