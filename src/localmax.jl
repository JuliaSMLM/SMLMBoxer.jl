"""
# genlocalmaximage(imagestack, kernelsize; minval=0.0, use_gpu=false)

Generate an image highlighting the local maxima.

# Arguments
- `imagestack`: An array of real numbers representing the image data.
- `kernelsize`: The size of the kernel used to identify local maxima.

# Keyword Arguments
- `minval`: The minimum value a local maximum must have to be considered valid (default: 0.0).
- `use_gpu`: Whether or not to use GPU acceleration (default: false).

# Returns
- `localmaximage`: An image with local maxima highlighted.
"""
function genlocalmaximage(imagestack::AbstractArray{<:Real}, kernelsize::Int; minval::Real=0.0, use_gpu=false)
    maxpool_layer = MaxPool((kernelsize, kernelsize), pad=SamePad(), stride=(1, 1))
    if use_gpu
        maxpool_layer = maxpool_layer |> gpu
    end
    maximage = maxpool_layer(imagestack) .== imagestack
    localmaximage = (maximage .& (imagestack .> minval)) .* imagestack
    return localmaximage
end

"""
# findlocalmax(imagestack, kernelsize; minval=0.0, use_gpu=false)

Find the coordinates of local maxima in an image.

# Arguments
- `imagestack`: An array of real numbers representing the image data.
- `kernelsize`: The size of the kernel used to identify local maxima.

# Keyword Arguments
- `minval`: The minimum value a local maximum must have to be considered valid (default: 0.0).
- `use_gpu`: Whether or not to use GPU acceleration (default: false).

# Returns
- `coords`: The coordinates of the local maxima in the image.
"""
function findlocalmax(imagestack::AbstractArray{<:Real}, kernelsize::Int; minval::Real=0.0f0, use_gpu=false)
    localmaximage = genlocalmaximage(imagestack, kernelsize; minval, use_gpu)
    coords = maxima2coords(localmaximage |> cpu)
    return coords
end

