"""
    localmax_kernel!(out, img, plo, k)

KernelAbstractions kernel for max pooling, batched over frames via
ndrange=(nrows, ncols, nframes). Takes the max over the in-bounds part of a k×k window
starting at offset `-plo`, reproducing NNlib.maxpool's odd (symmetric) and even
(asymmetric) padding.

# Arguments
- `out`: Output array (nrows, ncols, 1, nframes)
- `img`: Input array (nrows, ncols, 1, nframes)
- `plo`: Window offset, `(k - 1) ÷ 2`
- `k`: Window size
"""
@kernel function localmax_kernel!(out, @Const(img), plo, k)
    i, j, f = @index(Global, NTuple)
    nrows, ncols = size(img, 1), size(img, 2)
    m = typemin(eltype(out))
    for b in 1:k, a in 1:k
        ii = i + a - 1 - plo
        jj = j + b - 1 - plo
        if 1 <= ii <= nrows && 1 <= jj <= ncols
            @inbounds m = max(m, img[ii, jj, 1, f])
        end
    end
    @inbounds out[i, j, 1, f] = m
end

"""
# genlocalmaximage(imagestack, kernelsize; minval=0.0, use_gpu=false)

Generate an image highlighting the local maxima using NNlib max pooling on the CPU, or a
KernelAbstractions kernel on the GPU.

# Arguments
- `imagestack`: An array of real numbers representing the image data (H, W, 1, F).
- `kernelsize`: The size of the kernel used to identify local maxima.

# Keyword Arguments
- `minval`: The minimum value a local maximum must have to be considered valid (default: 0.0).
- `use_gpu`: Whether or not to use GPU acceleration (default: false).

# Returns
- `localmaximage`: An image with local maxima highlighted.
"""
function genlocalmaximage(imagestack::AbstractArray{<:Real}, kernelsize::Int; minval::Real=0.0, use_gpu=false)
    poolsize = (kernelsize, kernelsize)
    # NNlib padding: (pad_left, pad_right, pad_top, pad_bottom)
    # For "same" output size, need asymmetric padding for even kernels
    if isodd(kernelsize)
        # Odd kernel: symmetric padding
        p = kernelsize ÷ 2
        pad = (p, p, p, p)
    else
        # Even kernel: asymmetric padding (more on right/bottom)
        p_low = (kernelsize - 1) ÷ 2
        p_high = kernelsize ÷ 2
        pad = (p_low, p_high, p_low, p_high)
    end

    if use_gpu && CUDA.functional()
        # Transfer to GPU if not already there
        imagestack_gpu = imagestack isa CuArray ? imagestack : CuArray(imagestack)
        nrows, ncols, _, nframes = size(imagestack_gpu)
        plo = (kernelsize - 1) ÷ 2  # window offset; same for odd and even kernelsize

        # KernelAbstractions max pooling - KEEP RESULT ON GPU
        maxpooled = CUDA.zeros(eltype(imagestack_gpu), nrows, ncols, 1, nframes)
        backend = CUDABackend()
        kernel! = localmax_kernel!(backend)
        kernel!(maxpooled, imagestack_gpu, plo, kernelsize, ndrange=(nrows, ncols, nframes))
        KernelAbstractions.synchronize(backend)
        maximage = (maxpooled .== imagestack_gpu)
        localmaximage = (maximage .& (imagestack_gpu .> minval)) .* imagestack_gpu

        return localmaximage  # Returns CuArray - keep on GPU!
    else
        # NNlib.maxpool CPU implementation
        maxpooled = NNlib.maxpool(imagestack, poolsize; pad=pad, stride=1)
        maximage = (maxpooled .== imagestack)
        localmaximage = (maximage .& (imagestack .> minval)) .* imagestack
        return localmaximage
    end
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
    if localmaximage isa CuArray
        # GPU sparse extraction: transfers ~1 MB instead of full array (~1 GB)
        coords = _gpu_maxima2coords(localmaximage)
    else
        coords = maxima2coords(localmaximage)
    end
    return coords
end

