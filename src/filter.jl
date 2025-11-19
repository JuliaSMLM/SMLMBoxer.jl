"""
   reshape_for_flux(arr::AbstractArray)

Reshape array to have singleton dims for NNlib convolution.  

# Arguments
- `arr`: Input array, must be 2D or 3D

# Returns
- Reshaped array with added singleton dimensions
"""
function reshape_for_flux(arr::AbstractArray{<:Real})
    dims = ndims(arr)
    if dims == 2
        # If the array is 2D, we'll add two singleton dimensions at the end
        return reshape(arr, size(arr)..., 1, 1)
    elseif dims == 3
        # If the array is 3D, we'll add a singleton dimension in the third position
        s = size(arr)
        return reshape(arr, s[1], s[2], 1, s[3])
    else
        error("Input array must be 2D or 3D")
    end
end

"""
   gaussian_2d(sigma, ksize)
 
Create a 2D Gaussian kernel.

# Arguments  
- `sigma`: Standard deviation
- `kernelsize`: Kernel size

# Returns
- `kernel`: Normalized 2D Gaussian kernel  
""" 
function gaussian_2d(sigma::Float32, kernelsize::Int)
    kernel = zeros(Float32, kernelsize, kernelsize)
    center = kernelsize % 2 == 0 ? kernelsize ÷ 2 + 0.5 : (kernelsize + 1) ÷ 2

    for i in 1:kernelsize
        for j in 1:kernelsize
            x = i - center
            y = j - center
            kernel[i, j] = exp(-(x^2 + y^2) / (2 * sigma^2))
        end
    end
    return kernel ./ sum(kernel)
end

"""   
dog_kernel(s1, s2)

Compute difference of Gaussian kernels.   
 
# Arguments
- `sigma_small`: Sigma for small Gaussian 
- `sigma_large`: Sigma for large Gaussian

# Returns
- `dog`: Difference of Gaussians kernel
"""
function dog_kernel(sigma_small::Float32, sigma_large::Float32)
    minkernelsize = 3
    kernelsize = max(minkernelsize, Int(ceil(sigma_large * 4)))
    # Round up to nearest odd number for symmetry when using SamePad 
    kernelsize = isodd(ceil(kernelsize)) ? ceil(kernelsize) : ceil(kernelsize) + 1
    kernel_small = gaussian_2d(sigma_small, kernelsize)
    kernel_large = gaussian_2d(sigma_large, kernelsize)
    dog = kernel_small .- kernel_large
    return dog
end

"""
    dog_filter(imagestack, args)

Apply DoG filter to imagestack based on args.
Uses variance-weighted filtering if sCMOS camera is provided.

# Arguments
- `imagestack`: Input array of image data
- `args`: Arguments with sigma values and camera

# Returns
- `filtered_stack`: Filtered image stack
"""
function dog_filter(imagestack::AbstractArray{<:Real}, args::GetBoxesArgs)

    sigma_small = args.sigma_small
    sigma_large = args.sigma_large

    # Check if we have an sCMOS camera with variance information
    if args.camera isa SCMOSCamera
        # Use variance-weighted DoG filtering
        filtered_stack = dog_filter_variance_weighted(imagestack, sigma_small, sigma_large, args)
    else
        # Standard DoG filtering (no variance weighting)
        dog = dog_kernel(Float32(sigma_small), Float32(sigma_large))
        filtered_stack = convolve(imagestack, dog, use_gpu=args.use_gpu)
    end

    # Ensure result is on CPU (safety check)
    filtered_stack = filtered_stack isa CuArray ? Array(filtered_stack) : filtered_stack

    return filtered_stack
end

"""
    dog_filter_variance_weighted(imagestack, sigma_small, sigma_large, args)

Apply variance-weighted DoG filter using sCMOS variance map.
Implements SMITE-style inverse variance weighting during convolution.

# Arguments
- `imagestack`: Input image data (rows, cols, 1, frames)
- `sigma_small`: Sigma for small Gaussian
- `sigma_large`: Sigma for large Gaussian
- `args`: GetBoxesArgs with camera

# Returns
- `filtered_stack`: Variance-weighted filtered image
"""
function dog_filter_variance_weighted(imagestack::AbstractArray{<:Real},
                                       sigma_small::Real,
                                       sigma_large::Real,
                                       args::GetBoxesArgs)
    # Get image dimensions (rows, cols, 1, frames)
    nrows, ncols, _, nframes = size(imagestack)

    # Get variance map from camera (variance = readnoise²)
    variance_map = get_variance_map(args.camera, (nrows, ncols))

    # Apply small Gaussian with variance weighting
    filtered_small = convolve_variance_weighted(imagestack, variance_map,
                                                Float32(sigma_small), args.use_gpu)

    # Apply large Gaussian with variance weighting
    filtered_large = convolve_variance_weighted(imagestack, variance_map,
                                                Float32(sigma_large), args.use_gpu)

    # Difference of Gaussians
    filtered_stack = filtered_small .- filtered_large

    return filtered_stack
end

"""
    variance_weighted_gaussian_kernel!(output, input, variance, sigma, winsize)

KernelAbstractions kernel for variance-weighted Gaussian convolution.
Implements SMITE-style inverse variance weighting.

This follows the same KernelAbstractions pattern used in GaussMLE.jl (kernel-abstract branch)
for seamless CPU/GPU execution and consistent API across JuliaSMLM packages.

# Arguments
- `output`: Output array (nrows, ncols)
- `input`: Input array (nrows, ncols)
- `variance`: Variance map (nrows, ncols)
- `sigma`: Gaussian sigma
- `winsize`: Window size (pixels)

# Note
Same kernel code runs on CPU (via CPU() backend) or GPU (via CUDABackend()).
Backend is selected automatically based on use_gpu parameter.
"""
@kernel function variance_weighted_gaussian_kernel!(output, input, variance, sigma, winsize)
    i, j = @index(Global, NTuple)
    nrows, ncols = @ndrange()

    # Window bounds
    row_start = max(1, i - winsize)
    row_end = min(nrows, i + winsize)
    col_start = max(1, j - winsize)
    col_end = min(ncols, j + winsize)

    # Accumulate variance-weighted sum
    weightsum = zero(eltype(input))
    varsum = zero(eltype(variance))

    for ii in row_start:row_end
        for jj in col_start:col_end
            # Gaussian weight
            dist_sq = Float32((ii-i)^2 + (jj-j)^2)
            gauss_weight = exp(-dist_sq / (2 * sigma^2))

            # Inverse variance weight
            inv_var_weight = gauss_weight / variance[ii, jj]

            # Accumulate
            varsum += inv_var_weight
            weightsum += inv_var_weight * input[ii, jj]
        end
    end

    # Normalized result
    output[i, j] = weightsum / varsum
end

"""
    convolve_variance_weighted(imagestack, variance_map, sigma, use_gpu)

Apply variance-weighted Gaussian convolution using KernelAbstractions.
Device-agnostic: works on CPU and GPU with same code.

# Arguments
- `imagestack`: Input image (rows, cols, 1, frames)
- `variance_map`: Variance at each pixel (rows, cols)
- `sigma`: Gaussian sigma
- `use_gpu`: Use GPU if available

# Returns
- Variance-weighted filtered image
"""
function convolve_variance_weighted(imagestack::AbstractArray{T},
                                    variance_map::AbstractMatrix{T},
                                    sigma::Float32,
                                    use_gpu::Bool) where T<:Real
    nrows, ncols, _, nframes = size(imagestack)

    # Create output array
    filtered = similar(imagestack)

    # Gaussian kernel window size
    winsize = Int(ceil(3 * sigma))

    # Select backend: GPU or CPU
    if use_gpu && CUDA.functional()
        backend = CUDABackend()
        # Transfer data to GPU
        imagestack_dev = CuArray(imagestack)
        variance_dev = CuArray(variance_map)
        filtered_dev = CuArray(filtered)
    else
        backend = CPU()
        imagestack_dev = imagestack
        variance_dev = variance_map
        filtered_dev = filtered
    end

    # Launch kernel for each frame
    kernel! = variance_weighted_gaussian_kernel!(backend)

    for frame in 1:nframes
        # Get views for this frame
        input_frame = @view imagestack_dev[:, :, 1, frame]
        output_frame = @view filtered_dev[:, :, 1, frame]

        # Launch kernel
        kernel!(output_frame, input_frame, variance_dev, sigma, winsize, ndrange=(nrows, ncols))
    end

    # Wait for kernel completion
    KernelAbstractions.synchronize(backend)

    # Transfer back from GPU if needed
    if use_gpu && CUDA.functional()
        filtered = Array(filtered_dev)
    end

    return filtered
end


"""
    convolve(imagestack, kernel; use_gpu=false)

Convolve imagestack with given kernel using NNlib.

# Arguments
- `imagestack`: Input array of image data (H, W, 1, F)
- `kernel`: Kernel to convolve with (K, K)

# Keyword Arguments
- `use_gpu`: Whether to use GPU

# Returns
- `filtered_stack`: Convolved image stack
"""
function convolve(imagestack::AbstractArray{<:Real}, kernel::Matrix{Float32}; use_gpu = false)
    # Reshape kernel to NNlib format: (height, width, input_channels, output_channels)
    weights = reshape(kernel, size(kernel)..., 1, 1)

    # NNlib padding: (pad_left, pad_right, pad_top, pad_bottom)
    # For symmetric "same" padding
    p = size(kernel, 1) ÷ 2
    pad = (p, p, p, p)

    if use_gpu && CUDA.functional()
        # Transfer to GPU
        imagestack_gpu = CuArray(imagestack)
        weights_gpu = CuArray(weights)

        # NNlib.conv uses cuDNN on GPU
        filtered_gpu = NNlib.conv(imagestack_gpu, weights_gpu; pad=pad, stride=1)

        # Transfer back to CPU
        filtered_stack = Array(filtered_gpu)
    else
        # NNlib.conv CPU implementation
        filtered_stack = NNlib.conv(imagestack, weights; pad=pad, stride=1)
    end

    return filtered_stack
end



