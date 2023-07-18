"""
   reshape_for_flux(arr::AbstractArray)

Reshape array to have singleton dims for Flux.jl convolution.  

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

# Arguments
- `imagestack`: Input array of image data 
- `args`: Arguments with sigma values  

# Returns
- `filtered_stack`: Filtered image stack 
"""
function dog_filter(imagestack::AbstractArray{<:Real}, args::GetBoxesArgs)
    
    sigma_small = args.sigma_small
    sigma_large = args.sigma_large
    dog = dog_kernel(Float32(sigma_small), Float32(sigma_large))
    filtered_stack = convolve(imagestack, dog, use_gpu=args.use_gpu)

    return filtered_stack
end


"""
    convolve(imagestack, kernel; use_gpu=false)

Convolve imagestack with given kernel.

# Arguments
- `imagestack`: Input array of image data
- `kernel`: Kernel to convolve with 

# Keyword Arguments
- `use_gpu`: Whether to use GPU 

# Returns
- `filtered_stack`: Convolved image stack
""" 
function convolve(imagestack::AbstractArray{<:Real}, kernel::Matrix{Float32}; use_gpu = false)

    # Prepare the weights tensor to match the expected dimensions: height, width, input channels, output channels
    weights = reshape(kernel, size(kernel)..., 1, 1)

    # Create a bias tensor filled with zeros
    bias = zeros(Float32, 1)

    # Convolution layer with manually specified weights and bias, and identity activation function
    conv_layer = Conv(weights, bias, identity, pad=SamePad())
    
    if use_gpu
        imagestack = imagestack |> gpu
        conv_layer = conv_layer |> gpu
    end

    filtered_stack = conv_layer(imagestack)
    return filtered_stack
end



