"""
   reshape_for_flux(arr::AbstractArray)

Reshape array to have singleton dims for Flux.jl convolution.  

# Arguments
- `arr`: Input array, must be 2D or 3D

# Returns
- Reshaped array with added singleton dimensions
"""
function reshape_for_flux(arr::AbstractArray)
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
- `ksize`: Kernel size

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
   difference_of_gaussians(s1, s2, ksize)

Compute difference of Gaussian kernels.   
 
# Arguments
- `s1`: Sigma for small Gaussian 
- `s2`: Sigma for large Gaussian
- `ksize`: Kernel size

# Returns
- `dog`: Difference of Gaussians kernel
"""
function difference_of_gaussians(sigma_small::Float32, sigma_large::Float32, kernelsize::Int)
    kernel_small = gaussian_2d(sigma_small, kernelsize)
    kernel_large = gaussian_2d(sigma_large, kernelsize)
    dog = kernel_small .- kernel_large
    return dog
end

"""
   genlocalmaximage(stack, args)

Apply DoG filter and find local maxima. 

# Arguments 
- `stack`: Input image stack
- `args`: Parameters

# Returns
- `localmax`: Image with local max values
"""
function genlocalmaximage(imagestack::AbstractArray{<:Real}, kwargs::GetBoxesArgs)

    sigma_small = kwargs.sigma_small
    sigma_large = kwargs.sigma_large
    minval = kwargs.minval
    use_gpu = kwargs.use_gpu

    minkernelsize = 5
    kernelsize = max(minkernelsize, Int(ceil(sigma_large * 2)))

    # Make the difference of gaussians
    dog = difference_of_gaussians(Float32(sigma_small), Float32(sigma_large), kernelsize)

    # Prepare the weights tensor to match the expected dimensions: height, width, input channels, output channels
    weights = reshape(dog, size(dog)..., 1, 1)

    # Create a bias tensor filled with zeros
    bias = zeros(Float32, 1)

    # Convolution layer with manually specified weights and bias, and identity activation function
    conv_layer = Conv(weights, bias, identity, pad=SamePad())
    maxpool_layer = MaxPool((kernelsize, kernelsize), pad=SamePad(), stride=(1, 1))

    if use_gpu
        conv_layer = conv_layer |> gpu
        maxpool_layer = maxpool_layer |> gpu
    end

    filtered_stack = conv_layer(imagestack)
    maximage = maxpool_layer(filtered_stack) .== filtered_stack
    localmaximage = maximage .& (filtered_stack .> minval)
    return Float32.(localmaximage .* filtered_stack |> cpu)
end