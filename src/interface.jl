@kwdef mutable struct GetBoxesArgs
    imagestack = rand(Float32, 256, 256, 50) .> 0.999
    boxsize = 7
    overlap = 2.0
    sigma_small = 1.0
    sigma_large = 2.0
    minval = 0.0
    use_gpu = true
end

function getboxes(; kwargs...)

    kwargs = GetBoxesArgs(; kwargs...)

    imagestack = reshape_for_flux(kwargs.imagestack)

    # Determine whether to perform calculations on the GPU
    kwargs.use_gpu = kwargs.use_gpu && has_cuda() && CUDA.functional()

    # Get local max coords- this is only step done on GPU
    if kwargs.use_gpu
        coords = findlocalmax(CuArray(imagestack), kwargs)
    else
        coords = findlocalmax(imagestack, kwargs)
    end

    coords = removeoverlap(coords, kwargs)
    boxstack = getboxstack(imagestack, coords, kwargs)

    # Return the box stack and the coords  
    return boxstack, coords
end

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

function getboxstack(imagestack, coords, kwargs::GetBoxesArgs)

    nboxes = size(coords, 1)
    boxstack = zeros(eltype(imagestack), kwargs.boxsize, kwargs.boxsize, nboxes)

    # Cut out the boxes
    for i in 1:nboxes
        row = Int(coords[i, 1])
        col = Int(coords[i, 2])
        im = Int(coords[i, 3])
        box = @view boxstack[:, :, i]
        fillbox!(box, imagestack, row, col, im, kwargs.boxsize)
    end

    return boxstack
end

function fillbox!(box::AbstractArray{<:Real,2}, imagestack::AbstractArray{<:Real,4}, row::Int, col::Int, im::Int, boxsize::Int)
    # Get the size of the image stack
    (nrows, ncols, ~, nimages) = size(imagestack)

    # Define the boundaries for the box
    row_min = row - boxsize ÷ 2
    row_max = row + boxsize ÷ 2
    col_min = col - boxsize ÷ 2
    col_max = col + boxsize ÷ 2

    # Adjust the boundaries if they are out of the imagestack boundaries
    if row_min < 1
        row_min = 1
        row_max = boxsize
    end

    if row_max > nrows
        row_min = nrows - boxsize + 1
        row_max = nrows
    end

    if col_min < 1
        col_min = 1
        col_max = boxsize
    end

    if col_max > ncols
        col_min = ncols - boxsize + 1
        col_max = ncols
    end

    # Copy the portion of the image stack to the box
    box .= imagestack[row_min:row_max, col_min:col_max, 1, im]

    return nothing
end

function gaussian_2d(sigma::Float32, kernelsize::Int)
    kernel = zeros(Float32, kernelsize, kernelsize)
    center = kernelsize % 2 == 0 ? kernelsize ÷ 2 + 0.5 : (kernelsize + 1) ÷ 2
    # center = kernelsize ÷ 2 + 1

    for i in 1:kernelsize
        for j in 1:kernelsize
            x = i - center
            y = j - center
            kernel[i, j] = exp(-(x^2 + y^2) / (2 * sigma^2))
        end
    end
    return kernel ./ sum(kernel)
end

function difference_of_gaussians(sigma_small::Float32, sigma_large::Float32, kernelsize::Int)
    kernel_small = gaussian_2d(sigma_small, kernelsize)
    kernel_large = gaussian_2d(sigma_large, kernelsize)
    dog = kernel_small .- kernel_large
    return dog
end

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

    return localmaximage .* filtered_stack
end

function maxima2coords(localmaximage::AbstractArray{<:Real}, kwargs::GetBoxesArgs)

    # Determine whether to perform calculations on the GPU
    use_gpu = kwargs.use_gpu

    # Number of observations
    nobs = size(localmaximage, 4)

    # Generate coordinate grids
    if use_gpu
        ones_matrix = CUDA.ones(Float32, size(localmaximage, 1), size(localmaximage, 2))
    else
        ones_matrix = ones(Float32, size(localmaximage, 1), size(localmaximage, 2))
    end

    xx = cumsum(ones_matrix, dims=2)
    yy = cumsum(ones_matrix, dims=1)

    # Extract the coordinates for each observation
    coords = [hcat(
        view(yy, :, :)[localmaximage[:, :, 1, f].>0],
        view(xx, :, :)[localmaximage[:, :, 1, f].>0],
        fill(f, sum(localmaximage[:, :, 1, f] .> 0)),
        view(localmaximage[:, :, 1, f], localmaximage[:, :, 1, f] .> 0)
    )
              for f in 1:nobs]

    return coords |> cpu
end


function findlocalmax(imagestack::AbstractArray{<:Real}, kwargs::GetBoxesArgs)

    # Generate the local max image
    localmaximage = genlocalmaximage(imagestack, kwargs)

    # Convert the local max image to coordinates
    coords = maxima2coords(localmaximage, kwargs)

    return coords
end

function removeoverlap(coords::Vector{Matrix{Float32}}, kwargs::GetBoxesArgs)
    overlap = kwargs.overlap
    nobs = length(coords)
    for f in 1:nobs
        ncoords = size(coords[f], 1)
        keep = trues(ncoords)

        for i in 1:ncoords
            if keep[i]
                ci = coords[f][i, :]
                for j in (i+1):ncoords
                    if keep[j]
                        cj = coords[f][j, :]
                        dist = sqrt(sum((ci[1:2] - cj[1:2]) .^ 2)) # Only compare the x and y coordinates
                        if dist <= overlap
                            if ci[4] < cj[4] # Use the 4th column of ci and cj for intensity comparison
                                keep[i] = false
                                break
                            else
                                keep[j] = false
                            end
                        end
                    end
                end
            end
        end

        coords[f] = coords[f][keep, :]
    end

    return vcat(coords...)
end
