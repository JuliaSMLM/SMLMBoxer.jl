
function genlocalmaximage(imagestack::AbstractArray{<:Real}, kernelsize::Int; minval::Real = 0.0, use_gpu = false)


    maxpool_layer = MaxPool((kernelsize, kernelsize), pad=SamePad(), stride=(1, 1))

    if use_gpu
        maxpool_layer = maxpool_layer |> gpu
    end

    maximage = maxpool_layer(imagestack) .== imagestack
    localmaximage .= (maximage .& (imagestack .> minval)) .* imagestack

    return localmaximage
end


"""  
   findlocalmax(stack, args)

Apply DoG filter and extract local max coords. 

# Arguments
- `stack`: Input image stack
- `args`: Parameters

# Returns 
- `coords`: Local max coordinates
"""
function findlocalmax(imagestack::AbstractArray{<:Real}, kwargs::GetBoxesArgs)

    # Filter the image stack with a difference of Gaussians
    filtered_stack = dog_filter(imagestack, kwargs)

    # Find local maxima   
    localmaximage = genlocalmaximage(filtered_stack, kwargs)

    # Convert local maxima image to coordinates
    coords = maxima2coords(localmaximage, kwargs)
    return coords
end
