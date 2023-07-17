"""
    squeeze(im)

Remove all singleton dimensions from an array.

# Arguments
- `im`: The input array

# Returns
- The input array with singleton dimensions removed 

# Examples
```julia
a = reshape(1:8, 2, 1, 2, 1, 2)
squeeze(a) # returns a 2x2x2 array
```
"""
function squeeze(im)
    while any(size(im) .== 1)
        dim, = findall(i -> size(im, i) == 1, 1:ndims(im))
        im = dropdims(im, dims=dim)
    end
    return im
end


"""
    showscaled(im; nz=3, zoom=4)

Display a scaled grayscale image, squeezing size-1 dims.

# Arguments
- `im`: The input image   
- `nz=3`: Max number of z slices to show
- `zoom=4`: Scale factor for imresize 

# Details
- Singleton dimensions are removed with `squeeze`
- If im is 3D, only first `nz` z slices are shown 
- Image is scaled to 0-1 based on min/max intensities
- Final image is resized by `zoom` factor & shown grayscale

# Examples
```julia
im = rand(100,100,20)
showscaled(im) # Shows first 3 z slices, zoomed 4x
```
"""
function showscaled(im; nz=3, zoom=4)
    im = squeeze(im)
    if ndims(im) == 3
        maxz = min(size(im, 3), nz)
        im = im[:, :, 1:maxz]
        im = reshape(im, size(im, 1), size(im, 2) * size(im, 3))
    end

    # Get the min and max
    im_min = minimum(im)
    im_max = maximum(im)

    # Scale the image
    im_scaled = (im .- im_min) ./ (im_max - im_min)

    im_scaled = imresize(im_scaled, ratio=zoom)
    # Show the image
    Gray.(im_scaled)
end

