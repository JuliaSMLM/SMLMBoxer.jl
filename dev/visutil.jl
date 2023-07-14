
function squeeze(im)
    while any(size(im) .== 1)
        dim, = findall(i -> size(im, i) == 1, 1:ndims(im))
        im = dropdims(im, dims=dim)
    end
    return im
end

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

