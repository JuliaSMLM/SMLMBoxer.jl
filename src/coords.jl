"""
   maxima2coords(imagestack)

Get coordinates of all non-zero pixels in input stack

# Arguments
- `imagestack`: Input image stack

# Returns
- `coords`: List of coords for each frame (always Float32)
"""
function maxima2coords(imagestack::AbstractArray{T}) where T<:Real

    nframes = size(imagestack, 4)
    coords = Vector{Matrix{Float32}}(undef, nframes)

    # Count the number of non-zero elements in each frame
    nboxes = sum(!iszero, imagestack, dims=(1, 2))

    for f in 1:nframes
        coords[f] = zeros(Float32, nboxes[f], 4)
        idx_coords = 1
        for j in axes(imagestack, 2)
            for i in axes(imagestack, 1)
                if imagestack[i, j, 1, f] != 0
                    # Fill in the output array
                    coords[f][idx_coords, 1] = Float32(i)
                    coords[f][idx_coords, 2] = Float32(j)
                    coords[f][idx_coords, 3] = Float32(f)
                    coords[f][idx_coords, 4] = imagestack[i, j, 1, f]
                    idx_coords += 1
                end
            end
        end
    end
    return coords
end

"""
   removeoverlap(coords, args)
 
Remove overlapping coords based on distance.

# Arguments
- `coords`: List of coords
- `args`: Parameters  

# Returns
- `coords`: Coords with overlaps removed 
"""
function removeoverlap(coords::Vector{Matrix{Float32}}, kwargs::GetBoxesArgs)
    overlap = kwargs.overlap
    for f in 1:size(coords, 2)
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
    return vcat(coords...)[:, 1:3]
end