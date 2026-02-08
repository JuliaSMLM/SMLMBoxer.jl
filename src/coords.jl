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
    _gpu_maxima2coords(localmaximage::CuArray)

GPU-accelerated coordinate extraction using sparse compaction.

Instead of transferring the entire 4D array to CPU (e.g., 1 GB for 512x512x1x1000),
uses GPU `findall` (prefix-sum compaction) to find nonzero indices on-device, then
transfers only the sparse indices and values (~1.2 MB for ~100K maxima).

# Arguments
- `localmaximage`: 4D CuArray (nrows, ncols, 1, nframes) with nonzero values at maxima

# Returns
- `coords`: Vector{Matrix{Float32}} — same format as `maxima2coords`
"""
function _gpu_maxima2coords(localmaximage::CuArray{T}) where T<:Real
    nrows, ncols, _, nframes = size(localmaximage)

    # Flatten to 1D for findall — GPU prefix-sum compaction
    flat = reshape(localmaximage, :)
    mask = flat .!= zero(T)
    nz_indices = findall(mask)  # CuArray{Int64,1} — stays on GPU

    # Early return if no maxima found
    if isempty(nz_indices)
        return [zeros(Float32, 0, 4) for _ in 1:nframes]
    end

    # Gather values at nonzero indices (GPU gather)
    nz_values = flat[nz_indices]

    # Transfer only sparse data to CPU (~1.2 MB vs ~1 GB)
    nz_indices_cpu = Array(nz_indices)
    nz_values_cpu = Array(nz_values)

    # Convert linear indices to (row, col, 1, frame) subscripts
    ci = CartesianIndices((nrows, ncols, 1, nframes))

    # Build per-frame coordinate matrices
    coords = Vector{Matrix{Float32}}(undef, nframes)

    # Count per frame first
    frame_counts = zeros(Int, nframes)
    for idx in nz_indices_cpu
        _, _, _, f = Tuple(ci[idx])
        frame_counts[f] += 1
    end

    # Allocate and fill
    frame_pos = ones(Int, nframes)
    for f in 1:nframes
        coords[f] = zeros(Float32, frame_counts[f], 4)
    end

    for (k, idx) in enumerate(nz_indices_cpu)
        i, j, _, f = Tuple(ci[idx])
        pos = frame_pos[f]
        coords[f][pos, 1] = Float32(i)
        coords[f][pos, 2] = Float32(j)
        coords[f][pos, 3] = Float32(f)
        coords[f][pos, 4] = Float32(nz_values_cpu[k])
        frame_pos[f] = pos + 1
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