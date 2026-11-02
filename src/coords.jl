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

    flat = mask = nz_indices = nz_values = nothing
    local nz_indices_cpu, nz_values_cpu
    completed = false
    try
        # The reshape owns a reference to storage, not the caller's array object.
        flat = reshape(localmaximage, :)
        mask = flat .!= zero(T)
        nz_indices = findall(mask)
        CUDA.unsafe_free!(mask)
        mask = nothing

        if isempty(nz_indices)
            completed = true
            return [zeros(Float32, 0, 4) for _ in 1:nframes]
        end

        nz_values = flat[nz_indices]
        # Transfer only sparse data to CPU (~1.2 MB vs ~1 GB).
        nz_indices_cpu = Array(nz_indices)
        nz_values_cpu = Array(nz_values)
        completed = true
    finally
        free_gpu_arrays(mask, nz_indices, nz_values,
            flat === localmaximage ? nothing : flat; failed=!completed)
    end

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

Remove maxima sharing a max-pool window, using Chebyshev distance < cld(k, 2),
where k = max(3, floor(Int, boxsize - overlap)). Maxima from the filter that share a
window have equal intensities, so the earlier detection is kept. For other input, the
dimmer of two maxima is dropped and ties drop the later one. Keep intensities for
clamped-corner selection.

# Arguments
- `coords`: List of coords
- `args`: Parameters

# Returns
- `coords`: Matrix of (row, col, frame, filtered intensity) with overlaps removed
"""
function removeoverlap(coords::Vector{Matrix{Float32}}, kwargs::GetBoxesArgs)
    k = max(3, floor(Int, kwargs.boxsize - kwargs.overlap))
    separation = cld(k, 2)
    for f in eachindex(coords)
        c = coords[f]
        ncoords = size(c, 1)
        keep = trues(ncoords)

        for i in 1:ncoords
            if keep[i]
                xi = c[i, 1]
                yi = c[i, 2]
                intensity_i = c[i, 4]
                for j in (i+1):ncoords
                    if keep[j]
                        dx = xi - c[j, 1]
                        dy = yi - c[j, 2]
                        if max(abs(dx), abs(dy)) < separation
                            if intensity_i < c[j, 4]
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
