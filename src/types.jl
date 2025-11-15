@kwdef mutable struct GetBoxesArgs
    imagestack = rand(Float32, 256, 256, 50) .> 0.999
    camera::Union{AbstractCamera,Nothing} = nothing
    boxsize = 7
    overlap = 2.0
    sigma_small = 1.0
    sigma_large = 2.0
    minval = 0.0
    use_gpu = true
end

"""
    pixels_to_microns(pixel_coords, camera::AbstractCamera)

Convert pixel coordinates (row, col) to micron coordinates (x, y) using camera geometry.

# Arguments
- `pixel_coords`: N×2 matrix of (row, col) coordinates
- `camera`: Camera object with pixel_edges_x and pixel_edges_y

# Returns
- N×2 matrix of (x, y) coordinates in microns
"""
function pixels_to_microns(pixel_coords::AbstractMatrix, camera::AbstractCamera)
    ncoords = size(pixel_coords, 1)
    coords_microns = similar(pixel_coords)

    for i in 1:ncoords
        row, col = pixel_coords[i, 1], pixel_coords[i, 2]
        # Convert to 1-based pixel centers
        # pixel_edges are the edges, so center of pixel i is at (edges[i] + edges[i+1])/2
        x = (camera.pixel_edges_x[Int(col)] + camera.pixel_edges_x[Int(col)+1]) / 2
        y = (camera.pixel_edges_y[Int(row)] + camera.pixel_edges_y[Int(row)+1]) / 2
        coords_microns[i, 1] = x
        coords_microns[i, 2] = y
    end

    return coords_microns
end

"""
    extract_camera_roi(camera::AbstractCamera, row_range, col_range)

Extract a camera ROI with calibration data for the specified pixel region.

# Arguments
- `camera`: Source camera object
- `row_range`: Range of rows to extract
- `col_range`: Range of columns to extract

# Returns
- Camera object of the same type with ROI calibration data
"""
function extract_camera_roi(camera::IdealCamera{T}, row_range, col_range) where T
    return IdealCamera{T}(
        pixel_edges_x = camera.pixel_edges_x[col_range],
        pixel_edges_y = camera.pixel_edges_y[row_range]
    )
end

function extract_camera_roi(camera::SCMOSCamera{T}, row_range, col_range) where T
    # Handle both scalar and per-pixel calibration parameters
    offset = camera.offset isa AbstractArray ? camera.offset[row_range[1:end-1], col_range[1:end-1]] : camera.offset
    gain = camera.gain isa AbstractArray ? camera.gain[row_range[1:end-1], col_range[1:end-1]] : camera.gain
    readnoise = camera.readnoise isa AbstractArray ? camera.readnoise[row_range[1:end-1], col_range[1:end-1]] : camera.readnoise
    qe = camera.qe isa AbstractArray ? camera.qe[row_range[1:end-1], col_range[1:end-1]] : camera.qe

    return SCMOSCamera{T}(
        pixel_edges_x = camera.pixel_edges_x[col_range],
        pixel_edges_y = camera.pixel_edges_y[row_range],
        offset = offset,
        gain = gain,
        readnoise = readnoise,
        qe = qe
    )
end

"""
    get_variance_map(camera::AbstractCamera, imagesize)

Compute variance map from camera calibration.

# Arguments
- `camera`: Camera object with noise calibration
- `imagesize`: Tuple of (nrows, ncols) for the image

# Returns
- Variance map (variance = readnoise²) matching image dimensions
"""
function get_variance_map(camera::IdealCamera{T}, imagesize::Tuple{Int,Int}) where T
    # IdealCamera has no readnoise, return uniform variance of 1.0
    return ones(T, imagesize)
end

function get_variance_map(camera::SCMOSCamera{T}, imagesize::Tuple{Int,Int}) where T
    nrows, ncols = imagesize

    if camera.readnoise isa AbstractArray
        # Per-pixel readnoise map: variance = readnoise²
        variance_map = camera.readnoise .^ 2
        # Ensure it matches the image size
        @assert size(variance_map) == imagesize "Readnoise map size $(size(variance_map)) doesn't match image size $imagesize"
        return variance_map
    else
        # Scalar readnoise: uniform variance
        return fill(T(camera.readnoise^2), imagesize)
    end
end