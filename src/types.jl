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
    return IdealCamera(
        camera.pixel_edges_x[col_range],  # pixel_edges_x (positional)
        camera.pixel_edges_y[row_range]   # pixel_edges_y (positional)
    )
end

function extract_camera_roi(camera::SCMOSCamera{T}, row_range, col_range) where T
    # Handle both scalar and per-pixel calibration parameters
    offset = camera.offset isa AbstractArray ? camera.offset[row_range[1:end-1], col_range[1:end-1]] : camera.offset
    gain = camera.gain isa AbstractArray ? camera.gain[row_range[1:end-1], col_range[1:end-1]] : camera.gain
    readnoise = camera.readnoise isa AbstractArray ? camera.readnoise[row_range[1:end-1], col_range[1:end-1]] : camera.readnoise
    qe = camera.qe isa AbstractArray ? camera.qe[row_range[1:end-1], col_range[1:end-1]] : camera.qe

    return SCMOSCamera(
        camera.pixel_edges_x[col_range],  # pixel_edges_x (positional)
        camera.pixel_edges_y[row_range],  # pixel_edges_y (positional)
        readnoise = readnoise,
        offset = offset,
        gain = gain,
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

"""
    SingleROI{T}

Single ROI with its location context.

# Fields
- `data::Matrix{T}` - ROI image data (roi_size × roi_size)
- `corner::Tuple{Int32,Int32}` - (x, y) pixel corner in camera coordinates (1-indexed)
- `frame_idx::Int32` - Frame number (1-indexed)
"""
struct SingleROI{T}
    data::Matrix{T}
    corner::Tuple{Int32,Int32}  # (x, y) pixel corner (1-indexed)
    frame_idx::Int32
end

"""
    ROIBatch{T,N,A,C}

Batch of ROIs for efficient processing with location tracking.
Integrates with SMLMData.jl camera types for ecosystem compatibility.

Coordinate System:
- ROI data coordinates: 1-indexed, (1,1) is top-left pixel of ROI
- Camera coordinates: 1-indexed, corners specify (x,y) = (col,row) position in full image
- Frame indices: 1-indexed

# Fields
- `data::A` - ROI stack (roi_size × roi_size × n_rois)
- `corners::Matrix{Int32}` - ROI corners (2 × n_rois) for [x;y] = [col;row] in camera pixels
- `frame_indices::Vector{Int32}` - Frame number for each ROI (1-indexed)
- `camera::C` - SMLMData camera (IdealCamera or SCMOSCamera) for the full image
- `roi_size::Int` - Size of each ROI (square)

# Constructors
```julia
# From arrays
batch = ROIBatch(data, corners, frame_indices, camera)

# From separate x/y vectors
batch = ROIBatch(data, x_corners, y_corners, frame_indices, camera)

# From vector of SingleROI
batch = ROIBatch(rois, camera)
```

# Indexing and Iteration
```julia
roi = batch[i]  # Get SingleROI
for roi in batch
    # Process each SingleROI
end
```
"""
struct ROIBatch{T,N,A<:AbstractArray{T,N},C<:AbstractCamera}
    data::A
    corners::Matrix{Int32}
    frame_indices::Vector{Int32}
    camera::C
    roi_size::Int

    function ROIBatch(data::A, corners::Matrix{Int32}, frame_indices::Vector{Int32},
                     camera::C) where {T,A<:AbstractArray{T,3},C<:AbstractCamera}
        n_rois = size(data, 3)
        roi_size = size(data, 1)
        @assert size(data, 1) == size(data, 2) "ROIs must be square"
        @assert size(corners) == (2, n_rois) "Corners must be 2×n_rois"
        @assert length(frame_indices) == n_rois "Must have one frame index per ROI"
        new{T,3,A,C}(data, corners, frame_indices, camera, roi_size)
    end
end

# Convenience constructor from separate x/y corner vectors
function ROIBatch(data::AbstractArray{T,3}, x_corners::Vector, y_corners::Vector,
                  frame_indices::Vector, camera::C) where {T,C<:AbstractCamera}
    corners = Matrix{Int32}(undef, 2, length(x_corners))
    corners[1, :] = x_corners
    corners[2, :] = y_corners
    ROIBatch(data, corners, Int32.(frame_indices), camera)
end

# Constructor from vector of SingleROI
function ROIBatch(rois::Vector{SingleROI{T}}, camera::C) where {T,C<:AbstractCamera}
    if isempty(rois)
        return ROIBatch(zeros(T, 0, 0, 0), Matrix{Int32}(undef, 2, 0), Int32[], camera)
    end

    roi_size = size(first(rois).data, 1)
    n_rois = length(rois)

    # Pre-allocate arrays
    data = zeros(T, roi_size, roi_size, n_rois)
    corners = Matrix{Int32}(undef, 2, n_rois)
    frame_indices = Vector{Int32}(undef, n_rois)

    for (i, roi) in enumerate(rois)
        data[:, :, i] = roi.data
        corners[:, i] = [roi.corner[1], roi.corner[2]]
        frame_indices[i] = roi.frame_idx
    end

    ROIBatch(data, corners, frame_indices, camera)
end

# Indexing to get individual ROIs
Base.getindex(batch::ROIBatch, i::Int) = SingleROI(
    batch.data[:, :, i],
    (batch.corners[1, i], batch.corners[2, i]),
    batch.frame_indices[i]
)

Base.length(batch::ROIBatch) = size(batch.data, 3)
Base.size(batch::ROIBatch) = (length(batch),)

# Iteration support
Base.iterate(batch::ROIBatch, state=1) = state > length(batch) ? nothing : (batch[state], state + 1)