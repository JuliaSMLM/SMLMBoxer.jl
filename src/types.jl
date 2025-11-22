"""
    get_pixel_size(camera::AbstractCamera)

Extract pixel size from camera pixel edges (in microns).
Assumes approximately square pixels - returns x-direction pixel size.

For non-square pixels, pixel_size_x and pixel_size_y may differ slightly.
This function returns pixel_size_x for simplicity.

# Arguments
- `camera`: Camera object (IdealCamera or SCMOSCamera)

# Returns
- Pixel size in microns (x-direction)
"""
function get_pixel_size(camera::AbstractCamera)
    # Pixel size is the difference between consecutive edges
    return camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
end

"""
    photons_to_dog_threshold(min_photons, psf_sigma)

Convert total photon count threshold to DoG filter peak intensity threshold.

# Arguments
- `min_photons`: Minimum total photons for detection
- `psf_sigma`: PSF sigma in pixels

# Returns
- `minval`: DoG filter intensity threshold

# Physics
For a 2D Gaussian PSF with total photons N and sigma σ_psf, the peak intensity is:
    I_peak = N / (2π σ_psf²)

After convolution with the small Gaussian filter (sigma_small = 1.0 × psf_sigma),
the effective sigma becomes:
    σ_eff = √(σ_psf² + sigma_small²)

The peak after filtering is:
    I_filtered = N / (2π σ_eff²)

The DoG response (small - large Gaussian) has a lower peak than the small Gaussian alone.
For sigma_large = 2×sigma_small, empirical testing shows the DoG peak is approximately
0.65 times the small Gaussian peak.
"""
function photons_to_dog_threshold(min_photons::Real, psf_sigma::Real)
    # DoG filter uses sigma_small = 1.0 × psf_sigma
    sigma_small = 1.0 * psf_sigma

    # Effective sigma after small Gaussian blur (convolution of two Gaussians)
    σ_eff = sqrt(psf_sigma^2 + sigma_small^2)

    # Peak intensity after small Gaussian filtering
    peak_filtered = min_photons / (2π * σ_eff^2)

    # DoG reduces peak (empirical factor for sigma_large = 2×sigma_small)
    dog_factor = 0.65

    return Float32(dog_factor * peak_filtered)
end

"""
    GetBoxesArgs

Internal structure for getboxes parameters. Users should call getboxes() with
keyword arguments rather than constructing this directly.

# Primary Interface (Recommended)
- `psf_sigma::Real`: PSF sigma in microns (physical units, e.g., 0.13 for 130nm PSF)
  Requires camera to be provided for pixel size conversion.
- `min_photons::Real`: Minimum total photons for detection (default: 500.0)

When psf_sigma is provided:
- Converted to pixels using camera pixel size
- sigma_small = 1.0 × psf_sigma_pixels (automatically calculated)
- sigma_large = 2.0 × psf_sigma_pixels (automatically calculated)
- minval = photons_to_dog_threshold(min_photons, psf_sigma_pixels) (automatically calculated)

# Advanced Interface (Direct Control)
- `sigma_small::Real`: Small Gaussian sigma in pixels (default: 1.0)
- `sigma_large::Real`: Large Gaussian sigma in pixels (default: 2.0)
- `minval::Real`: DoG intensity threshold (default: 0.0)

# Other Parameters
- `imagestack`: Input image stack
- `camera`: Camera object (IdealCamera or SCMOSCamera)
- `boxsize::Int`: ROI box size in pixels (default: 7)
- `overlap::Real`: Maximum overlap between detections in pixels (default: 2.0)
- `use_gpu::Bool`: Use GPU acceleration (default: true)
"""
mutable struct GetBoxesArgs
    imagestack::AbstractArray
    camera::Union{AbstractCamera,Nothing}
    boxsize::Int
    overlap::Float32
    sigma_small::Float32
    sigma_large::Float32
    minval::Float32
    use_gpu::Bool

    # Inner constructor handles conversion logic
    function GetBoxesArgs(;
        imagestack = rand(Float32, 256, 256, 50) .> 0.999,
        camera::Union{AbstractCamera,Nothing} = nothing,
        boxsize::Int = 7,
        overlap::Real = 2.0,
        psf_sigma::Union{Real,Nothing} = nothing,
        min_photons::Real = 500.0,
        sigma_small::Union{Real,Nothing} = nothing,
        sigma_large::Union{Real,Nothing} = nothing,
        minval::Union{Real,Nothing} = nothing,
        use_gpu::Bool = true
    )
        # Determine which interface is being used
        if psf_sigma !== nothing
            # NEW INTERFACE: PSF-aware detection (recommended)
            # psf_sigma is in physical units (microns) - convert to pixels
            if camera !== nothing
                pixel_size_μm = get_pixel_size(camera)
                psf_sigma_pixels = psf_sigma / pixel_size_μm
            else
                error("psf_sigma in physical units (microns) requires camera to be provided. " *
                      "Either provide a camera or use the advanced interface with sigma_small/sigma_large in pixels.")
            end

            σ_small = Float32(1.0 * psf_sigma_pixels)
            σ_large = Float32(2.0 * psf_sigma_pixels)
            min_val = photons_to_dog_threshold(min_photons, psf_sigma_pixels)
        else
            # OLD INTERFACE: Direct control (backward compatible)
            σ_small = Float32(sigma_small !== nothing ? sigma_small : 1.0)
            σ_large = Float32(sigma_large !== nothing ? sigma_large : 2.0)
            min_val = Float32(minval !== nothing ? minval : 0.0)
        end

        new(imagestack, camera, boxsize, Float32(overlap), σ_small, σ_large, min_val, use_gpu)
    end
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
    # Always Float32 to match imagestack type (getboxes converts all images to Float32)
    return ones(Float32, imagesize)
end

function get_variance_map(camera::SCMOSCamera{T}, imagesize::Tuple{Int,Int}) where T
    nrows, ncols = imagesize

    if camera.readnoise isa AbstractArray
        # Per-pixel readnoise map: variance = readnoise²
        variance_map = camera.readnoise .^ 2
        # Ensure it matches the image size
        @assert size(variance_map) == imagesize "Readnoise map size $(size(variance_map)) doesn't match image size $imagesize"
        # Convert to Float32 to match imagestack type (getboxes converts all images to Float32)
        return Float32.(variance_map)
    else
        # Scalar readnoise: uniform variance (always Float32 to match imagestack)
        return fill(Float32(camera.readnoise^2), imagesize)
    end
end