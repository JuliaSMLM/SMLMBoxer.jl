# Example: Using SMLMBoxer with sCMOS Camera
# This demonstrates the new SMLMData 0.4 integration with sCMOS camera support

using SMLMBoxer
using SMLMData

## Example 1: Basic usage with IdealCamera
println("Example 1: IdealCamera")
println("="^50)

# Create test image
image = zeros(Float32, 256, 256, 10)
# Add some simulated spots
for i in 1:10
    x, y, frame = rand(20:240), rand(20:240), rand(1:10)
    image[x, y, frame] = 1000.0
end

# Create IdealCamera
pixel_size = 0.1  # microns/pixel
camera = IdealCamera(
    pixel_edges_x = Float32.(0:pixel_size:256*pixel_size),
    pixel_edges_y = Float32.(0:pixel_size:256*pixel_size)
)

# Detect boxes
result = getboxes(image, camera;
    boxsize=7,
    overlap=2.0,
    sigma_small=1.0,
    sigma_large=2.0,
    minval=0.1,
    use_gpu=false
)

println("Detected $(result.metadata.ndetections) spots")
println("Boxes shape: $(size(result.boxes))")
println("First detection (pixels): $(result.coords_pixels[1, :])")
println("First detection (microns): $(result.coords_microns[1, :])")
println()

## Example 2: sCMOS Camera with scalar calibration
println("Example 2: SCMOSCamera (scalar params)")
println("="^50)

# Create sCMOS camera with uniform calibration
camera_scmos = SCMOSCamera(
    pixel_edges_x = Float32.(0:pixel_size:256*pixel_size),
    pixel_edges_y = Float32.(0:pixel_size:256*pixel_size),
    offset = 100.0f0,      # ADU
    gain = 2.0f0,          # e-/ADU
    readnoise = 5.0f0,     # e- RMS
    qe = 0.9f0             # quantum efficiency
)

result_scmos = getboxes(image, camera_scmos;
    boxsize=7,
    overlap=2.0,
    sigma_small=1.0,
    sigma_large=2.0,
    minval=0.1,
    use_gpu=false
)

println("Detected $(result_scmos.metadata.ndetections) spots")
println("Camera ROI type: $(typeof(result_scmos.camera_rois[1]))")
println("Camera ROI offset: $(result_scmos.camera_rois[1].offset)")
println("Camera ROI gain: $(result_scmos.camera_rois[1].gain)")
println()

## Example 3: sCMOS Camera with per-pixel calibration
println("Example 3: SCMOSCamera (per-pixel calibration)")
println("="^50)

# Create spatially-varying calibration maps
offset_map = 100.0f0 .+ 10.0f0 .* randn(Float32, 256, 256)  # Varying offset
gain_map = 2.0f0 .+ 0.1f0 .* randn(Float32, 256, 256)       # Varying gain
readnoise_map = 5.0f0 .+ 0.5f0 .* rand(Float32, 256, 256)   # Varying read noise
qe_map = 0.9f0 .+ 0.05f0 .* (rand(Float32, 256, 256) .- 0.5f0)  # Varying QE

camera_scmos_pp = SCMOSCamera(
    pixel_edges_x = Float32.(0:pixel_size:256*pixel_size),
    pixel_edges_y = Float32.(0:pixel_size:256*pixel_size),
    offset = offset_map,
    gain = gain_map,
    readnoise = readnoise_map,
    qe = qe_map
)

result_scmos_pp = getboxes(image, camera_scmos_pp;
    boxsize=7,
    overlap=2.0,
    sigma_small=1.0,
    sigma_large=2.0,
    minval=0.1,
    use_gpu=false
)

println("Detected $(result_scmos_pp.metadata.ndetections) spots")
println("Camera ROI type: $(typeof(result_scmos_pp.camera_rois[1]))")
println("Camera ROI has per-pixel calibration: $(result_scmos_pp.camera_rois[1].offset isa AbstractArray)")
if result_scmos_pp.camera_rois[1].offset isa AbstractArray
    println("Camera ROI offset shape: $(size(result_scmos_pp.camera_rois[1].offset))")
    println("Expected shape for 7x7 box: (7, 7)")
end
println()

## Example 3b: Variance-Weighted Detection Demo
println("Example 3b: Variance-Weighted Detection (CPU)")
println("="^50)

# Create image with two spots of equal intensity
demo_image = zeros(Float32, 100, 100)
demo_image[30, 30] = 200.0  # Spot in low-noise region
demo_image[70, 70] = 200.0  # Spot in high-noise region

# Create readnoise map with varying noise
demo_readnoise = 2.0f0 .* ones(Float32, 100, 100)
demo_readnoise[60:80, 60:80] .= 20.0f0  # 10x more noise in this region

demo_camera = SCMOSCamera(
    pixel_edges_x = Float32.(0:pixel_size:100*pixel_size),
    pixel_edges_y = Float32.(0:pixel_size:100*pixel_size),
    offset = 100.0f0,
    gain = 2.0f0,
    readnoise = demo_readnoise,  # Per-pixel noise map
    qe = 0.9f0
)

# Detect with variance weighting (CPU)
result_weighted_cpu = getboxes(demo_image, demo_camera;
    boxsize=7,
    overlap=2.0,
    sigma_small=1.0,
    sigma_large=2.0,
    minval=1.0,
    use_gpu=false
)

println("Variance-weighted detection (CPU) found $(result_weighted_cpu.metadata.ndetections) spots")
println("Note: Spots in low-noise regions are preferentially detected")
println("High-noise regions (10x readnoise) are down-weighted during filtering")
println()

## Example 3c: GPU Accelerated sCMOS Detection (if available)
println("Example 3c: GPU Accelerated sCMOS Detection")
println("="^50)

if CUDA.functional()
    # Same detection but with GPU acceleration via KernelAbstractions
    result_weighted_gpu = getboxes(demo_image, demo_camera;
        boxsize=7,
        overlap=2.0,
        sigma_small=1.0,
        sigma_large=2.0,
        minval=1.0,
        use_gpu=true  # KernelAbstractions backend automatically selects GPU
    )

    println("Variance-weighted detection (GPU) found $(result_weighted_gpu.metadata.ndetections) spots")
    println("GPU acceleration via KernelAbstractions.jl")
    println("Same kernel code runs on CPU/GPU (device-agnostic)")
else
    println("CUDA not available - GPU acceleration disabled")
    println("Install CUDA.jl and have compatible GPU for GPU support")
end
println()

## Example 4: Integration with GaussMLE (conceptual)
println("Example 4: Workflow for GaussMLE integration")
println("="^50)

println("""
# Complete workflow:
# 1. Detect boxes with SMLMBoxer
result = getboxes(imagestack, scmos_camera; boxsize=7)

# 2. Pass to GaussMLE for fitting (kernel-abstract branch)
using GaussMLE
fitter = GaussMLEFitter(
    GaussianXYNB,           # PSF model
    result.camera_rois;      # sCMOS calibration for each box
    device=:gpu
)

# 3. Fit the boxes
fitted_smld = fit(fitter, result.boxes, result.coords_microns)
# Returns: BasicSMLD{Float32, Emitter2DFit{Float32}}

# 4. Access fitted emitters
for emitter in fitted_smld.emitters
    println("Position: (\$(emitter.x), \$(emitter.y)) μm")
    println("Photons: \$(emitter.photons)")
    println("Uncertainty: (\$(emitter.σ_x), \$(emitter.σ_y)) μm")
end
""")

println()
println("="^50)
println("Example complete!")
