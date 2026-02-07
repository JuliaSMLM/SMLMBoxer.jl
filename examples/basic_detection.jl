"""
Basic Detection Example

Demonstrates the complete workflow:
1. Generate synthetic SMLM data with SMLMSim
2. Detect spots using SMLMBoxer.getboxes()
3. Validate detection accuracy

Run with: julia --project=. basic_detection.jl
"""

using SMLMSim
using SMLMBoxer
using SMLMData
using MicroscopePSFs
using CUDA
using Statistics
using Printf

println("="^80)
println("SMLMBoxer Basic Detection Example")
println("="^80)
println()

# ============================================================================
# Step 1: Create Ground Truth Emitters
# ============================================================================
println("Step 1: Creating ground truth emitters...")

pixel_size = 0.1f0  # μm/pixel
n_pixels = 256
box_size = n_pixels * pixel_size  # μm

# Create grid of emitters with known positions
emitters = Emitter2DFit{Float64}[]
n_emitters_per_axis = 10
spacing = box_size / (n_emitters_per_axis + 1)

for i in 1:n_emitters_per_axis
    for j in 1:n_emitters_per_axis
        x = Float64(i * spacing)
        y = Float64(j * spacing)
        photons = 1000.0 + 500.0 * rand()  # 1000-1500 photons
        bg = 10.0  # background photons/pixel

        push!(emitters, Emitter2DFit{Float64}(
            x, y,           # position (μm)
            photons, bg,    # photons and background
            0.0, 0.0,       # uncertainties (not used for simulation)
            0.0, 0.0,       # uncertainties
            1,              # frame
            1, 0, length(emitters)+1  # dataset, track_id, id
        ))
    end
end

println("  Created $(length(emitters)) emitters in grid pattern")
println("  Spacing: $(round(spacing, digits=3)) μm")
println()

# ============================================================================
# Step 2: Create Camera and SMLD
# ============================================================================
println("Step 2: Setting up camera...")

camera = IdealCamera(
    1:n_pixels,      # pixel range x
    1:n_pixels,      # pixel range y
    pixel_size       # pixel size
)

smld = BasicSMLD(emitters, camera, 1, 1)
println("  Camera: $(n_pixels)×$(n_pixels) pixels @ $(pixel_size) μm/pixel")
println()

# ============================================================================
# Step 3: Generate Synthetic Images
# ============================================================================
println("Step 3: Generating synthetic images with SMLMSim...")

psf = GaussianPSF(0.13f0)  # σ = 130 nm
println("  PSF: Gaussian with σ = 0.13 μm")

# Generate clean image (no noise)
println("  Generating clean image...")
img_clean = Float32.(gen_images(smld, psf, bg=10.0))

# Generate noisy image (Poisson + readnoise)
println("  Generating noisy image (Poisson + readnoise)...")
img_noisy = Float32.(gen_images(smld, psf, poisson_noise=true, bg=10.0))

println("  Image size: $(size(img_noisy))")
println("  Signal range: [$(round(minimum(img_noisy), digits=1)), $(round(maximum(img_noisy), digits=1))] ADU")
println()

# ============================================================================
# Step 4: Detect Boxes with SMLMBoxer (PSF-Aware Interface)
# ============================================================================
println("Step 4: Detecting spots with SMLMBoxer (PSF-aware interface)...")

# Calculate PSF sigma in pixels
psf_sigma_pixels = 0.13f0 / pixel_size  # 0.13 μm / 0.1 μm/pixel = 1.3 pixels
println("  PSF sigma: $(round(psf_sigma_pixels, digits=2)) pixels")
println("  Detection threshold: 500 photons")
println()

# Detect on clean image (CPU)
println("  Running on clean image (CPU)...")
(roi_batch_clean, info_clean) = getboxes(img_clean, camera;
    psf_sigma = psf_sigma_pixels,  # PSF-aware detection
    min_photons = 500.0,            # Detect emitters with ≥500 photons
    boxsize = 11,                   # Larger box for better fitting
    overlap = 3.0,
    backend = :cpu
)
t_clean = info_clean.elapsed_s

# Detect on noisy image (CPU)
println("  Running on noisy image (CPU)...")
(roi_batch_noisy, info_noisy) = getboxes(img_noisy, camera;
    psf_sigma = psf_sigma_pixels,
    min_photons = 500.0,
    boxsize = 11,
    overlap = 3.0,
    backend = :cpu
)
t_noisy = info_noisy.elapsed_s

# Try GPU if available
if CUDA.functional()
    println("  Running on noisy image (GPU)...")
    (roi_batch_gpu, info_gpu) = getboxes(img_noisy, camera;
        psf_sigma = psf_sigma_pixels,
        min_photons = 500.0,
        boxsize = 11,
        overlap = 3.0,
        backend = :gpu
    )
    t_gpu = info_gpu.elapsed_s
else
    roi_batch_gpu = nothing
    info_gpu = nothing
    t_gpu = NaN
end

println()

# ============================================================================
# Step 5: Results and Validation
# ============================================================================
println("="^80)
println("RESULTS")
println("="^80)
println()

println("Detection Summary:")
println("  Ground truth emitters:    $(length(emitters))")
println("  Detected (clean image):   $(length(roi_batch_clean))")
println("  Detected (noisy image):   $(length(roi_batch_noisy))")
if CUDA.functional()
    println("  Detected (GPU):           $(length(roi_batch_gpu))")
end
println()

println("Detection Accuracy:")
accuracy_clean = length(roi_batch_clean) / length(emitters) * 100
accuracy_noisy = length(roi_batch_noisy) / length(emitters) * 100
println(@sprintf("  Clean image: %.1f%%", accuracy_clean))
println(@sprintf("  Noisy image: %.1f%%", accuracy_noisy))
if CUDA.functional()
    accuracy_gpu = length(roi_batch_gpu) / length(emitters) * 100
    println(@sprintf("  GPU: %.1f%%", accuracy_gpu))
end
println()

println("Processing Time:")
println(@sprintf("  Clean image (CPU): %.3f sec", t_clean))
println(@sprintf("  Noisy image (CPU): %.3f sec", t_noisy))
if CUDA.functional()
    println(@sprintf("  Noisy image (GPU): %.3f sec", t_gpu))
    println(@sprintf("  GPU speedup: %.1fx", t_noisy / t_gpu))
end
println()

println("ROIBatch Structure:")
println("  Type: $(typeof(roi_batch_noisy))")
println("  ROI data size: $(size(roi_batch_noisy.data))")
println("  X corners: $(length(roi_batch_noisy.x_corners)) positions")
println("  Y corners: $(length(roi_batch_noisy.y_corners)) positions")
println("  Frame indices: $(length(roi_batch_noisy.frame_indices))")
println("  Camera type: $(typeof(roi_batch_noisy.camera))")
println("  ROI size: $(roi_batch_noisy.roi_size)")
println()

# Show first few detections
if length(roi_batch_noisy) > 0
    println("First 5 Detections (corner positions):")
    println("  [x, y] = [col, row] in camera pixels")
    for i in 1:min(5, length(roi_batch_noisy))
        x_corner = roi_batch_noisy.x_corners[i]
        y_corner = roi_batch_noisy.y_corners[i]
        frame = roi_batch_noisy.frame_indices[i]
        println(@sprintf("    ROI %d: corner=(%3d, %3d), frame=%d", i, x_corner, y_corner, frame))
    end
    println()
end

println("="^80)
println("Example complete!")
println("="^80)
