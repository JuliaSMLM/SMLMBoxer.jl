"""
Blinking Workflow Example - sCMOS Camera

Demonstrates realistic SMLM detection workflow with sCMOS camera:
1. Simulate multi-frame blinking data with SMLMSim
2. Generate sCMOS camera images with spatially-varying noise
3. Detect spots using variance-weighted DoG filtering
4. Compare to IdealCamera performance

This example shows the advantage of variance-weighted detection for sCMOS data.

Run with: julia --project=. blinking_workflow_scmos.jl
"""

using SMLMSim
using SMLMBoxer
using SMLMData
using MicroscopePSFs
using Statistics
using Printf
using FileIO
using ImageCore: Gray

println("="^80)
println("SMLMBoxer: Blinking Workflow with sCMOS Camera")
println("="^80)
println()

# ============================================================================
# Step 1: Setup sCMOS Camera with Spatial Noise Variation
# ============================================================================
println("Step 1: Setting up sCMOS camera with spatial noise...")

pixel_size = 0.1f0  # 100nm pixels
n_pixels = 256

# Create per-pixel readnoise map with spatial variation
# Simulate realistic sCMOS: low noise center, higher noise edges
readnoise_map = zeros(Float32, n_pixels, n_pixels)
center = n_pixels ÷ 2
for i in 1:n_pixels
    for j in 1:n_pixels
        # Distance from center
        dx = (i - center) / center
        dy = (j - center) / center
        dist = sqrt(dx^2 + dy^2)

        # Readnoise increases with distance from center
        # Center: ~3e⁻, edges: ~8e⁻
        readnoise_map[i, j] = 3.0f0 + 5.0f0 * dist
    end
end

camera_scmos = SCMOSCamera(
    n_pixels,           # npixels_x
    n_pixels,           # npixels_y
    pixel_size,         # pixel size
    readnoise_map,      # per-pixel readnoise (e⁻)
    offset = 100.0f0,   # camera offset (ADU)
    gain = 2.0f0,       # gain (e⁻/ADU)
    qe = 0.9f0          # quantum efficiency
)

# Also create IdealCamera for comparison
camera_ideal = IdealCamera(n_pixels, n_pixels, pixel_size)

println("  sCMOS Camera: $(n_pixels)×$(n_pixels) pixels")
println("  Pixel size: $(pixel_size * 1000) nm")
println("  FOV: $(n_pixels * pixel_size) μm × $(n_pixels * pixel_size) μm")
println("  Readnoise: $(round(minimum(readnoise_map), digits=1)) - $(round(maximum(readnoise_map), digits=1)) e⁻ (spatial variation)")
println("  Gain: $(camera_scmos.gain) e⁻/ADU")
println("  Offset: $(camera_scmos.offset) ADU")
println()

# ============================================================================
# Step 2: Setup Simulation Parameters (Same as IdealCamera example)
# ============================================================================
println("Step 2: Setting up simulation parameters...")

sim_params = StaticSMLMParams(
    density = 0.3,          # 0.3 patterns per μm² (fast demo, ~200 patterns total)
    σ_psf = 0.13,           # 130nm PSF width
    nframes = 100,          # 100 frames of blinking
    framerate = 20.0,       # 20 fps
    ndims = 2               # 2D simulation
)

pattern = Nmer2D(n=8, d=0.15)  # Octamer

molecule = GenericFluor(
    photons = 1500.0,       # 1500 photons per localization
    k_off = 8.0,            # Fast off-switching
    k_on = 0.08             # Sparse blinking
)

# Extract k_on and k_off from rate matrix
k_off = molecule.q[1, 2]
k_on = molecule.q[2, 1]

println("  Pattern: $(pattern.n)-mer, diameter=$(pattern.d * 1000) nm")
println("  Density: $(sim_params.density) patterns/μm²")
println("  Frames: $(sim_params.nframes)")
println("  Photon rate: $(molecule.γ) Hz")
println("  Blinking: k_off=$(k_off) Hz, k_on=$(k_on) Hz")
println("  Duty cycle: $(round(k_on / (k_on + k_off) * 100, digits=1))%")
println()

# ============================================================================
# Step 3: Simulate Blinking SMLM Data
# ============================================================================
println("Step 3: Simulating blinking SMLM data...")

t_start = time()
_, _, smld_ground_truth = simulate(sim_params; pattern=pattern, molecule=molecule, camera=camera_scmos)
t_sim = time() - t_start

n_emitters = length(smld_ground_truth.emitters)
println("  Simulation complete ($(round(t_sim, digits=2))s)")
println("  Ground truth emitters: $n_emitters")
println("  Emitters per frame (avg): $(round(n_emitters / sim_params.nframes, digits=1))")
println()

# ============================================================================
# Step 4: Generate sCMOS Camera Images
# ============================================================================
println("Step 4: Generating sCMOS camera images...")

psf = GaussianPSF(Float32(sim_params.σ_psf))

# Generate images with full sCMOS noise model
t_start = time()
images_scmos = gen_images(smld_ground_truth, psf;
    bg = 10.0,              # 10 photons/pixel background
    camera_noise = true     # Full sCMOS noise: QE, Poisson, readnoise, gain, offset
)
t_gen = time() - t_start

println("  sCMOS image generation complete ($(round(t_gen, digits=2))s)")
println("  Image stack: $(size(images_scmos))")
println("  Value range: [$(round(minimum(images_scmos), digits=1)), $(round(maximum(images_scmos), digits=1))] ADU")
println("  (includes offset=$(camera_scmos.offset) ADU)")
println()

# Also generate ideal camera images for comparison
println("  Generating IdealCamera images for comparison...")
images_ideal = gen_images(smld_ground_truth, psf;
    bg = 10.0,
    poisson_noise = true
)
println("  IdealCamera images: [$(round(minimum(images_ideal), digits=1)), $(round(maximum(images_ideal), digits=1))] photons")
println()

# ============================================================================
# Step 5: Detect with sCMOS (Variance-Weighted)
# ============================================================================
println("Step 5: Detecting with sCMOS variance-weighted filtering...")

println("  PSF sigma: $(sim_params.σ_psf) μm (= $(round(sim_params.σ_psf / pixel_size, digits=2)) pixels)")
println("  Detection threshold: 50 photons (matched to framerate)")
println("  Using variance-weighted DoG filtering (sCMOS-optimized)")
println()

t_start = time()
roi_batch_scmos = getboxes(images_scmos, camera_scmos;
    psf_sigma = sim_params.σ_psf,  # Physical units (microns)
    min_photons = 50.0,
    boxsize = 11,
    overlap = 3.0,
    use_gpu = false
)
t_detect_scmos = time() - t_start

n_detected_scmos = length(roi_batch_scmos)
println("  sCMOS detection complete ($(round(t_detect_scmos * 1000, digits=1)) ms)")
println("  Detected: $n_detected_scmos")
println("  Detection rate: $(round(n_detected_scmos / n_emitters * 100, digits=1))%")
println()

# ============================================================================
# Step 6: Detect with IdealCamera (for comparison)
# ============================================================================
println("Step 6: Detecting with IdealCamera (for comparison)...")

t_start = time()
roi_batch_ideal = getboxes(images_ideal, camera_ideal;
    psf_sigma = sim_params.σ_psf,  # Physical units (microns)
    min_photons = 50.0,
    boxsize = 11,
    overlap = 3.0,
    use_gpu = false
)
t_detect_ideal = time() - t_start

n_detected_ideal = length(roi_batch_ideal)
println("  IdealCamera detection complete ($(round(t_detect_ideal * 1000, digits=1)) ms)")
println("  Detected: $n_detected_ideal")
println("  Detection rate: $(round(n_detected_ideal / n_emitters * 100, digits=1))%")
println()

# ============================================================================
# Step 7: Compare Performance
# ============================================================================
println("="^80)
println("COMPARISON: sCMOS vs IdealCamera")
println("="^80)
println()

println("Detection Results:")
println("  Ground truth emitters: $n_emitters")
println()
println("  sCMOS (variance-weighted):")
println("    Detected: $n_detected_scmos")
println("    Rate: $(round(n_detected_scmos / n_emitters * 100, digits=1))%")
println("    Time: $(round(t_detect_scmos * 1000, digits=1)) ms")
println()
println("  IdealCamera (standard DoG):")
println("    Detected: $n_detected_ideal")
println("    Rate: $(round(n_detected_ideal / n_emitters * 100, digits=1))%")
println("    Time: $(round(t_detect_ideal * 1000, digits=1)) ms")
println()

# Calculate detection difference
diff_detected = n_detected_scmos - n_detected_ideal
diff_pct = (n_detected_scmos - n_detected_ideal) / n_emitters * 100

println("  Difference (sCMOS - Ideal):")
println("    Δ Detections: $(diff_detected > 0 ? "+" : "")$(diff_detected)")
println("    Δ Rate: $(diff_pct > 0 ? "+" : "")$(round(diff_pct, digits=1))%")
if abs(diff_detected) > 0
    if diff_detected > 0
        println("    → Variance weighting improved detection by $(round(abs(diff_pct), digits=1))%")
    else
        println("    → Note: Ideal camera performed better (less noise)")
    end
end
println()

println("Processing Speed:")
println("  sCMOS: $(round(sim_params.nframes / t_detect_scmos, digits=1)) frames/sec")
println("  Ideal: $(round(sim_params.nframes / t_detect_ideal, digits=1)) frames/sec")
if t_detect_scmos > t_detect_ideal
    println("  → sCMOS is $(round(t_detect_scmos / t_detect_ideal, digits=2))x slower (variance computation overhead)")
else
    println("  → Similar performance")
end
println()

# ============================================================================
# Step 8: Noise Analysis
# ============================================================================
println("="^80)
println("NOISE CHARACTERISTICS")
println("="^80)
println()

println("sCMOS Readnoise Map:")
println("  Min: $(round(minimum(readnoise_map), digits=2)) e⁻ (center)")
println("  Max: $(round(maximum(readnoise_map), digits=2)) e⁻ (edges)")
println("  Mean: $(round(mean(readnoise_map), digits=2)) e⁻")
println("  Std: $(round(std(readnoise_map), digits=2)) e⁻")
println()

println("Detection in High-Noise Regions:")
println("  Variance weighting downweights high-noise pixels")
println("  → Reduces false positives in noisy regions")
println("  → Maintains sensitivity in low-noise regions")
println()

println("="^80)
println("Example complete!")
println("="^80)
println()
println("Key Insights:")
println("  ✓ sCMOS variance-weighted detection handles spatial noise variation")
println("  ✓ Same PSF-aware interface works for both camera types")
println("  ✓ Starting from images + camera (realistic workflow)")
println("  ✓ Full sCMOS noise model: QE, Poisson, readnoise, gain, offset")
println()
println("Next steps:")
println("  → Use roi_batch_scmos.data with GaussMLE for fitting")
println("  → Camera calibration automatically applied during fitting")
println()
