"""
Blinking Workflow Example - IdealCamera

Demonstrates the complete realistic SMLM detection workflow:
1. Simulate multi-frame blinking data with SMLMSim
2. Generate camera images with noise
3. Detect spots using PSF-aware interface
4. Validate detection across frames

This example shows what a real user would do starting from images+camera.

Run with: julia --project=. blinking_workflow_ideal.jl
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
println("SMLMBoxer: Blinking Workflow with IdealCamera")
println("="^80)
println()

# ============================================================================
# Step 1: Setup Camera
# ============================================================================
println("Step 1: Setting up camera...")

pixel_size = 0.1f0  # 100nm pixels
n_pixels = 256
camera = IdealCamera(n_pixels, n_pixels, pixel_size)

println("  Camera: $(n_pixels)×$(n_pixels) pixels")
println("  Pixel size: $(pixel_size * 1000) nm")
println("  FOV: $(n_pixels * pixel_size) μm × $(n_pixels * pixel_size) μm")
println()

# ============================================================================
# Step 2: Setup Simulation Parameters (Blinking)
# ============================================================================
println("Step 2: Setting up simulation parameters...")

# Simulation parameters for blinking SMLM (optimized for quick demo)
sim_params = StaticSMLMParams(
    density = 0.3,          # 0.3 patterns per μm² (fast demo, ~200 patterns total)
    σ_psf = 0.13,           # 130nm PSF width
    nframes = 100,          # 100 frames of blinking
    framerate = 20.0,       # 20 fps
    ndims = 2               # 2D simulation
)

# Pattern: Octamer (8-mer with 150nm diameter)
pattern = Nmer2D(n=8, d=0.15)

# Fluorophore with blinking dynamics
molecule = GenericFluor(
    photons = 1500.0,       # 1500 photons per localization
    k_off = 8.0,            # Off-switching rate (1/frames) - fast off
    k_on = 0.08             # On-switching rate (1/frames) - sparse blinking
)

println("  Simulation:")
println("    Pattern: $(pattern.n)-mer, diameter=$(pattern.d * 1000) nm")
println("    Density: $(sim_params.density) patterns/μm²")
println("    Total emitters: $(sim_params.density * pattern.n) emitters/μm²")
println("    PSF σ: $(sim_params.σ_psf * 1000) nm")
println("    Frames: $(sim_params.nframes)")
println()
# Extract k_on and k_off from rate matrix
k_off = molecule.q[1, 2]  # Off-switching rate
k_on = molecule.q[2, 1]   # On-switching rate

println("  Fluorophore dynamics:")
println("    Photon rate: $(molecule.γ) Hz")
println("    k_off: $(k_off) Hz (off-switching rate)")
println("    k_on: $(k_on) Hz (on-switching rate)")
println("    Expected duty cycle: $(k_on / (k_on + k_off) * 100)%")
println()

# ============================================================================
# Step 3: Simulate Blinking SMLM Data
# ============================================================================
println("Step 3: Simulating blinking SMLM data...")

# Run simulation (returns pattern, smld_true, smld_noisy)
t_start = time()
_, _, smld_ground_truth = simulate(sim_params; pattern=pattern, molecule=molecule, camera=camera)
t_sim = time() - t_start

n_emitters = length(smld_ground_truth.emitters)
n_patterns = sim_params.density * (n_pixels * pixel_size)^2

println("  Simulation complete ($(round(t_sim, digits=2))s)")
println("  Ground truth patterns: $(round(Int, n_patterns))")
println("  Total ground truth emitters: $n_emitters")
println("  Emitters per frame (avg): $(round(n_emitters / sim_params.nframes, digits=1))")
println()

# Analyze blinking statistics
frames_with_emitters = unique([e.frame for e in smld_ground_truth.emitters])
println("  Blinking statistics:")
println("    Frames with emitters: $(length(frames_with_emitters)) / $(sim_params.nframes)")
println("    Min emitters/frame: $(minimum([count(e -> e.frame == f, smld_ground_truth.emitters) for f in frames_with_emitters]))")
println("    Max emitters/frame: $(maximum([count(e -> e.frame == f, smld_ground_truth.emitters) for f in frames_with_emitters]))")
println()

# ============================================================================
# Step 4: Generate Camera Images
# ============================================================================
println("Step 4: Generating camera images...")

psf = GaussianPSF(Float32(sim_params.σ_psf))

# Generate images with Poisson noise and background
t_start = time()
images = gen_images(smld_ground_truth, psf;
    bg = 10.0,              # 10 photons/pixel background
    poisson_noise = true    # Add Poisson noise
)
t_gen = time() - t_start

println("  Image generation complete ($(round(t_gen, digits=2))s)")
println("  Image stack: $(size(images))")
println("  Data type: $(eltype(images))")
println("  Value range: [$(round(minimum(images), digits=1)), $(round(maximum(images), digits=1))] photons")
println()

# ============================================================================
# Step 5: Detect Spots (PSF-Aware Interface with Physical Units)
# ============================================================================
println("Step 5: Detecting spots with PSF-aware interface (physical units)...")

println("  PSF sigma: $(sim_params.σ_psf) μm (= $(round(sim_params.σ_psf / pixel_size, digits=2)) pixels)")
println("  Detection threshold: 50 photons (matched to framerate)")
println()

# Detect using PSF-aware interface with physical units
(roi_batch, info) = getboxes(images, camera;
    psf_sigma = sim_params.σ_psf,  # PSF sigma in microns (auto-converts to pixels)
    min_photons = 50.0,             # Photon threshold (auto-converts to intensity)
    boxsize = 11,                   # 11×11 pixel ROIs
    overlap = 3.0,                  # Max 3 pixel overlap
    use_gpu = false                 # Use CPU (set to true if CUDA available)
)
t_detect = info.elapsed_s

n_detected = length(roi_batch)

println("  Detection complete ($(round(t_detect * 1000, digits=1)) ms)")
println("  Detected ROIs: $n_detected")
println("  Detection rate: $(round(n_detected / n_emitters * 100, digits=1))%")
println()

# ============================================================================
# Step 6: Analyze Detection Per Frame
# ============================================================================
println("Step 6: Analyzing detection statistics...")

# Ground truth per frame
gt_per_frame = Dict{Int, Int}()
for e in smld_ground_truth.emitters
    gt_per_frame[e.frame] = get(gt_per_frame, e.frame, 0) + 1
end

# Detections per frame
det_per_frame = Dict{Int, Int}()
for f in roi_batch.frame_indices
    det_per_frame[f] = get(det_per_frame, f, 0) + 1
end

# Calculate per-frame detection rates
detection_rates = Float64[]
for frame in 1:sim_params.nframes
    n_gt = get(gt_per_frame, frame, 0)
    n_det = get(det_per_frame, frame, 0)
    if n_gt > 0
        rate = n_det / n_gt
        push!(detection_rates, rate)
    end
end

println("  Per-frame analysis:")
println("    Frames with ground truth: $(length(gt_per_frame))")
println("    Frames with detections: $(length(det_per_frame))")
if !isempty(detection_rates)
    println("    Mean detection rate: $(round(mean(detection_rates) * 100, digits=1))%")
    println("    Std detection rate: $(round(std(detection_rates) * 100, digits=1))%")
    println("    Min detection rate: $(round(minimum(detection_rates) * 100, digits=1))%")
    println("    Max detection rate: $(round(maximum(detection_rates) * 100, digits=1))%")
end
println()

# ============================================================================
# Step 7: Summary Statistics
# ============================================================================
println("="^80)
println("SUMMARY")
println("="^80)
println()

println("Simulation:")
println("  Patterns: $(round(Int, n_patterns))")
println("  Ground truth emitters: $n_emitters")
println("  Frames: $(sim_params.nframes)")
println("  Emitters/frame (avg): $(round(n_emitters / sim_params.nframes, digits=1))")
println()

psf_sigma_pixels_calc = sim_params.σ_psf / pixel_size

println("Detection (PSF-aware interface):")
println("  PSF sigma: $(sim_params.σ_psf) μm (= $(round(psf_sigma_pixels_calc, digits=2)) pixels)")
println("  Photon threshold: 50.0")
println("  → DoG sigma_small: $(round(1.0 * psf_sigma_pixels_calc, digits=2)) pixels")
println("  → DoG sigma_large: $(round(2.0 * psf_sigma_pixels_calc, digits=2)) pixels")
println("  → Intensity threshold: $(round(SMLMBoxer.photons_to_dog_threshold(50.0, psf_sigma_pixels_calc), digits=2)) ADU")
println()

println("Results:")
println("  Detected: $n_detected")
println("  Detection rate: $(round(n_detected / n_emitters * 100, digits=1))%")
println("  Processing time: $(round(t_detect * 1000, digits=1)) ms")
println("  Throughput: $(round(sim_params.nframes / t_detect, digits=1)) frames/sec")
println()

println("="^80)
println("Example complete!")
println("="^80)
println()
println("This example demonstrates:")
println("  ✓ Realistic blinking simulation with SMLMSim")
println("  ✓ Multi-frame image stack generation")
println("  ✓ PSF-aware detection interface")
println("  ✓ Starting from images + camera (user workflow)")
println("  ✓ Frame-by-frame validation")
println()

# ============================================================================
# Step 8: Save Output Files
# ============================================================================
println("Step 7: Saving output files...")

output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

# Save example input images (first 3 frames)
for i in 1:min(3, size(images, 3))
    frame = images[:, :, i]
    # Normalize to 0-1 for saving
    frame_min, frame_max = extrema(frame)
    if frame_max > frame_min
        frame_norm = (frame .- frame_min) ./ (frame_max - frame_min)
    else
        frame_norm = zeros(size(frame))
    end
    filename = joinpath(output_dir, "input_frame_$(lpad(i,3,'0')).png")
    save(filename, Gray.(Float32.(frame_norm)))
    println("  Saved: input_frame_$(lpad(i,3,'0')).png")
end

# Save example ROI images (first 9 detected ROIs as 3x3 montage)
if n_detected > 0
    n_rois_to_save = min(9, n_detected)
    roi_montage = zeros(Float32, roi_batch.roi_size * 3, roi_batch.roi_size * 3)

    for i in 1:n_rois_to_save
        row_idx = div(i-1, 3)
        col_idx = mod(i-1, 3)
        roi_data = roi_batch.data[:, :, i]

        # Normalize ROI
        roi_min, roi_max = extrema(roi_data)
        if roi_max > roi_min
            roi_norm = (roi_data .- roi_min) ./ (roi_max - roi_min)
        else
            roi_norm = zeros(size(roi_data))
        end

        r_start = row_idx * roi_batch.roi_size + 1
        r_end = (row_idx + 1) * roi_batch.roi_size
        c_start = col_idx * roi_batch.roi_size + 1
        c_end = (col_idx + 1) * roi_batch.roi_size

        roi_montage[r_start:r_end, c_start:c_end] = roi_norm
    end

    filename = joinpath(output_dir, "detected_rois_montage.png")
    save(filename, Gray.(roi_montage))
    println("  Saved: detected_rois_montage.png ($(n_rois_to_save) ROIs)")
end

# Save statistics to text file
stats_filename = joinpath(output_dir, "blinking_workflow_ideal_stats.txt")
open(stats_filename, "w") do io
    println(io, "="^80)
    println(io, "SMLMBOXER BLINKING WORKFLOW - IDEALCAMERA")
    println(io, "="^80)
    println(io)

    println(io, "SIMULATION PARAMETERS:")
    println(io, "  Pattern: $(pattern.n)-mer, diameter=$(pattern.d * 1000) nm")
    println(io, "  Density: $(sim_params.density) patterns/μm²")
    println(io, "  PSF σ: $(sim_params.σ_psf * 1000) nm")
    println(io, "  Frames: $(sim_params.nframes)")
    println(io, "  Photon rate: $(molecule.γ) Hz")
    println(io, "  k_off: $(k_off) Hz, k_on: $(k_on) Hz")
    println(io, "  Duty cycle: $(round(k_on / (k_on + k_off) * 100, digits=2))%")
    println(io)

    println(io, "GROUND TRUTH:")
    println(io, "  Patterns: $(round(Int, n_patterns))")
    println(io, "  Total emitters: $n_emitters")
    println(io, "  Emitters/frame (avg): $(round(n_emitters / sim_params.nframes, digits=1))")
    println(io, "  Emitters/frame (min): $(minimum([count(e -> e.frame == f, smld_ground_truth.emitters) for f in frames_with_emitters]))")
    println(io, "  Emitters/frame (max): $(maximum([count(e -> e.frame == f, smld_ground_truth.emitters) for f in frames_with_emitters]))")
    println(io)

    println(io, "IMAGES:")
    println(io, "  Stack size: $(size(images))")
    println(io, "  Value range: [$(round(minimum(images), digits=1)), $(round(maximum(images), digits=1))] photons")
    println(io)

    println(io, "DETECTION (PSF-AWARE):")
    println(io, "  PSF sigma: $(sim_params.σ_psf) μm (= $(round(psf_sigma_pixels_calc, digits=2)) pixels)")
    println(io, "  Photon threshold: 50.0")
    println(io, "  DoG sigma_small: $(round(1.0 * psf_sigma_pixels_calc, digits=2)) pixels")
    println(io, "  DoG sigma_large: $(round(2.0 * psf_sigma_pixels_calc, digits=2)) pixels")
    println(io, "  Intensity threshold: $(round(SMLMBoxer.photons_to_dog_threshold(50.0, psf_sigma_pixels_calc), digits=2)) ADU")
    println(io)

    println(io, "RESULTS:")
    println(io, "  Detected ROIs: $n_detected")
    println(io, "  Detection rate: $(round(n_detected / n_emitters * 100, digits=1))%")
    if !isempty(detection_rates)
        println(io, "  Per-frame detection rate (mean): $(round(mean(detection_rates) * 100, digits=1))%")
        println(io, "  Per-frame detection rate (std): $(round(std(detection_rates) * 100, digits=1))%")
        println(io, "  Per-frame detection rate (min): $(round(minimum(detection_rates) * 100, digits=1))%")
        println(io, "  Per-frame detection rate (max): $(round(maximum(detection_rates) * 100, digits=1))%")
    end
    println(io)

    println(io, "PERFORMANCE:")
    println(io, "  Simulation time: $(round(t_sim, digits=2))s")
    println(io, "  Image generation time: $(round(t_gen, digits=2))s")
    println(io, "  Detection time: $(round(t_detect, digits=2))s")
    println(io, "  Detection throughput: $(round(sim_params.nframes / t_detect, digits=1)) frames/sec")
    println(io)

    println(io, "="^80)
    println(io, "Output files in: $(output_dir)")
    println(io, "  - input_frame_XXX.png: Example input images")
    if n_detected > 0
        println(io, "  - detected_rois_montage.png: Example detected ROIs")
    end
    println(io, "  - blinking_workflow_ideal_stats.txt: This file")
    println(io, "="^80)
end

println("  Saved: blinking_workflow_ideal_stats.txt")
println()
println("Output directory: $(output_dir)")
println()
