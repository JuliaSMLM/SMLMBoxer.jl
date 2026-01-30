# SMLMBoxer.jl Examples

Examples demonstrating particle detection workflows from simulation to validation.

## Setup

The examples use their own independent environment with development versions of JuliaSMLM packages.

```bash
cd examples/
julia --project=.
```

Dependencies are already configured to use local dev versions:
- SMLMBoxer (parent directory)
- SMLMData (../../SMLMData)
- SMLMSim (../../SMLMSim)
- MicroscopePSFs (../../MicroscopePSFs)

## Examples Overview

### 🎯 Recommended: Blinking Workflow Examples

**These demonstrate the complete realistic SMLM workflow that mirrors real experiments.**

#### Blinking Workflow - IdealCamera
```bash
julia --project=. blinking_workflow_ideal.jl
```

**The complete production workflow:**
- ✓ Multi-frame blinking simulation (100 frames) with SMLMSim
- ✓ Realistic fluorophore dynamics (k_on, k_off)
- ✓ Pattern-based emitter placement (Nmer2D octamers)
- ✓ **Starting from images + camera** (user reality)
- ✓ PSF-aware detection interface
- ✓ Frame-by-frame validation and statistics

**This shows what you actually do with experimental data.**

#### Blinking Workflow - sCMOS Camera
```bash
julia --project=. blinking_workflow_scmos.jl
```

**Same workflow with sCMOS camera:**
- ✓ Spatially-varying readnoise map (realistic sCMOS)
- ✓ Full camera noise model (QE, Poisson, readnoise, gain, offset)
- ✓ **Variance-weighted DoG filtering** (sCMOS-optimized)
- ✓ Side-by-side comparison with IdealCamera
- ✓ Shows advantage of variance weighting

---

### 📚 Algorithm Validation Examples

**These are simpler examples for understanding the detection algorithm.**

#### Basic Detection (Single Frame)
```bash
julia --project=. basic_detection.jl
```

Shows:
- Algorithm validation on clean/noisy single frames
- Grid pattern emitters (not blinking)
- CPU vs GPU performance comparison
- Updated to PSF-aware interface

#### sCMOS Detection (Single Frame)
```bash
julia --project=. scmos_detection.jl
```

Shows:
- Variance-weighted filtering mechanics
- Spatial noise variation effects
- Low-noise vs high-noise region comparison
- Single frame for clarity

---

## Which Example Should I Run?

**Starting with SMLMBoxer?** → `blinking_workflow_ideal.jl`
**Have sCMOS camera?** → `blinking_workflow_scmos.jl`
**Understanding algorithm?** → `basic_detection.jl` or `scmos_detection.jl`

## Output

Blinking workflow examples print:
- Simulation parameters (density, PSF, blinking dynamics)
- Image generation statistics
- Detection parameters (PSF-aware interface values)
- Per-frame detection statistics
- Overall detection rate and processing speed

Algorithm validation examples print:
- Ground truth emitter count
- Detected spot count
- Detection accuracy
- CPU/GPU performance comparison

## Standard Workflow Pattern

All examples follow this realistic pattern:

1. **Setup Camera** - IdealCamera or SCMOSCamera with calibration
2. **Configure Simulation** - StaticSMLMParams with pattern and molecule
3. **Simulate Blinking** - `simulate(params; pattern, molecule, camera)` → SMLD
4. **Generate Images** - `gen_images(smld, psf; camera_noise=...)` → image stack
5. **Detect Particles** - `getboxes(images, camera; psf_sigma, min_photons)` → ROIBatch
6. **Validate** - Compare detections to ground truth

## PSF-Aware Interface

All examples use the new PSF-aware detection interface:

```julia
# Calculate PSF sigma in pixels
psf_sigma_pixels = psf_sigma_microns / pixel_size_microns

# Detect with clear physical parameters
roi_batch = getboxes(images, camera;
    psf_sigma = psf_sigma_pixels,  # Automatically sets DoG scales
    min_photons = 500.0)            # Physical photon threshold
```

**Benefits:**
- DoG filter automatically scaled to PSF width
- Photon threshold → intensity threshold conversion
- No manual parameter tuning
- Clear physical meaning

## Next Steps

After detection, you typically:
1. **Fit localizations** - Use GaussMLE.jl with the ROIBatch
2. **Filter results** - Remove low-quality fits
3. **Render images** - Use SMLMRender.jl
4. **Analyze structures** - Clustering, DBSCAN, etc.

See SMLMAnalysis.jl for complete workflows including fitting.
