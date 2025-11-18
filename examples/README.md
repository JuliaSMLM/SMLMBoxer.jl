# SMLMBoxer.jl Examples

Examples demonstrating the complete workflow from simulation to detection.

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

## Running Examples

### Basic Detection
Demonstrates simple workflow with IdealCamera:
```bash
julia --project=. basic_detection.jl
```

Shows:
- Generating synthetic SMLM data with SMLMSim
- Detecting spots with getboxes()
- Comparing found boxes to ground truth
- Detection accuracy metrics

### sCMOS Detection
Demonstrates variance-weighted detection with sCMOS camera:
```bash
julia --project=. scmos_detection.jl
```

Shows:
- Creating sCMOS camera with spatially-varying noise
- Generating realistic sCMOS camera images
- Variance-weighted DoG filtering
- Improved detection in high-noise regions

## Output

Each example prints:
- Simulation parameters
- Number of ground truth emitters
- Number of detected boxes
- Detection accuracy
- Processing time (CPU/GPU if available)

## Workflow

All examples follow this pattern:

1. **Create Emitters** - Define emitter positions and photon counts
2. **Create Camera** - IdealCamera or SCMOSCamera with calibration
3. **Simulate Images** - Use SMLMSim.gen_images() with PSF model
4. **Detect Boxes** - Use SMLMBoxer.getboxes() → ROIBatch
5. **Validate** - Compare detected positions to ground truth
