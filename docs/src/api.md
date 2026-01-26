# API Reference

```@index
Pages = ["api.md"]
```

## Exported Functions

### Main Interface

```@docs
getboxes
```

## Re-exported from SMLMData

SMLMBoxer re-exports key types from SMLMData.jl for convenience:

- `ROIBatch`: Container for multiple ROIs with coordinate tracking
- `SingleROI`: Individual ROI with data and position information

For detailed documentation of these types, see the [SMLMData.jl documentation](https://JuliaSMLM.github.io/SMLMData.jl).

## Internal Types

These types are used internally but may be useful for understanding the implementation:

```@autodocs
Modules = [SMLMBoxer]
Order = [:type]
```

## Internal Functions

These functions are not exported but are documented for developers and advanced users.

### Filtering Functions

```@docs
SMLMBoxer.dog_filter
SMLMBoxer.dog_filter_variance_weighted
SMLMBoxer.convolve
SMLMBoxer.convolve_variance_weighted
SMLMBoxer.gaussian_2d
SMLMBoxer.dog_kernel
```

### Detection Functions

```@docs
SMLMBoxer.findlocalmax
SMLMBoxer.genlocalmaximage
```

### Utility Functions

```@docs
SMLMBoxer.get_pixel_size
SMLMBoxer.photons_to_dog_threshold
SMLMBoxer.get_variance_map
SMLMBoxer.extract_camera_roi
SMLMBoxer.reshape_for_flux
```

## Implementation Details

### Difference of Gaussians Filtering

The DoG filter is computed as:

```math
\text{DoG} = G_{\sigma_{\text{small}}} - G_{\sigma_{\text{large}}}
```

where ``G_\sigma`` is a 2D Gaussian kernel with standard deviation ``\sigma``. The filter enhances blob-like features of size matching ``\sigma_{\text{small}}`` while suppressing larger-scale background variations.

### Variance-Weighted Filtering

For sCMOS cameras with per-pixel readnoise maps, the filtered value at pixel ``(i,j)`` is:

```math
F_{i,j} = \frac{\sum_{k,l} w(i-k, j-l) \cdot I_{k,l} / \text{Var}_{k,l}}{\sum_{k,l} w(i-k, j-l) / \text{Var}_{k,l}}
```

where:
- ``w(x,y) = \exp(-\frac{x^2+y^2}{2\sigma^2})`` is the Gaussian kernel weight
- ``I_{k,l}`` is the image intensity at pixel ``(k,l)``
- ``\text{Var}_{k,l} = \text{readnoise}_{k,l}^2`` is the variance at pixel ``(k,l)``

This implements optimal inverse variance weighting for detection in spatially-varying noise.

### PSF-Aware Threshold Conversion

When using the PSF-aware interface, the photon threshold is converted to a DoG intensity threshold:

```math
\begin{aligned}
I_{\text{peak}} &= \frac{N_{\text{photons}}}{2\pi \sigma_{\text{psf}}^2} \\
\sigma_{\text{eff}} &= \sqrt{\sigma_{\text{psf}}^2 + \sigma_{\text{small}}^2} \\
I_{\text{filtered}} &= \frac{N_{\text{photons}}}{2\pi \sigma_{\text{eff}}^2} \\
\text{minval} &\approx 0.65 \times I_{\text{filtered}}
\end{aligned}
```

where the 0.65 factor accounts for the reduction in peak intensity from the DoG operation (empirically determined for ``\sigma_{\text{large}} = 2\sigma_{\text{small}}``).

### GPU Acceleration

SMLMBoxer uses multiple GPU acceleration strategies:

1. **Standard DoG Filtering**: NNlib.conv with cuDNN backend for convolution
2. **Variance-Weighted Filtering**: KernelAbstractions.jl for device-agnostic kernels
3. **Automatic Memory Management**: Batched processing for large datasets exceeding GPU memory

The `use_gpu` parameter controls GPU usage, with automatic fallback to CPU if CUDA is unavailable.

### Coordinate System

ROIBatch uses the SMLMData.jl coordinate conventions:
- `x_corners`: Column positions (horizontal axis)
- `y_corners`: Row positions (vertical axis)
- Corner positions are the top-left pixel of each ROI
- Camera pixel edges define the physical coordinate system
