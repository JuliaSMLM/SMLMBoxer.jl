```@meta
CurrentModule = SMLMBoxer
```

# API Reference


Images must be finite 2D images or 3D `(rows, cols, frames)` stacks, with each image
dimension at least `boxsize`. Unsupported dimensions and undersized images raise
`ArgumentError` before filtering. A zero-frame stack returns an empty `ROIBatch`
and `BoxesInfo` (CPU, zero batches). NaN/Inf pixels raise `ArgumentError` with the
count and first `(row, col, frame)`, including under `:auto`; integers skip the scan.

`SCMOSCamera` requires every `readnoise`, `gain` and `qe` value to be finite and
strictly positive, including maps. Errors name the field and affected pixel count.
A noise-free readout should use `IdealCamera` with the image in photons.

With an `SCMOSCamera`, the PSF-aware path (`psf_sigma`/`min_photons`) detects in
photons per pixel: `(adu - offset) * gain / qe`, with variance `(readnoise / qe)^2`
in photons². Offset, gain and QE may each be scalar or maps; offset must be finite
when this conversion is used. The threshold uses effective gain 1 and the
variance-weighted reference response to `min_photons` photons (uniform variance).
Conversion runs on an owned batch copy on its device. ROI pixels are still cut
from the caller's ADU image and retain the supplied camera and coordinate convention.
The direct interface (`sigma_small`, `sigma_large`, `minval`) filters the image as
given with `(readnoise / gain)^2` in ADU²; `minval` stays in those image units.


Use the `BoxerConfig` keyword constructor. It converts values to field types before
validation, including `boxsize=9.0` and `Int32` numeric fields. `min_photons` defaults to `nothing` and
requires `psf_sigma`; when `psf_sigma` is given and `min_photons` omitted, the config
stores **500.0**. Both must be finite and > 0. PSF-aware detection requires a camera
whose pixels are all square and uniformly sized (relative tolerance 1e-4).
Other cameras can use `sigma_small`/`sigma_large` in pixels with the direct interface.

Direct sigmas must be finite with `0 < sigma_small < sigma_large`, and `minval`
must be finite. `boxsize >= 1`, `overlap` must be finite and >= 0,
`backend` must be `:cpu`, `:gpu` or `:auto`, and both `auto_timeout` and `gpu_timeout`
must be >= 0 (Inf allowed). `on_wait` is a function or `nothing`.
Invalid parameters raise `ArgumentError` naming the field and value.


`min_photons` calibrates the filtered peak of a noise-free, pixel-integrated Gaussian
centered on a pixel. At exactly that photon count it just reaches the threshold.
See [`BoxerConfig`](@ref) for subpixel-position effects. Each camera path uses its
own filter kernels, with uniform variance as the sCMOS reference.

Camera and image dimensions must match. Odd and even box sizes are supported;
after edge clamping, only the brightest filtered detection per corner and frame remains.

```@index
Pages = ["api.md"]
```

```@autodocs
Modules = [SMLMBoxer]
```
