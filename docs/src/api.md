```@meta
CurrentModule = SMLMBoxer
```

# API Reference

`min_photons` calibrates the filtered peak of a noise-free, pixel-integrated Gaussian
centered on a pixel. At exactly that photon count it just reaches the threshold;
at a pixel corner it needs about 1.1× more photons. Each camera path uses its own
filter kernels, with uniform variance as the sCMOS reference.

Camera and image dimensions must match. Odd and even box sizes are supported;
after edge clamping, only the brightest filtered detection per corner and frame remains.

```@index
Pages = ["api.md"]
```

```@autodocs
Modules = [SMLMBoxer]
```
