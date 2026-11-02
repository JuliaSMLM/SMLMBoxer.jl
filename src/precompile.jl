# Cover both filtering paths and calling conventions without creating a CUDA context.
@setup_workload begin
    image = zeros(Float32, 32, 32, 2)
    image[16, 16, :] .= 2000f0
    ideal = IdealCamera(1:32, 1:32, 0.1f0)
    scmos = SCMOSCamera(32, 32, 0.1f0, 2f0)
    config = BoxerConfig(sigma_small=1.3, sigma_large=2.6, minval=1.0, backend=:cpu)

    @compile_workload begin
        getboxes(image; sigma_small=1.3, sigma_large=2.6, minval=1.0, backend=:cpu)
        getboxes(image, ideal; psf_sigma=0.135, min_photons=50.0, backend=:cpu)
        getboxes(image, scmos; psf_sigma=0.135, min_photons=50.0, backend=:cpu)
        getboxes(image, nothing, config)
    end
end
