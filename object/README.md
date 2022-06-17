To get augmented data corrupted by Noise_obj, run:

# Simulation

## Noise_obj
```
python Simulation_noise_obj.py -d ${path_to_kitti} -n ${cor_model} -r ${sev_level}
```
where `${path_to_kitti}` is the path to kitti data, e.g., `../data/kitti`; `${cor_model}` is the model of corruption in `{uniform_obj, gaussian_obj, impulse_obj, upsampling_obj}`; `${sev_level}` is the severity level from 1 to 5.

## Density_obj
```
python Simulation_density_obj.py -d ${path_to_kitti} -n ${cor_model} -r ${sev_level}
```
where `${cor_model}` is in `{cutout_obj, density_dec_obj, density_inc_obj}`.

## Transformation
```
python Simulation_transformation_obj.py -d ${path_to_kitti} -n ${cor_model} -r ${sev_level};
```
where `${cor_model}` is in `{shear_obj, scale_obj, translocate_obj, rotation_obj, distortion_ffd_obj}`.

# Results:
Output data files in `${path_to_kitti}/training`. Note that, besides LiDAR scanning, `scale_obj, translocate_obj, rotation_obj` will generate modified label files.

