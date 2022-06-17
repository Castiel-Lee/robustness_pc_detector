# Simulation
```
python Simulation_density.py --path_data ${path_to_kitti}/training/velodyne --save_dir=${path_to_kitti}/training --noise_model ${cor_model} -r ${sev_level}
```
where `${path_to_kitti}` is the path to kitti data, e.g., `../data/kitti`; 
`${cor_model}` is the model of corruption in `{cutout, density_dec, density_inc, layer_del, beam_del}`; 
`${sev_level}` is the severity level from 1 to 5.

# Results:
After running the above, it will output data files in `${path_to_kitti}/training`.
