# Simulation
```
python  Simulation_weather.py --path_data ${path_to_kitti}/training/velodyne --save_dir=${path_to_kitti}/training --atm_model ${cor_model} --rate ${sev_level}
```
where `${path_to_kitti}` is the path to kitti data, e.g., `../data/kitti`; 
`${cor_model}` is the model of corruption in `{snow, rain}`; 
`${sev_level}` is the severity level, for `rain` in `{5.0, 15.0, 50.0, 150.0, 500.0}` and `snow` in `{0.5, 1.5, 5.0, 15.0, 50.0}`.

# Results:
After running the above, it will output data files in `${path_to_kitti}/training`.
