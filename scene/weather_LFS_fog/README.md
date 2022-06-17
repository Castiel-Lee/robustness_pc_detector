# Simulation
```
python Simulation_fog.py -d ${path_to_kitti}/training/velodyne -s ${path_to_kitti}/training -n 10 -r ${sev_level}
```
where `${path_to_kitti}` is the path to kitti data, e.g., `../data/kitti`, and 
`${sev_level}` is the severity level in `{0.005, 0.01, 0.02, 0.05, 0.1}`.

# Results:
After running the above, it will output data files in `${path_to_kitti}/training`.
