# robustness_pc_detector
We propose the first robustness benchmark of point cloud detectors against common corruption patterns. We first introduce different corruption patterns collected for this benchmark and dataset. Then we propose the evaluation metrics used in our benchmark. Finally, we introduce the subject object detection methods and robustness enhancment methods selected for this benchmark.

# Installation
All the codes are tested (not only) on Ubuntu 20.04 with Python 3.8 tools: 
* `PyGeM` and `PyMieScatt`
* `Scipy`, and other basic tools, like `numpy`, `os`, `argparse`, `glob`, etc.
* As for the weather simulation, please refer to [LISA](https://github.com/velatkilic/LISA) and [Lidar_fog_simulation](https://github.com/MartinHahner/LiDAR_fog_sim).

Most tools can be installed by `pip install ${tool}`.

# Data Preparation
This kit of corruption simulation applies to KITTI data. Please download the [KITTI 3D object detection dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the files as the below:

```
data
├── kitti
│   │── ImageSets
│   │── training
│   │   ├──calib & velodyne & label_2 & image_2 
│   │── testing (optional)
│   │   ├──calib & velodyne & image_2
```
More details of the implementation are as in `object` and `scene` folder. 

# Corruption Simulation

We formulate `25 corruptions` covering 2 affecting ranges $\times$ 4 corruption categories, i.e., `{object, scene}` $\times$ `{weather, noise, density, transformation}`. 

For the implementation of `25 corruptions`, corresponding `README` files under the `object` and `scene` folders give the details on the Python commands to generate simulated data.

We show some corruption examples based on the KITTI LiDAR example with ID = `000008` as the bellow:

### Snow
![snow example](https://github.com/Castiel-Lee/robustness_pc_detector/blob/main/image/snow.png)

### Scene-level uniform noise
![uniform_rad example](https://github.com/Castiel-Lee/robustness_pc_detector/blob/main/image/uniform_rad.png)

# Acknowledgement
Some parts of the code implement are learned from the official released codes of the below methods:
* Snow and Rain simulation: [LISA](https://github.com/velatkilic/LISA)
* Fog simulation: [Lidar_fog_simulation](https://github.com/MartinHahner/LiDAR_fog_sim)
* Object-level corruptions: [ModelNet40-C](https://github.com/jiachens/ModelNet40-C/tree/master/data)
* Extraction of object-level points: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

We would like to thank for their proposed methods and the official implementation.
