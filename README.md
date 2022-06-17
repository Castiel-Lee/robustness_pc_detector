# robustness_pc_detector
We propose the first robustness benchmark of point cloud detectors against common corruption patterns. We first introduce different corruption patterns collected for this benchmark and dataset. Then we propose the evaluation metrics used in our benchmark. Finally, we introduce the subject object detection methods and robustness enhancment methods selected for this benchmark.


This kit of corruption simulation applies to KITTI data. Please download the [KITTI 3D object detection dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the files as the below:

```
data
├── kitti
│   │── ImageSets
│   │── training
│   │   ├──calib & velodyne & label_2 & image_2 
│   │── testing
│   │   ├──calib & velodyne & image_2
```
