import numpy as np
from skimage import io
from pathlib import Path
from utils import box_utils, calibration_kitti, common_utils, object3d_kitti

def get_image_shape(root_split_path, idx):
    img_file = Path(root_split_path) / 'training/image_2' / ('%s.png' % idx)
    assert img_file.exists()
    return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

def get_calib(root_split_path, idx):
    calib_file = Path(root_split_path) / 'training/calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

def get_label(root_split_path, idx):
    label_file = Path(root_split_path) / 'training/label_2' / ('%s.txt' % idx)
    assert label_file.exists()
    return object3d_kitti.get_objects_from_label(label_file)

def get_lidar(root_split_path, idx):
    lidar_file = Path(root_split_path) / 'training/velodyne' / ('%s.bin' % idx)
    assert lidar_file.exists()
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

def get_fov_flag(pts_rect, img_shape, calib):
    """
    Args:
        pts_rect:
        img_shape:
        calib:
    Returns:
    """
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

def Lidar_to_Max2(points, gt_boxes_lidar):
    """
    Args:
        points: N x 3+C
        gt_boxes_lidar: 7 
    Returns:
        points normalized to max-2 unit square box: N x 3+C
    """
    # shift
    points[:,:3]=points[:,:3]-gt_boxes_lidar[:3]
    # normalize to 2 units 
    points[:,:3]=points[:,:3]/np.max(gt_boxes_lidar[3:6])*2
    # reversely rotate 
    points=rotate_pts_along_z(points, -gt_boxes_lidar[6])
    
    return points

def Max2_to_Lidar(points, gt_boxes_lidar):
    """
    Args:
        points: N x 3+C
        gt_boxes_lidar: 7 
    Returns:
        points denormalized to lidar coordinates
    """
       
    # rotate 
    points=rotate_pts_along_z(points, gt_boxes_lidar[6])
    # denormalize to lidar
    points[:,:3]=points[:,:3]*np.max(gt_boxes_lidar[3:6])/2
    # shift
    points[:,:3]=points[:,:3]+gt_boxes_lidar[:3]
    
    return points
    

def rotate_pts_along_z(points, angle):
    """
    Args:
        points: (N x 3 + C) narray
        angle: angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array(
        [cosa,  sina, 0.0,
         -sina, cosa, 0.0,
         0.0,   0.0,  1.0]).reshape(3, 3)
    points_rot = np.matmul(points[ :, 0:3], rot_matrix)
    points_rot = np.hstack((points_rot, points[ :, 3:].reshape(-1,1)))
    
    return points_rot


def process_1_scan_all(root_path, sample_idx):
    root_split_path=Path(root_path)
    print('The sample_idx: %s' % ( sample_idx))
    info = {}
    info['sample_idx'] = sample_idx
    
    info['image_sample'] = get_image_shape(root_split_path, sample_idx)
    calib = get_calib(root_split_path, sample_idx)

    P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
    R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
    R0_4x4[3, 3] = 1.
    R0_4x4[:3, :3] = calib.R0
    V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
    calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

    info['calib'] = calib_info

    
    obj_list = get_label(root_split_path, sample_idx)
    annotations = {}
    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
    annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
    annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
    annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
    annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
    annotations['score'] = np.array([obj.score for obj in obj_list])

    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(annotations['name'])
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['num_objs'] = num_objects
    loc = annotations['location'][:num_objects]
    dims = annotations['dimensions'][:num_objects]
    rots = annotations['rotation_y'][:num_objects]
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    print(gt_boxes_lidar.shape)
    info['annos'] = annotations

    
    points = get_lidar(root_split_path, sample_idx)
    
    # calib = get_calib(root_split_path, idx)
    # pts_rect = calib.lidar_to_rect(points[:, 0:3])
    # fov_flag = get_fov_flag(pts_rect, info['image']['image_shape'], calib)
    # pts_fov = points[fov_flag]
    
    pts_fov = points
    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
    pts_in_gt = []

    for k in range(num_objects):
        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
        pts_in_gt += [{'pts': pts_fov[flag, 0:3], 'idx':flag}]
    annotations['pts_in_gt'] = pts_in_gt

    return info

def process_1_scan(root_path, sample_idx):
    root_split_path=Path(root_path)
    info = {}    
    calib = get_calib(root_split_path, sample_idx)    
    obj_list = get_label(root_split_path, sample_idx)
    info['name'] = np.array([obj.cls_type for obj in obj_list])    
    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    info['num_objs'] = num_objects
    loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)[:num_objects]
    dims = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])[:num_objects]
    rots = np.array([obj.ry for obj in obj_list])[:num_objects]
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    info['gt_boxes_lidar'] = gt_boxes_lidar    
    points = get_lidar(root_split_path, sample_idx)    
    pts_fov = points
    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
    pts_in_gt = []
    for k in range(num_objects):
        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
        pts_in_gt += [flag]
    info['pts_in_gt'] = pts_in_gt
    return info

# constrained to max2 with the origin ratio
def normalize(points):
    """
    Args:
        points: N x 3+C 
    Returns:
        limit points to max-2 unit square box: N x 3+C
    """
    if points.shape[0] != 0:
        indicator = np.max(np.abs(points[:,:3]))
        if indicator>1:
            points[:,:3] = points[:,:3]/indicator
    return points

# constrained to gt
def normalize_gt(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C
        gt_box_ratio: 3 
    Returns:
        limit points to gt: N x 3+C
    """    
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio/np.max(gt_box_ratio)
        for i in range(3):
            indicator = np.max(np.abs(points[:,i])) / box_boundary_normalized[i]
            if indicator > 1: 
                points[:,i] /= indicator 
    return points

# constrained to gt with the origin ratio
def normalize_max2(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C
        gt_box_ratio: 3 
    Returns:
        limit points to gt with the origin ratio: N x 3+C
    """    
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio/np.max(gt_box_ratio)
        indicators = np.zeros(3)
        for i in range(3):
            indicators[i] = np.max(np.abs(points[:,i]))/box_boundary_normalized[i] if np.max(np.abs(points[:,i])) > box_boundary_normalized[i] else 1.0
        points[:,:3]/=np.max(indicators)  

    return points



# constrained to gt with the origin ratio
def normalize_max2_2(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C 
    Returns:
        limit points to gt with the origin ratio: N x 3+C
        scalling factor
    """
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio/np.max(gt_box_ratio)
        indicators = np.zeros(3)
        for i in range(3):
            indicators[i] = np.max(np.abs(points[:,i]))/box_boundary_normalized[i] if np.max(np.abs(points[:,i])) > box_boundary_normalized[i] else 1.0
        indicator_rat = np.max(indicators)
        points[:,:3]/=indicator_rat
        # points touch the ground (almost)
        points[:,2]=points[:,2]-box_boundary_normalized[2]*(1.0-1.0/indicator_rat)
        return points, indicator_rat
    else:
        return points, 1.