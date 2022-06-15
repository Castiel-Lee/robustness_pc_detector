import numpy as np
from os.path import basename, join, exists
from os import makedirs
from point_extract import process_1_scan,get_lidar 
from point_extract import Lidar_to_Max2, Max2_to_Lidar, normalize
from utils import distortion

''' adding uniform noise on object-level pointcloud'''
def uniform_noise(root_path, sample_idx,severity):    
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity-1]   
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## uniform noise
        noise = np.random.uniform(-c,c,(sum(idx_pts), 3))
        new_pc = (pts_obj_max2[:,:3] + noise).astype('float32')        
        pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:,3].reshape(-1,1)))        
        pts_obj_max2_crp = normalize(pts_obj_max2_crp)    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp                
    return pointcloud

''' adding gaussian noise on object-level pointcloud'''
def gaussian_noise(root_path, sample_idx,severity):    
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [0.02, 0.03, 0.04, 0.05, 0.06][severity-1]   
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## gaussian noise
        noise =  np.random.normal(size=(sum(idx_pts), 3)) * c
        new_pc = (pts_obj_max2[:,:3] + noise).astype('float32')        
        pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:,3].reshape(-1,1)))        
        pts_obj_max2_crp = normalize(pts_obj_max2_crp)    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp                
    return pointcloud

''' adding impulse noise on object-level pointcloud'''
def impulse_noise(root_path, sample_idx,severity):    
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [30, 25, 20, 15, 10][severity-1]   
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## impulse noise 
        N_pts = sum(idx_pts)
        index = np.random.choice(N_pts, N_pts//c, replace=False)
        pts_obj_max2[index, :3] += np.random.choice([-1,1], size=(N_pts//c,3)) * 0.1            
        pts_obj_max2_crp = normalize(pts_obj_max2)    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp                
    return pointcloud

''' adding impulse noise on object-level pointcloud'''
def upsampling(root_path, sample_idx,severity):    
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [5, 4, 3, 2, 1][severity-1] 
    add_points_all = np.zeros((1, 4))
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## upsampling
        N_pts = sum(idx_pts)
        index = np.random.choice(N_pts, N_pts//c, replace=False)
        add = pts_obj_max2[index, :3] + np.random.uniform(-0.05,0.05,(N_pts//c,3))  
        add = np.hstack((add, np.zeros(add.shape[0]).reshape(-1,1)))
        # fill up the reflectivity for new points, by the nearest point
        for i_add in range(add.shape[0]):
            dist = np.sum((pts_obj_max2[:, :3] - add[i_add, :3])**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, 0, axis=0)[0]
            add[i_add, 3] = pts_obj_max2[idx, 3]            
        pts_obj_max2_crp = normalize(np.vstack((pts_obj_max2, add)))    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp[:N_pts, :]
        add_points_all = np.vstack((add_points_all, pts_obj_crp[N_pts:, :]))
    if add_points_all.shape[0] > 1:
        pointcloud = np.vstack((pointcloud, add_points_all[1:, :]))
    return pointcloud

MAP = {'uniform_obj': uniform_noise,
       'gaussian_obj': gaussian_noise,
       'impulse_obj': impulse_noise,
       'upsampling_obj': upsampling,
       'original': None,
}

def opr_pc(root_path, sample_idx, corruption, severity, saving=False):
    pc = get_lidar(root_path, sample_idx)        
    pc_crp = MAP[corruption](root_path, sample_idx, severity)
    pc_crp =pc_crp.reshape(-1,)
    pc_crp =pc_crp.astype(np.float32)
    if saving==True:
        dir_save = root_path + '/training/velodyne_' + corruption + '_' + str(severity)
        if not exists(dir_save):
            makedirs(dir_save)
        pc_crp.tofile(join(dir_save,basename(sample_idx+'.bin')))
    return pc, pc_crp