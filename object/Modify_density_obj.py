import numpy as np
from os.path import basename, join, exists
from os import makedirs
from point_extract import process_1_scan,get_lidar 
from point_extract import Lidar_to_Max2, Max2_to_Lidar, normalize
from scipy import interpolate
from utils import distortion

''' cutout on object-level pointcloud'''
def cutout(root_path, sample_idx,severity):    
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [(1,20), (2,20), (3,20), (4,20), (5,20)][severity-1]
    idx_add = np.zeros(pointcloud.shape[0], dtype=bool)
    pts_add = np.zeros((1, pointcloud.shape[1]))
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        if sum(idx_pts) > c[0]*c[1]:
            pts_obj = pointcloud[idx_pts]
            # convert to max-2
            pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
            points = pts_obj_max2            
            ## cutout                  
            for _ in range(c[0]):
                i_pts = np.random.choice(points.shape[0],1)
                picked = points[i_pts, :3]
                dist = np.sum((points[:, :3] - picked)**2, axis=1, keepdims=True)
                idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
                points = np.delete(points, idx.squeeze(), axis=0)    
            pts_obj_max2_crp = points    
            # convert to Lidar
            pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
            idx_add = idx_add | np.array(idx_pts)
            pts_add = np.vstack((pts_add, pts_obj_crp))
        else:
            idx_add = idx_add | np.array(idx_pts)
    pointcloud = np.delete(pointcloud, idx_add, axis=0)       
    if pts_add.shape[0]>1: # if pointcloud is indeed processed        
        pointcloud =np.vstack((pointcloud, pts_add[1:, :]))
    return pointcloud

''' locally decrease point density on object-level pointcloud'''
def density_dec(root_path, sample_idx,severity):       
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [(1,30), (2,30), (3,30), (4,30), (5,30)][severity-1]
    idx_add = np.zeros(pointcloud.shape[0], dtype=bool)
    pts_add = np.zeros((1, pointcloud.shape[1]))
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        if sum(idx_pts) > c[0]*int((3/4) * c[1]):
            pts_obj = pointcloud[idx_pts]
            # convert to max-2
            pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
            points = pts_obj_max2            
            ## density decreasing on object                  
            for _ in range(c[0]):
                i_pts = np.random.choice(points.shape[0],1)
                picked = points[i_pts, :3]
                dist = np.sum((points[:, :3] - picked)**2, axis=1, keepdims=True)
                N_near = min(c[1], points.shape[0]-1)
                idx = np.argpartition(dist, N_near, axis=0)[:N_near]                
                idx_2 = np.random.choice(N_near,int((3/4) * c[1]),replace=False)
                idx = idx[idx_2]
                points = np.delete(points, idx.squeeze(), axis=0)    
            pts_obj_max2_crp = points   
            # convert to Lidar
            pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
            idx_add = idx_add | np.array(idx_pts)
            pts_add = np.vstack((pts_add, pts_obj_crp))
        else:
            idx_add = idx_add | np.array(idx_pts)
    pointcloud = np.delete(pointcloud, idx_add, axis=0)       
    if pts_add.shape[0]>1: # if pointcloud is indeed processed        
        pointcloud =np.vstack((pointcloud, pts_add[1:, :]))
    return pointcloud


''' locally increase point density on object-level pointcloud'''
def density_inc(root_path, sample_idx,severity):
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [(1,30), (2,30), (3,30), (4,30), (5,30)][severity-1]
    idx_add = np.zeros(pointcloud.shape[0], dtype=bool)
    pts_add = np.zeros((1, pointcloud.shape[1]))
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        if sum(idx_pts) > c[1]:
            pts_obj = pointcloud[idx_pts]
            # convert to max-2
            pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
            points = pts_obj_max2 
            points_add = np.zeros((1, pointcloud.shape[1]))
            ## density increasing on object 
            N_opr = min(c[0], (pts_obj_max2.shape[0]-1)//c[1])
            for _ in range(N_opr):
                i_pts = np.random.choice(points.shape[0],1)
                picked = points[i_pts, :3]
                dist = np.sum((points[:, :3] - picked)**2, axis=1, keepdims=True)
                idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
                points_add = np.vstack((points_add, interp_3D(points[idx.squeeze()],2)))    
            pts_obj_max2_crp = normalize(np.vstack((points, points_add[1:, :])))   
            # convert to Lidar
            pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
            idx_add = idx_add | np.array(idx_pts)
            pts_add = np.vstack((pts_add, pts_obj_crp))
        
    if pts_add.shape[0]>1: # if pointcloud is indeed processed
        pointcloud = np.delete(pointcloud, idx_add, axis=0)
        pointcloud =np.vstack((pointcloud, pts_add[1:, :]))
    return pointcloud

''' interpolation by Least-Square '''
def interp_3D(points, inp_rate):
    ''' 
    Args: 
        origin points: N x (3+C) 
        inp_rate: float
    Return:
        new points
    '''
    N,_=points.shape
    z_idx = np.argmin( np.max(points[:,:3],axis=0) - np.min(points[:,:3],axis=0))
    if z_idx ==0:
        x_idx, y_idx = 1, 2
    elif z_idx ==1:
        x_idx, y_idx = 2, 0
    else:
        x_idx, y_idx = 0, 1
        
    X_=np.hstack((np.ones((N,1)), points[:,x_idx].reshape(-1,1), points[:,y_idx].reshape(-1,1)))
    Y_=points[:,z_idx].reshape(-1,1)
    
    # Least-Square closed-form solution
    W =np.linalg.inv(X_.T @ X_)@X_.T@Y_
    
    N_new = int(N * inp_rate)
    x_new = np.random.randn(N_new) * np.std(points[:,x_idx]) + np.mean(points[:,x_idx])
    y_new = np.random.randn(N_new) * np.std(points[:,y_idx]) + np.mean(points[:,y_idx])
    r_mean = np.mean(points[:,3])
    
    X_new = np.hstack((np.ones((N_new,1)), x_new.reshape(-1,1), y_new.reshape(-1,1)))
    z_new = X_new @ W.reshape(-1,1)
    
    if z_idx ==0:
        points_new = np.hstack((z_new.reshape(-1,1), x_new.reshape(-1,1), y_new.reshape(-1,1)))
    elif z_idx ==1:
        points_new = np.hstack((y_new.reshape(-1,1), z_new.reshape(-1,1), x_new.reshape(-1,1)))
    else:
        points_new = np.hstack((x_new.reshape(-1,1), y_new.reshape(-1,1), z_new.reshape(-1,1)))
    
    points_new = np.hstack((points_new, np.zeros(N_new).reshape(-1, 1)+r_mean))
    return points_new
    

MAP = {'cutout_obj': cutout,
       'density_dec_obj': density_dec,
       'density_inc_obj': density_inc,
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