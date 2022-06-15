import numpy as np
from os.path import basename, join, exists
from os import makedirs
from pathlib import Path
from point_extract import process_1_scan,get_lidar 
from point_extract import Lidar_to_Max2, Max2_to_Lidar
from point_extract import normalize_gt as normalize

from utils import distortion
''' Shear x and y, except z'''
def shear(root_path, sample_idx,severity):
    
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
#     print(info['name'],info['num_objs'])
    N, _ = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]
       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## shear        
        a = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        b = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        d = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        e = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        matrix = np.array([1, a, 0,
                           b, 1, 0,
                           d, e, 1]).reshape(3,3)      
        new_pc = np.matmul(pts_obj_max2[:,:3], matrix).astype('float32')
        
        pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:,3].reshape(-1,1)))
        pts_obj_max2_crp = normalize(pts_obj_max2_crp, info['gt_boxes_lidar'][i][3:6])    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp
                
    return pointcloud

def ffd_distortion(root_path, sample_idx,severity):
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [0.1,0.2,0.3,0.4,0.5][severity-1]
        
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## ffd        
        new_pc = distortion.distortion(pts_obj_max2[:,:3],severity=c)      
        # convert to lidar
        pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:,3].reshape(-1,1)))
        pts_obj_max2_crp = normalize(pts_obj_max2_crp, info['gt_boxes_lidar'][i][3:6])    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp
        
    return pointcloud

def rbf_distortion(root_path, sample_idx,severity):
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][severity-1]
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## RBF       
        new_pc = distortion.distortion_2(pts_obj_max2[:,:3],severity=c,func='multi_quadratic_biharmonic_spline')
        # convert to lidar
        pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:,3].reshape(-1,1)))
        pts_obj_max2_crp = normalize(pts_obj_max2_crp, info['gt_boxes_lidar'][i][3:6])    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp
        
    return pointcloud

def rbf_distortion_inv(root_path, sample_idx,severity):
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][severity-1]
    for i in range(info['num_objs']):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## RBF_inv       
        new_pc = distortion.distortion_2(pts_obj_max2[:,:3],severity=c,func='inv_multi_quadratic_biharmonic_spline')
        # convert to lidar
        pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:,3].reshape(-1,1)))
        pts_obj_max2_crp = normalize(pts_obj_max2_crp, info['gt_boxes_lidar'][i][3:6])    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp
        
    return pointcloud

def rotation(root_path, sample_idx,severity):
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [1, 3, 5, 7, 9][severity-1]
    betas = []    
    for i in range(info['num_objs']):
        beta = np.random.uniform(c-1,c+1) * np.random.choice([-1,1]) * np.pi / 180.
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]       
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])
        ## rotation       
        matrix_roration_z = np.array([[np.cos(beta),np.sin(beta),0],[-np.sin(beta),np.cos(beta),0],[0,0,1]])
        pts_rotated = np.matmul(pts_obj_max2[:,:3], matrix_roration_z)
        pts_obj_max2_crp = np.hstack((pts_rotated, pts_obj_max2[:,3].reshape(-1,1)))
        # convert to lidar
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        # cover the old points
        pointcloud[idx_pts]=pts_obj_crp 
        betas.append(beta)
    generate_label_rotation(root_path, sample_idx, info['num_objs'], betas, severity)
    return pointcloud

def generate_label_rotation(root_path, sample_idx, N_obj, betas, severity):
    label_file_read = Path(root_path) / 'training/label_2' / ('%s.txt' % sample_idx)
    lines = open(label_file_read, 'r').readlines()
    label_objects = [line.strip().split(' ') for line in lines]
    
    for i in range(N_obj):
        angle_modified = float(label_objects[i][14])-betas[i]
        # constraint to -pi~pi
        if abs(angle_modified)>np.pi:
            angle_modified = angle_modified - angle_modified/abs(angle_modified) * 2*np.pi
        label_objects[i][14] = str(angle_modified)
    
    lines = [' '.join(line) for line in label_objects]
    lines = [line+' \n' for line in lines]
    dir_save = Path(root_path) / ('training/label_2_rotation_obj_'+str(severity))
    if not exists(dir_save):
            makedirs(dir_save)
    label_file_write = dir_save / ('%s.txt' % sample_idx)
    open(label_file_write, 'w').writelines(lines)

'''
Scale the point cloud
'''
def scale(root_path, sample_idx, severity):
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [0.04, 0.08, 0.12, 0.16, 0.20][severity-1]
    xs_list,ys_list,zs_list=[],[],[]
    N_obj=info['num_objs']
    for i in range(N_obj):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]        
        # convert to max-2
        pts_obj_max2 = Lidar_to_Max2(pts_obj, info['gt_boxes_lidar'][i])       
        ## scale on two randomly selected directions     
        xs, ys, zs = 1.0, 1.0, 1.0
        r = np.random.randint(0,3)
        t = np.random.choice([-1,1])
        if r == 0:
            xs += c * t
        elif r == 1:
            ys += c * t
        else:
            zs += c * t
        matrix = np.array([[xs,0,0,0],[0,ys,0,0],[0,0,zs,0],[0,0,0,1]])
        pts_obj_max2_crp = np.matmul(pts_obj_max2, matrix)
        pts_obj_max2_crp[:,2] += (zs-1) * info['gt_boxes_lidar'][i][5]/np.max(info['gt_boxes_lidar'][i][3:6])
        xs_list.append(xs)
        ys_list.append(ys)
        zs_list.append(zs)
        # convert to Lidar
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, info['gt_boxes_lidar'][i])
        pointcloud[idx_pts]=pts_obj_crp
    generate_label_scale(root_path, sample_idx, N_obj, xs_list, ys_list, zs_list, severity)   
    return pointcloud

def generate_label_scale(root_path, sample_idx, N_obj, xs_list, ys_list, zs_list, severity):
    label_file_read = Path(root_path) / 'training/label_2' / ('%s.txt' % sample_idx)
    lines = open(label_file_read, 'r').readlines()
    label_objects = [line.strip().split(' ') for line in lines]
    
    for i in range(N_obj):
        label_objects[i][10] = str(round(float(label_objects[i][10])*xs_list[i], 2)) # x or l
        label_objects[i][9] = str(round(float(label_objects[i][9])*ys_list[i], 2)) # y or w
        label_objects[i][8] = str(round(float(label_objects[i][8])*zs_list[i], 2)) # z or h
    
    lines = [' '.join(line) for line in label_objects]
    lines = [line+' \n' for line in lines]
    dir_save = Path(root_path) / ('training/label_2_scale_obj_'+str(severity))
    if not exists(dir_save):
            makedirs(dir_save)
    label_file_write = dir_save / ('%s.txt' % sample_idx)
    open(label_file_write, 'w').writelines(lines)
    
    
'''
Translocate the point cloud
'''
def translocate(root_path, sample_idx, severity):
    pointcloud = get_lidar(root_path, sample_idx)
    info = process_1_scan(root_path, sample_idx)
    N, _ = pointcloud.shape
    c = [0.1, 0.3, 0.5, 0.7, 0.9][severity-1]
    x_list,y_list=[],[]
    N_obj=info['num_objs']
    for i in range(N_obj):
        idx_pts = info['pts_in_gt'][i]
        pts_obj = pointcloud[idx_pts]             
        ## Translocate on x&y directions     
        x_trl = np.random.uniform(c-0.1,c+0.1) * np.random.choice([-1,1])
        y_trl = np.random.uniform(c-0.1,c+0.1) * np.random.choice([-1,1])
        pts_obj[:,0] += x_trl
        pts_obj[:,1] += y_trl
        x_list.append(x_trl)
        y_list.append(y_trl)
        pointcloud[idx_pts]=pts_obj
    generate_label_trl(root_path, sample_idx, N_obj, x_list, y_list, severity)   
    return pointcloud

def generate_label_trl(root_path, sample_idx, N_obj, x_list, y_list, severity):
    label_file_read = Path(root_path) / 'training/label_2' / ('%s.txt' % sample_idx)
    lines = open(label_file_read, 'r').readlines()
    label_objects = [line.strip().split(' ') for line in lines]
    
    for i in range(N_obj):
        label_objects[i][13] = str(round(float(label_objects[i][13]) + x_list[i], 2)) # x or z_c
        label_objects[i][11] = str(round(float(label_objects[i][11]) - y_list[i], 2)) # y or y_c
    
    lines = [' '.join(line) for line in label_objects]
    lines = [line+' \n' for line in lines]
    dir_save = Path(root_path) / ('training/label_2_translocate_obj_'+str(severity))
    if not exists(dir_save):
            makedirs(dir_save)
    label_file_write = dir_save / ('%s.txt' % sample_idx)
    open(label_file_write, 'w').writelines(lines)
    
    
MAP = {'shear_obj': shear,       
       'scale_obj': scale, 
       'translocate_obj': translocate,
       'rotation_obj': rotation,  
       
       'distortion_ffd_obj': ffd_distortion,
       'distortion_rbf_obj': rbf_distortion,
       'distortion_rbf_inv_obj': rbf_distortion_inv,      
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

# import open3d 

# def plot_pc(pc_data, draw_origin=True, draw_bboxes=False, bboxes_corners=None):
#     points = pc_data[:, :3].reshape(-1, 3)
#     vis = open3d.visualization.Visualizer()
#     vis.create_window()
#     vis.get_render_option().point_size = 1.0
#     vis.get_render_option().background_color = np.zeros(3)
#     # draw origin
#     if draw_origin:
#         axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
#         vis.add_geometry(axis_pcd)
#     if draw_bboxes:
#         for i in range(bboxes_corners.shape[0]):
#             bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
#             colors = [[0, 1, 0] for _ in range(len(bbox_lines))]  #green
#             bbox = open3d.geometry.LineSet()
#             bbox.lines  = open3d.utility.Vector2iVector(bbox_lines)
#             bbox.colors = open3d.utility.Vector3dVector(colors)
#             bbox.points = open3d.utility.Vector3dVector(bboxes_corners[i])
#             vis.add_geometry(bbox)
    
#     pts = open3d.geometry.PointCloud()
#     pts.points = open3d.utility.Vector3dVector(points[:, :3])
#     vis.add_geometry(pts)
#     pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
#     vis.run()
    # vis.destroy_window()