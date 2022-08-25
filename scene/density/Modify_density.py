import numpy as np
from numpy import random
from os.path import basename, join, exists
from os import makedirs

### PC Density Modification ###
''' Cutout several part in the point cloud with reflectivities '''
def cutout(pointcloud, severity):
    N, _ = pointcloud.shape
    c = [(N//200,20), (N//150,20), (N//100,20), (N//80,20), (N//60,20)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i,:3]
        dist = np.sum((pointcloud[:,:3] - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    return pointcloud

''' Density-based locally upsampling on the point cloud '''
def density_inc(pointcloud, severity):
    N, _ = pointcloud.shape
    c = [(N//2000,100), (N//1500,100), (N//1000,100), (N//800,100), (N//600,100)][severity-1]
    points_add = np.zeros((1, pointcloud.shape[1]))
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        points_add = np.vstack((points_add, interp_3D(pointcloud[idx.squeeze()], 1)))        
    if points_add.shape[0]>1: # if pointcloud is indeed processed
        pointcloud =np.vstack((pointcloud, points_add[1:, :]))
    return pointcloud

# quadratic-polynomial fitting
def interp_3D(points, inp_rate):
    ''' 
    Args: 
        origin points: N x (3+C) 
        inp_rate: float
    Return:
        new points
    '''
    N,_=points.shape
    # select dimension with lowest variance as the target
    z_idx = np.argmin( np.max(points[:,:3],axis=0) - np.min(points[:,:3],axis=0))
    if z_idx ==0:
        x_idx, y_idx = 1, 2
    elif z_idx ==1:
        x_idx, y_idx = 2, 0
    else:
        x_idx, y_idx = 0, 1
    
    def poly_2D(x, y):
        ''' 
        Input: 
            N x 2: (x, y)
        Output:
            N x 5: (x, y, x^2, y^2, xy)        
        '''
        return np.hstack((x.reshape(-1,1), y.reshape(-1,1),   # x, y
                          (x**2).reshape(-1,1), (y**2).reshape(-1,1),   # x^2, y^2 
                          (x*y).reshape(-1,1)))   # xy 
        
    X_=np.hstack((np.ones((N,1)), poly_2D(points[:,x_idx].reshape(-1,1), points[:,y_idx].reshape(-1,1))))
    Y_=points[:,z_idx].reshape(-1,1)
    W =np.linalg.inv(X_.T @ X_)@X_.T@Y_
    
    N_new = int(N * inp_rate)
    x_new = np.random.randn(N_new) * np.std(points[:,x_idx]) + np.mean(points[:,x_idx])
    y_new = np.random.randn(N_new) * np.std(points[:,y_idx]) + np.mean(points[:,y_idx])
    
    X_new = np.hstack((np.ones((N_new,1)), poly_2D(x_new.reshape(-1,1), y_new.reshape(-1,1))))
    z_new = X_new @ W.reshape(-1,1)
    
    if z_idx ==0:
        points_new = np.hstack((z_new.reshape(-1,1), x_new.reshape(-1,1), y_new.reshape(-1,1)))
    elif z_idx ==1:
        points_new = np.hstack((y_new.reshape(-1,1), z_new.reshape(-1,1), x_new.reshape(-1,1)))
    else:
        points_new = np.hstack((x_new.reshape(-1,1), y_new.reshape(-1,1), z_new.reshape(-1,1)))
    
    # fill up the reflectivity of new points with that of the nearest point
    r_new = np.zeros(N_new).reshape(-1, 1)
    for i in range(N_new):
        dist = np.sum((points[:,:3] - points_new[i,:3])**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, 0, axis=0)[0] 
        r_new[i] = points[idx.squeeze(),3]
    points_new = np.hstack((points_new, r_new))
    
    return points_new
'''
Density-based sampling the point cloud (delete 75% points)
'''
def density_dec(pointcloud, severity):
    N, _ = pointcloud.shape
    c = [(N//300,100), (N//250,100), (N//200,100), (N//150,100), (N//100,100)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i,:3]
        dist = np.sum((pointcloud[:,:3] - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # de
        idx_2 = np.random.choice(c[1],int((4/5) * c[1]),replace=False)
        idx = idx[idx_2]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        # pointcloud[idx.squeeze()] = 0
    # print(pointcloud.shape)
    return pointcloud

'''
Sensor-based beam missing, globally (delete beams)
'''
def beam_del(pointcloud, severity):
    N, _ = pointcloud.shape
    c = [N//100, N//30, N//10, N//5, N//3][severity-1]
    idx_del = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx_del, axis=0)
    
    return pointcloud

    
'''
Sensor-based layer missing, globally (delete layers)
'''
def layer_del(pointcloud, severity):
    N, _ = pointcloud.shape
    c = [3, 7, 11, 15, 19][severity-1]
    N_del = int(c + np.random.choice(3,1) - 1)
    idx_layer = list(np.random.choice(64, N_del, replace=False))
    
    pointcloud_sph=car2sph_pc(pointcloud)
    pointcloud_sph[:,1:3]=pointcloud_sph[:,1:3]/np.pi*180
    bins=get_64bins(pointcloud_sph[:,2])
    
    idx_del = np.zeros(N, dtype=bool)
    for i in idx_layer:
        temp_idx= (bins[i][0]<pointcloud_sph[:,2])&(bins[i][1]>pointcloud_sph[:,2])
        idx_del = idx_del|temp_idx
    pointcloud=pointcloud[~idx_del]
    return pointcloud

def car2sph_pc(pointcloud):
    '''
    args:
        points: N x (3 + c) : x, y, and z
    return:
        points: N x (3 + c) : r, phi, and theta
    '''
    r_sph = np.sqrt(pointcloud[:,0]**2 + pointcloud[:,1]**2 + pointcloud[:,2]**2)
    phi = np.arctan2(pointcloud[:,1],pointcloud[:,0])
    the = np.arccos(pointcloud[:,2]/r_sph)
    return np.hstack((r_sph.reshape(-1,1), phi.reshape(-1,1), the.reshape(-1,1), pointcloud[:,3].reshape(-1,1)))

def sph2car_pc(pointcloud):
    '''
    args:
        points: N x (3 + c) : r, phi, and theta
    return:
        points: N x (3 + c) : x, y, and z
    '''
    x = pointcloud[:,0]*np.sin(pointcloud[:,2])*np.cos(pointcloud[:,1])
    y = pointcloud[:,0]*np.sin(pointcloud[:,2])*np.sin(pointcloud[:,1])
    z = pointcloud[:,0]*np.cos(pointcloud[:,2])
    return np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1), pointcloud[:,3].reshape(-1,1)))

def get_64bins(data):    
    ## filter first
    N_b=1000
    array_den, array_bin = np.histogram(data, bins=N_b, density=True)
    bin_head = array_bin[:-1]
    bin_end = array_bin[1:]
    bin_step = np.sum(bin_head-bin_end)/N_b
    thr_bin = 1/N_b/4 # The 1/N_b is the average density  
    bin_head_filtered = bin_head[array_den>thr_bin]
    bin_end_filtered = bin_end[array_den>thr_bin]
    range_filtered = [np.min(bin_head_filtered), np.max(bin_end_filtered)]
    #filtered points
    data = data[(range_filtered[0]<data) & (data<range_filtered[1])]
    
    ##get bins
    arr_bin = np.linspace(np.min(data), np.max(data), 65)
    bins = []
    for i in range(64):
        bins.append([arr_bin[i], arr_bin[i+1]])    
    return bins 


MAP = {'cutout': cutout,
       'density_dec': density_dec,
       'density_inc': density_inc,
       'layer_del': layer_del,
       'beam_del': beam_del,
       'original': None,
}

def opr_pc(path_pc, corruption, severity, dir_save=None, saving=False):
    pc = np.fromfile(path_pc,dtype=np.float32,count=-1).reshape([-1,4])        
    pc_crp = MAP[corruption](pc, severity)
    pc_crp =pc_crp.reshape(-1,)
    pc_crp =pc_crp.astype(np.float32)
    if saving==True:
        if not exists(dir_save):
            makedirs(dir_save)
        pc_crp.tofile(join(dir_save,basename(path_pc)))
    return pc
