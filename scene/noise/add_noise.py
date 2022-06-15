import numpy as np
from numpy import random
from os.path import basename, join, exists
from os import makedirs

### tools ##
''' Fill vacant intesity of new pc if new points exist， by KNN-distance clustering '''
def fill_intensity(pc, pc_cor_xyz, n_b=5):
    # pc：N x 4(xyz+intensity)
    # pc_cor_xyz: N+d x 3(xyz)
    N, _ = pc.shape
    N_all, _ = pc_cor_xyz.shape
    if N == N_all:
        return np.hstack((pc_cor_xyz, pc[:, 3].reshape([-1,1])))
    else:
        pc_cor = np.hstack((pc_cor_xyz, np.vstack((pc[:,3].reshape([-1,1]), np.zeros((N_all-N,1))))))
        for i in range(N, N_all):     
            dist = np.sum((pc[:,:3] - pc_cor[i,:3])**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, n_b, axis=0)[:n_b]
            pc_cor[i,3] = np.sum(pc_cor[idx,3])/n_b
        return pc_cor
    
''' Delete outliers by zscore'''
def del_outlier_axis(data, num=10):
    # by samples
    z_abs = np.abs(data - data.mean()) / (data.std())
    list_ = []
    for _ in range(num):
        index_max = np.argmax(z_abs)
        list_.append(index_max)
        z_abs[index_max] = 0
    return np.delete(data, list_)


'''convert cartesian coordinates into spherical ones'''
def car2sph_pc(pointcloud):
    '''
    args:
        points: N x 3 : x, y, and z
    return:
        points: N x 3 : r, phi, and theta
    '''
    r_sph = np.sqrt(pointcloud[:,0]**2 + pointcloud[:,1]**2 + pointcloud[:,2]**2)
    phi = np.arctan2(pointcloud[:,1],pointcloud[:,0])
    the = the = np.arccos(pointcloud[:,2]/r_sph)
    return np.hstack((r_sph.reshape(-1,1), phi.reshape(-1,1), the.reshape(-1,1)))

'''convert spherical coordinates into cartesian ones'''
def sph2car_pc(pointcloud):
    '''
    args:
        points: N x 3 : r, phi, and theta
    return:
        points: N x 3 : x, y, and z
    '''
    x = pointcloud[:,0]*np.sin(pointcloud[:,2])*np.cos(pointcloud[:,1])
    y = pointcloud[:,0]*np.sin(pointcloud[:,2])*np.sin(pointcloud[:,1])
    z = pointcloud[:,0]*np.cos(pointcloud[:,2])
    return np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))


### Noise ###
'''
Add Uniform noise to point cloud 
'''
def uniform_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.1][severity-1]
    jitter = np.random.uniform(-c,c,(N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc

'''
Add Gaussian noise to point cloud 
'''
def gaussian_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.02, 0.03, 0.04, 0.05, 0.06][severity-1]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc

'''
Add noise to the edge-length-2 cude
'''
def background_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//45, N//40, N//35, N//30, N//20][severity-1]
    noise_pc = np.zeros((c, C))
    # noise_pc[:,0] = np.random.uniform(np.min(del_outlier_axis(pointcloud[:,0], 20)), np.max(del_outlier_axis(pointcloud[:,0], 20)),(c,))
    noise_pc[:,0] = np.random.uniform(np.min(pointcloud[:,0]), np.max(pointcloud[:,0]),(c,))
    # noise_pc[:,1] = np.random.uniform(np.min(del_outlier_axis(pointcloud[:,1], 20)), np.max(del_outlier_axis(pointcloud[:,1], 20)),(c,))
    noise_pc[:,1] = np.random.uniform(np.min(pointcloud[:,1]), np.max(pointcloud[:,1]),(c,))
    # delete outliers on z-direction because the hard boundary of pc_z range
    temp_z = del_outlier_axis(pointcloud[:,2], 200)
    noise_pc[:,2] = np.random.uniform(np.min(temp_z), np.max(temp_z),(c,))
    new_pc = np.concatenate((pointcloud,noise_pc),axis=0).astype('float32')
    return new_pc

'''
Upsampling
'''
def upsampling(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//10, N//8, N//6, N//4, N//2][severity-1]
    index = np.random.choice(N, c, replace=False)
    add = pointcloud[index] + np.random.uniform(-0.1,0.1,(c, C))
    new_pc = np.concatenate((pointcloud,add),axis=0).astype('float32')
    return new_pc
    
'''
Add impulse noise
'''
def impulse_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//30, N//25, N//20, N//15, N//10][severity-1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.choice([-1,1], size=(c,C)) * 0.2
    return pointcloud
    
'''
Add Uniform noise to point cloud radially (sensor-based)
'''
def uniform_noise_radial(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.04, 0.08, 0.12, 0.16, 0.2][severity-1]
    jitter = np.random.uniform(-c,c,N)
    new_pc = car2sph_pc(pointcloud)
    new_pc[:, 0] = new_pc[:, 0] + jitter * np.sqrt(3)
    new_pc = sph2car_pc(new_pc).astype('float32')
    return new_pc

'''
Add Gaussian noise to point cloud radially (sensor-based)
'''
def gaussian_noise_radial(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.04, 0.06, 0.08, 0.10, 0.12][severity-1]
    jitter = np.random.normal(size=N) * c
    new_pc = car2sph_pc(pointcloud)
    new_pc[:, 0] = new_pc[:, 0] + jitter * np.sqrt(3)
    new_pc = sph2car_pc(new_pc).astype('float32')
    return new_pc

'''
Add impulse noise radially (sensor-based)
'''
def impulse_noise_radial(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//30, N//25, N//20, N//15, N//10][severity-1]
    index = np.random.choice(N, c, replace=False)
    pointcloud = car2sph_pc(pointcloud)    
    pointcloud[index, 0] += np.random.choice([-1,1], size=c) * 0.4 * np.sqrt(3)
    pointcloud = sph2car_pc(pointcloud).astype('float32')
    return pointcloud


MAP = {'uniform': uniform_noise,
       'gaussian': gaussian_noise,
       'background': background_noise,
       'impulse': impulse_noise,
       'upsampling': upsampling,
       
       'uniform_rad': uniform_noise_radial,
       'gaussian_rad': gaussian_noise_radial,
       'impulse_rad': impulse_noise_radial,
       'original': None,
}



def opr_pc(path_pc, corruption, severity, dir_save=None, saving=False):
    pc = np.fromfile(path_pc,dtype=np.float32,count=-1).reshape([-1,4])
    pc_xyz = pc[:, 0:3]
    pc_r = pc[:, 3]        
    pc_crp_xyz = MAP[corruption](pc_xyz, severity)
    pc_crp = fill_intensity(pc, pc_crp_xyz)
    pc_crp =pc_crp.reshape(-1,)
    pc_crp =pc_crp.astype(np.float32)
    if saving==True:
        if not exists(dir_save):
            makedirs(dir_save)
        pc_crp.tofile(join(dir_save,basename(path_pc)))
    return pc_crp