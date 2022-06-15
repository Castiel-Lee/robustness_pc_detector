import numpy as np
from fog_simulation import ParameterSet, simulate_fog
import argparse
import glob
from pathlib import Path
from os.path import basename, join, exists
from os import makedirs

def opr_one_pc(path_pc, dir_save, rate, noise, saving=False):
    # path_pc='/wangzhijie_backup/kitti/training/velodyne/000016.bin'
    parameter_set = ParameterSet(alpha=rate, gamma=0.000001)
    points = np.fromfile(path_pc,dtype=np.float32,count=-1).reshape([-1,4])
    points[:,3] = points[:,3]*255
    pc_crp, _, _ = simulate_fog(parameter_set, points, noise)
    pc_crp[:,3] /= 255.0
    pc_crp =pc_crp.astype(np.float32)
    pc_crp =pc_crp.reshape(-1,)
    if saving==True:
        if not exists(dir_save):
            makedirs(dir_save)
        pc_crp.tofile(join(dir_save,basename(path_pc)))
    return pc_crp
