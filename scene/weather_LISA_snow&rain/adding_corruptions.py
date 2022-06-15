import argparse
import glob
from pathlib import Path
from atmos_models import LISA
import numpy as np
from os.path import basename, join, exists
from os import makedirs

# parser = argparse.ArgumentParser()
# parser.add_argument('-d','--path_data', default=None)
# parser.add_argument('-s','--save_dir', default=None)
# args = parser.parse_args()
# path_pc = args.path_data
# save_dir = args.save_dir

def opr_one_pc(path_pc, dir_save, atm_model, saving=False, rate=0):
    pc = np.fromfile(path_pc,dtype=np.float32,count=-1).reshape([-1,4])
    lisa = LISA(atm_model=atm_model)
    if atm_model == 'snow' or atm_model == 'rain':
        # print(atm_model,rate)
        pc_crp = lisa.augment(pc,rate)
    else:
        print(atm_model)
        pc_crp = lisa.augment(pc)
    pc_crp = pc_crp[:,0:4]
    pc_crp =pc_crp.reshape(-1,)
    pc_crp =pc_crp.astype(np.float32)
    if saving==True:
        if not exists(dir_save):
            makedirs(dir_save)
        pc_crp.tofile(join(dir_save,basename(path_pc)))
    return pc_crp
    

