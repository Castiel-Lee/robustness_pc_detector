import argparse
import glob
from pathlib import Path
from atmos_models import LISA
import numpy as np
from os.path import basename, join, isdir
from adding_corruptions import opr_one_pc as corrupting_scan
import time

parser = argparse.ArgumentParser()
parser.add_argument('-d','--path_data', type=str, default=None)
parser.add_argument('-s','--save_dir', type=str, default=None)
parser.add_argument('-a','--atm_model', type=str, default=None)
parser.add_argument('-r','--rate', type=float, default=0)

parser.add_argument('-o','--start_id', type=int, default=0)
parser.add_argument('-e','--end_id', type=int, default=None)

args = parser.parse_args()
path_pc = args.path_data
dir_save = args.save_dir

# path_pc = '/home/malei/shuangzhL_workspace/OpenPCDet-0.5.0/data/kitti/training/velodyne'
# dir_save = '/home/malei/shuangzhL_workspace/OpenPCDet-0.5.0/data/kitti/training'

ext='.bin'
pc_list = glob.glob(str(path_pc + f'/*{ext}')) if isdir(path_pc) else [path_pc]
pc_list.sort()
atm_model=args.atm_model
rate=args.rate
dir_save =join(dir_save,'velodyne_'+atm_model+'_'+str(rate))


star_pc=args.start_id
if args.end_id==None:
    end_pc=len(pc_list)
else:
    end_pc=args.end_id

N_pc=end_pc-star_pc
print('Total number of pc scans:',N_pc)
print('Corrupting model:',atm_model+'_'+str(rate))
start = time.time()
for (n, path_scan) in enumerate(pc_list):
    if end_pc > n >=star_pc:
        corrupting_scan(path_pc=path_scan, dir_save=dir_save, atm_model=atm_model, saving=True, rate=rate)
        now=time.time()
        exp_rem_tim = (now - start) / (n + 1 - star_pc) * (end_pc  - (n + 1))
        print('\t{} scans have been corrupted.'.format(n+1),'\t estimated remaining time {:.2f} mins or {:.2f} hrs'.format(exp_rem_tim/60, exp_rem_tim/3600))
