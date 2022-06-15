import argparse
import glob
from pathlib import Path
import numpy as np
from os.path import basename, join, isdir
from adding_fog import opr_one_pc as corrupting_scan
import time

parser = argparse.ArgumentParser()
parser.add_argument('-d','--path_data', type=str, default=None)
parser.add_argument('-s','--save_dir', type=str, default=None)
parser.add_argument('-r','--rate', type=float, default=None) # 0.005, 0.01, 0.02, 0.03, 0.06
parser.add_argument('-n','--noise', type=float, default=None)

parser.add_argument('-o','--start_id', type=int, default=0)
parser.add_argument('-e','--end_id', type=int, default=None)

args = parser.parse_args()
path_pc = args.path_data
dir_save = args.save_dir

ext='.bin'
pc_list = glob.glob(str(path_pc + f'/*{ext}')) if isdir(path_pc) else [path_pc]
pc_list.sort()
rate=args.rate
noise = args.noise
dir_save =join(dir_save,'velodyne_LFS_fog'+str(noise)+'_'+str(rate))

star_pc=args.start_id
if args.end_id==None:
    end_pc=len(pc_list)
else:
    end_pc=args.end_id

N_pc=end_pc-star_pc
print('Total number of pc scans:',N_pc)
print('Corrupting model:','velodyne_LFS_fog'+str(noise)+'_'+str(rate))
start = time.time()
for (n, path_scan) in enumerate(pc_list):
    if end_pc > n >=star_pc:
        corrupting_scan(path_pc=path_scan, dir_save=dir_save, saving=True, rate=rate, noise = noise)
        now=time.time()
        exp_rem_tim = (now - start) / (n + 1 - star_pc) * (end_pc  - (n + 1))
        print('\t{} scans have been corrupted.'.format(n+1),'\t estimated remaining time {:.2f} mins or {:.2f} hrs'.format(exp_rem_tim/60, exp_rem_tim/3600))
