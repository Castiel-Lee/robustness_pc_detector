import argparse
import glob
from pathlib import Path
import numpy as np
import time
from Add_noise_obj import opr_pc as corrupting_scan

def get_idx(root_split_path, split):
    idx_file = Path(root_split_path) / 'ImageSets' / ('%s.txt' % split)
    assert idx_file.exists()
    temp_list = open(idx_file).readlines()
    return [temp_item.strip() for temp_item in temp_list]

parser = argparse.ArgumentParser()
parser.add_argument('-d','--root_path', type=str, default=None)
parser.add_argument('-n','--noise_model', type=str, default=None)
parser.add_argument('-r','--severity', type=int, default=None)

parser.add_argument('-o','--start_id', type=int, default=0)
parser.add_argument('-e','--end_id', type=int, default=None)

args = parser.parse_args()

root_path = args.root_path
noise_model=args.noise_model
severity=args.severity
star_pc=args.start_id
end_pc=args.end_id

# root_path='D:/dataset/kitti'
# noise_model='distortion'
# severity=5
# star_pc=0
# end_pc=None

# splits=['train','val']
splits=['val']

idx_list=[]
for split in splits:
    idx_list += get_idx(root_path, split)
idx_list.sort()

if end_pc==None:
    end_pc=len(idx_list)

N_pc=end_pc-star_pc
print('Total number of pc scans:',N_pc)
print('Corrupting model:',noise_model+'_'+str(severity))
start = time.time()
for (n, idx) in enumerate(idx_list):
    
    if end_pc > n >=star_pc:
        corrupting_scan(root_path=root_path, sample_idx=idx, corruption=noise_model, severity=severity, saving=True)
        now=time.time()
        exp_rem_tim = (now - start) / (n + 1 - star_pc) * (end_pc  - (n + 1))
        print('\t{} scans have been corrupted.'.format(idx),'\t Estimated remaining time {:.2f} mins or {:.2f} hrs'.format(exp_rem_tim/60, exp_rem_tim/3600))