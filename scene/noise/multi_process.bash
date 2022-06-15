echo '$1 will be processed.'

for i in $(seq 1 5)
do   
echo severity $i will be processed
python Simulation_noise.py --path_data=/wangzhijie/kitti/training/velodyne --save_dir=/wangzhijie/kitti/training -n=$1 -r=$i
done

echo '$1 will be processed.'