echo '$1 will be processed.'

for i in $(seq 1 5)
do   
echo severity $i will be processed
python Simulation_noise_obj.py -d=/wangzhijie/kitti -n=$1 -r=$i
done
echo '$1 has been be processed.'