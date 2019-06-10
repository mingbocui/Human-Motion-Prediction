cd ../datasets/three_traj/train
rm -rf *.txt 

cd ../val
rm -rf *.txt

cd ../test
rm -rf *.txt

cd ../../../generatedSyntheticTraj


cp -avx 3Traj.txt ../datasets/three_traj/test
cp -avx 3Traj.txt ../datasets/three_traj/train
cp -avx 3Traj.txt ../datasets/three_traj/val