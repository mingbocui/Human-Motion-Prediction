cd ../datasets/five_traj/train
rm -rf *.txt 

cd ../val
rm -rf *.txt

cd ../test
rm -rf *.txt

cd ../../../generatedSyntheticTraj


cp -avx 5Traj.txt ../datasets/five_traj/test
cp -avx 5Traj.txt ../datasets/five_traj/train
cp -avx 5Traj.txt ../datasets/five_traj/val