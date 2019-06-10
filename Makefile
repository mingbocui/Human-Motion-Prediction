@ csgan:
	# python scripts/cgs_train.py --mode 'refinement' --rollout_steps 50 --rollout_rate 10
	python scripts/cgs_train.py --mode 'refinement' --rollout_steps 75 --rollout_rate 10
	python scripts/cgs_train.py --mode 'refinement' --rollout_steps 50 --rollout_rate 20
	python scripts/cgs_train.py --mode 'refinement' --rollout_steps 75 --rollout_rate 20
	python scripts/cgs_train.py --mode 'refinement' --rollout_steps 25 --rollout_rate 20
	python scripts/cgs_train.py --mode 'refinement' --rollout_steps 75 --rollout_rate 5
	python scripts/cgs_train.py --mode 'refinement' --rollout_steps 50 --rollout_rate 5


# @ ff20:
	# python scripts/train.py --pred_len 12 --dataset_name 'eth'
	# python scripts/evaluate_model.py 
	# python scripts/train.py --pred_len 12 --dataset_name 'hotel'
	# python scripts/evaluate_model.py 
	# python scripts/train.py --pred_len 12 --dataset_name 'univ'
	# python scripts/evaluate_model.py 
	# python scripts/train.py --pred_len 12 --dataset_name 'zara1'
	# python scripts/evaluate_model.py 
	# python scripts/train.py --pred_len 12 --dataset_name 'zara2'
	# python scripts/evaluate_model.py 
	# python scripts/train.py --pred_len 12 --dataset_name 'eth' --D_type 'ff'
	# python scripts/evaluate_model.py --D_type 'ff'
	# python scripts/train.py --pred_len 12 --dataset_name 'hotel' --D_type 'ff'
	# python scripts/evaluate_model.py --D_type 'ff'
	# # python scripts/train.py --pred_len 12 --dataset_name 'univ' --D_type 'ff'
	# # python scripts/evaluate_model.py --D_type 'ff'
	# # python scripts/train.py --pred_len 12 --dataset_name 'zara1' --D_type 'ff'
	# # python scripts/evaluate_model.py --D_type 'ff'
	# # python scripts/train.py --pred_len 12 --dataset_name 'zara2' --D_type 'ff'
	# # python scripts/evaluate_model.py --D_type 'ff'
	# cp mult20.txt safemult20.txt


# @ ff1:
# 	python scripts/train.py --pred_len 8 --dataset_name 'eth'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 8 --dataset_name 'hotel'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 8 --dataset_name 'univ'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 8 --dataset_name 'zara1'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 8 --dataset_name 'zara2'
# 	python scripts/evaluate_model.py
# 	python scripts/train.py --pred_len 12 --dataset_name 'eth'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 12 --dataset_name 'hotel'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 12 --dataset_name 'univ'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 12 --dataset_name 'zara1'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 12 --dataset_name 'zara2'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 8 --dataset_name 'eth' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 8 --dataset_name 'hotel' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 8 --dataset_name 'univ' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 8 --dataset_name 'zara1' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 8 --dataset_name 'zara2' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 12 --dataset_name 'eth' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 12 --dataset_name 'hotel' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 12 --dataset_name 'univ' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 12 --dataset_name 'zara1' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/train.py --pred_len 12 --dataset_name 'zara2' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	cp mult1.txt safemult1.txt


# @ ff3:
# 	python scripts/train.py --pred_len 8 --dataset_name 'eth'
# 	python scripts/evaluate_model.py 
# 	python scripts/ffd_train.py --pred_len 8 --dataset_name 'hotel'
# 	python scripts/evaluate_model.py 
# 	python scripts/ffd_train.py --pred_len 8 --dataset_name 'univ'
# 	python scripts/evaluate_model.py 
# 	python scripts/ffd_train.py --pred_len 8 --dataset_name 'zara1'
# 	python scripts/evaluate_model.py 
# 	python scripts/ffd_train.py --pred_len 8 --dataset_name 'zara2'
# 	python scripts/evaluate_model.py
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'eth'
# 	python scripts/evaluate_model.py 
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'hotel'
# 	python scripts/evaluate_model.py 
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'univ'
# 	python scripts/evaluate_model.py 
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'zara1'
# 	python scripts/evaluate_model.py 
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'zara2'
# 	python scripts/evaluate_model.py 
# 	python scripts/train.py --pred_len 8 --dataset_name 'eth' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 8 --dataset_name 'hotel' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 8 --dataset_name 'univ' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 8 --dataset_name 'zara1' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 8 --dataset_name 'zara2' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'eth' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'hotel' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'univ' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'zara1' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	python scripts/ffd_train.py --pred_len 12 --dataset_name 'zara2' --D_type 'ff'
# 	python scripts/evaluate_model.py --D_type 'ff'
# 	cp ff3.txt safeff3.txt
