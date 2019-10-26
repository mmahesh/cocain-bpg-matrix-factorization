from subprocess import call
import numpy as np
from time import sleep


lam_vals_to_run = [0,1e-1] #Regularization vals
rank_vals_to_run = [5] # Rank
data_exp_options_to_run = [[3,1]] # currently only unconstrained setting 
betas_to_run = [0.0,0.4,0.2] #beta values 0 for PALM and others for iPALM
force_exp_to_run = [1] #0 to not force the experiment if the file exists
algo_breg_nums_to_run = [[1,2],[2,1],[3,2],[4,2]] # combination of algo and breg_num  described below
fun_num_abs_fun_num_vals_to_run = [[2,2],[1,3]] # combination of fun_num and abs_fun_num described below
seed_exp_to_run = [1] # 1 to enforce seed_exps
seed_exp_nums_to_run = np.arange(50) # number of seeds

# algos:  1: BPG-MF, 2: iPALM and PALM, 3: CoCaIn BPG-MF, 4: BPG-MF-WB
# breg_num: 1: Euclidean distance, 2: New Bregman distance as in the following paper

# Beyond Alternating Updates for Matrix Factorization with Inertial Bregman Proximal Gradient Algorithms
# https://arxiv.org/abs/1905.09050


# create combinations
cart_prod = [(a,b,c[0], c[1], e, f, g[0],g[1], h[0],h[1],i,j)\
for a in lam_vals_to_run \
for b in rank_vals_to_run \
for c in data_exp_options_to_run\
for e in betas_to_run\
for f in force_exp_to_run\
for g in algo_breg_nums_to_run\
for h in fun_num_abs_fun_num_vals_to_run\
for i in seed_exp_to_run\
for j in seed_exp_nums_to_run\
]


count = 0
for item in cart_prod:
	temp_list = item
	print(temp_list)

	# command to execute
	command_to_exec = ' python3 main.py --lam=' + str(temp_list[0]) \
	+ ' --rank=' + str(temp_list[1]) \
	+ ' --dataset_option=' + str(temp_list[2])\
	+ ' --exp_option=' + str(temp_list[3])\
	+ ' --beta=' + str(temp_list[4])\
	+ ' --force_exp=' + str(temp_list[5])\
	+ ' --algo=' + str(temp_list[6])\
	+ ' --breg_num=' + str(temp_list[7])\
	+ ' --fun_num=' + str(temp_list[8])\
	+ ' --abs_fun_num=' + str(temp_list[9])\
    + ' --seed_exp=' + str(temp_list[10])\
    + ' --seed_exp_num=' + str(temp_list[11])\
	+ ' &' # for going to next iteration without job in background.
    

	print("Command executing is " + command_to_exec)
	call(command_to_exec, shell=True)
	print('done executing in '+str(count))
	count += 1
	if (count >0) and (count%100 == 0):
		sleep(120)



