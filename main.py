"""
Code for the paper:
Beyond Alternating Updates for Matrix Factorization with 
Inertial Bregman Proximal Gradient Algorithms
Paper authors: Mahesh Chandra Mukkamala, Peter Ochs

Algorithms Implemented:
BPG: Bregman Proximal Gradient
CoCaIn BPG-MF: Convex Concave Inertial (CoCaIn) BPG for Matrix Factorization
BPG-MF-WB: BPG for Matrix Factorization with Backtracking
PALM: Proximal Alternating Linearized Minimization
iPALM: Inertial Proximal Alternating Linearized Minimization

References:
CoCaIn BPG paper: https://arxiv.org/abs/1904.03537
PALM paper: https://link.springer.com/article/10.1007/s10107-013-0701-9
iPALM paper: https://arxiv.org/abs/1702.02505

Contact: Mahesh Chandra Mukkamala (mukkamala@math.uni-sb.de)
"""

# starting to track time
import time
st_time = time.time()
time_vals = [st_time]

# load necessary packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from my_functions import *
import os 

# for logging
# import logging
# logging.basicConfig(filename='logs/main.log', filemode='a', format='%(levelname)s - %(message)s',level=logging.ERROR)
# logging.info('About to start the processing.')

np.random.seed(0) # incase of random initialization

# creating arguments to automate the experiments
import argparse
parser = argparse.ArgumentParser(description='Simple Experiments')
parser.add_argument('--lam', '--regularization-parameter', default=1e-1,type=float,  dest='lam')
parser.add_argument('--algo', '--algorithm', default=1,type=int,  dest='algo')
parser.add_argument('--beta', '--palm-beta', default=0,type=float,  dest='beta')
parser.add_argument('--max_iter', '--max_iter', default=1000,type=int,  dest='max_iter')
parser.add_argument('--dataset_option', '--dataset-option', default=2,type=int,  dest='dataset_option')
parser.add_argument('--rank', '--rank', default=5,type=int,  dest='rank')
parser.add_argument('--exp_option', '--exp_option', default=1,type=int,  dest='exp_option')
parser.add_argument('--fun_num', '--fun_num', default=1,type=int,  dest='fun_num')
parser.add_argument('--abs_fun_num', '--abs_fun_num', default=3,type=int,  dest='abs_fun_num')
parser.add_argument('--breg_num', '--breg_num', default=1,type=int,  dest='breg_num')
parser.add_argument('--uL_est', '--uL_est', default=0.01,type=float,  dest='uL_est')
parser.add_argument('--lL_est', '--lL_est', default=0.01,type=float,  dest='lL_est')
parser.add_argument('--force_exp', '--force_exp', default=0,type=int,  dest='force_exp')
parser.add_argument('--seed_exp', '--seed_exp', default=0,type=int,  dest='seed_exp')
parser.add_argument('--seed_exp_num', '--seed_exp_num', default=0,type=int,  dest='seed_exp_num')

args = parser.parse_args()

# force_exp: to force exp even though file exists
# 0 for force_exp = False # 1 for force_exp=True

# uL_est used for CoCaIn BPG-MF, BPG-MF-WB (estimate for upper bound)
# lL_est used for CoCaIn BPG-MF  (estimate for lower bound bound)


# some backward compatibility and initialization
lam = args.lam
algo=args.algo
rank = args.rank
fun_num = args.fun_num
abs_fun_num = args.abs_fun_num
breg_num = args.breg_num
exp_option = args.exp_option
dataset_option = args.dataset_option
max_iter = args.max_iter
seed_exp = args.seed_exp
seed_exp_num = args.seed_exp_num
uL_est = args.uL_est
if algo==1:
	uL_est = 1.1 # BPG-MF with fixed upper bound value thus fixed step-size
lL_est = args.lL_est
beta = args.beta
if args.force_exp == 0:
	force_exp = False
else:
	force_exp = True

# logging.info('Arguments are '+ str(args) )

# Loading datasets with some backward compatibility
if dataset_option ==2:
	# Dataset option 2 = Medulloblastoma data set
	# More info give at  http://nimfa.biolab.si/nimfa.examples.medulloblastoma.html

	import nimfa
	A = nimfa.examples.medulloblastoma.read(normalize=True)
	U = np.ones((5893,rank))*0.1
	Z = np.ones((rank,34))*0.1
	
		
elif dataset_option == 3:
	# Dataset option 2 = Randomly generated synthetic data set
	A = np.loadtxt('matrix_200.txt', delimiter=',')
	dim = 200
	
	if seed_exp==0:
		U = np.ones((dim,rank))*0.1
		Z = np.ones((rank,dim))*0.1
	else:
		np.random.seed(seed_exp_num)
		U = np.random.rand(dim,rank)*0.1
		Z = np.random.rand(rank,dim)*0.1
	

elif dataset_option == 4:
	dim = 900
	rank = 10000
	A = np.loadtxt('data/Wcp.txt')
	
	print(A.shape)
	U = np.random.rand(dim,rank)*4 #(dim,rank)*0.1
	Z = U.T 
	lL_est = 0.0000001
	uL_est = 0.001
else:
	pass

# more initialization 
prev_U = U
prev_Z = Z


# Some functions required to run CoCaIn BPG based algorithms
def find_gamma(A,U,Z,prev_U,prev_Z,uL_est, lL_est):
	# Finding gamma for for CoCaIn BPG-MF
	gamma = 1 # best initial guess 
	kappa = 0.999999*(uL_est/(uL_est+lL_est)) # delta-epsilon chosen close to 1
	y_U = U+ gamma*(U-prev_U)
	y_Z = Z+ gamma*(Z-prev_Z)

	while ((kappa*breg(prev_U, prev_Z, U, Z, breg_num=breg_num,c_1=c_1,c_2=c_2)\
		-breg(U, Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2))<-1e-10):
		# thresholding

		# reduce inertia if the condition above fails
		gamma = gamma*0.9 
		y_U = U+ gamma*(U-prev_U)
		y_Z = Z+ gamma*(Z-prev_Z)
		if gamma <= 1e-10:
			# thresholding (not required)
			gamma = 0

	return y_U,y_Z, gamma

def do_lb_search(A, U, Z, U1, Z1, lam, uL_est,lL_est, warm_option=False):
	# Lower Bound Backtracking for CoCaIn BPG-MF
	backtracking_iter_counter = 0
	y_U,y_Z, gamma = find_gamma(A,U,Z,U1,Z1,uL_est, lL_est)
	while((abs_func(A, U, Z, y_U, y_Z, lam, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		-main_func(A, U, Z, lam, fun_num=fun_num)\
		-(lL_est*breg(U, Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2)))>1e-10):
		# thresholding
		lL_est = (1.1)*lL_est
		# print('Lower Backtracking with '+ str(lL_est))
		# print((abs_func(A, U, Z, y_U, y_Z, lam, abs_fun_num = abs_fun_num, fun_num=fun_num)))
		# print(main_func(A, U, Z, lam, fun_num=fun_num))
		# print(breg(U, Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2))

		
		# logging.info('Lower Backtracking with '+ str(lL_est))
		y_U,y_Z, gamma = find_gamma(A,U,Z,U1,Z1,uL_est, lL_est)
		backtracking_iter_counter+=1
	if backtracking_iter_counter == 0 and warm_option==True:
		lL_est = (0.9)*lL_est

	return lL_est, y_U, y_Z, gamma

def do_ub_search(A, y_U,y_Z, uL_est, warm_option=False):
	# Upper Bound Backtracking for CoCaIn BPG-MF, BPG-MF-WB
	backtracking_iter_counter = 0
	x_U,x_Z = make_update(y_U,y_Z, uL_est,lam, fun_num=fun_num, \
		abs_fun_num=abs_fun_num,breg_num=breg_num, A=A,c_1=c_1,c_2=c_2, exp_option=exp_option)

	while((abs_func(A, x_U,x_Z,y_U,y_Z, lam, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		-main_func(A, x_U,x_Z, lam,  fun_num=fun_num)\
		+(uL_est*breg(x_U, x_Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2)))<-1e-10):
		# thresholding
		backtracking_iter_counter+=1
		uL_est = (1.1)*uL_est
		# print('Upper Backtracking with '+ str(uL_est))
		# print(abs_func(A, x_U,x_Z,y_U,y_Z, lam, abs_fun_num = abs_fun_num, fun_num=fun_num))
		# print(main_func(A, x_U,x_Z, lam,  fun_num=fun_num))
		# print(breg(x_U, x_Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2))
		# logging.info('Lower Delta is ' + str(abs_func(A, x_U,x_Z,y_U,y_Z, lam, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		# -main_func(A, x_U,x_Z, lam,  fun_num=fun_num)\
		# +(uL_est*breg(x_U, x_Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2))))
		# logging.info('Lower Backtracking with '+ str(uL_est))
		
		x_U,x_Z = make_update(y_U,y_Z, uL_est,lam, fun_num=fun_num, \
			abs_fun_num=abs_fun_num,breg_num=breg_num, A=A,c_1=c_1,c_2=c_2, exp_option=exp_option)
	
	if backtracking_iter_counter==0 and warm_option==True:
		uL_est = (0.9)*uL_est

	return uL_est, x_U, x_Z


if algo==1:
	# BPG-MF implementation
	# BPG-MF: Bregman Proximal Gradient for Matrix Factorization

	# BPG-MF takes the following two parameters as input
	c_1 = 3
	c_2 = (np.linalg.norm(A))

	# Filenames creation
	if exp_option==1 and seed_exp==0:
		# without non-negativity constraints
		filename = 'results/bpg_mf_fun_name_'+str(fun_num)+'_dataset_option_'\
				+str(dataset_option)+'_abs_fun_num_'+str(abs_fun_num)\
				+'_breg_num_'+str(breg_num) + '_lam_val_'+str(lam)+'_rank_val_'+str(rank)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==0:
		# NMF settings with non-negativity constraints
		filename = 'results/bpg_mf_fun_name_'+str(fun_num)+'_dataset_option_'\
				+str(dataset_option)+'_abs_fun_num_'+str(abs_fun_num)\
				+'_breg_num_'+str(breg_num) + '_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
				+'_exp_option_'+str(exp_option)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==1 and seed_exp==1:
		# without non-negativity constraints
		filename = 'results/bpg_mf_fun_name_'+str(fun_num)+'_dataset_option_'\
				+str(dataset_option)+'_abs_fun_num_'+str(abs_fun_num)\
				+'_breg_num_'+str(breg_num) + '_lam_val_'+str(lam)\
					+'_rank_val_'+str(rank)+'_seed_exp_num_'+str(seed_exp_num)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==1:
		# NMF settings with non-negativity constraints
		filename = 'results/bpg_mf_fun_name_'+str(fun_num)+'_dataset_option_'\
				+str(dataset_option)+'_abs_fun_num_'+str(abs_fun_num)\
				+'_breg_num_'+str(breg_num) + '_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
				+'_exp_option_'+str(exp_option)+'_seed_exp_num_'+str(seed_exp_num)+'.txt'
	else:
		pass

	# can ignore the following if, elif statements
	# if statement for force_exp which repeats the experiment if it 
	# cannot find the file.
	# elif is just to handle the automation script the beta argument
	# is used for iPALM, so BPG does not require this and so we just use
	# one value of beta to run BPG once and ignore other betas.
	# TODO: Remove beta and find a better way to handle this
	if os.path.isfile(filename) and not force_exp:
		pass
	elif beta>0:
		pass
	else:
		# BPG for Matrix Factorization
		temp = main_func(A, U, Z, lam, fun_num=fun_num)
		print('temp is '+ str(temp))
		train_rmse = [np.sqrt((main_func(A, U, Z, lam, fun_num=0)*2)/A.size)]

		func_vals = [temp]
		lyapunov_vals = [temp]


		for i in range(max_iter):
			U,Z = make_update(U,Z, uL_est,lam, fun_num=fun_num, abs_fun_num=abs_fun_num,\
				breg_num=breg_num, A=A,c_1=c_1,c_2=c_2, exp_option=exp_option)
			gamma = 0
			temp = main_func(A, U, Z, lam, fun_num=fun_num)
			
			rmse = (main_func(A, U, Z, lam, fun_num=0)*2)/A.size
			train_rmse = train_rmse + [rmse]
			print('BPG fun val is '+ str(temp)+ ' iter ' + str(i) + ' rmse ' + str(rmse))
			# print('rmse is '+ str(rmse))
			if np.isnan(temp):
				raise
			if np.isnan(rmse):
				raise

			func_vals = func_vals + [temp]
			time_vals = time_vals + [time.time()]

		
		np.savetxt(filename,np.c_[func_vals, time_vals, train_rmse])
elif algo==2:
	# iPALM and PALM

	# Filenames creation
	if exp_option==1 and seed_exp==0:
		# without non-negativity constraints
		filename = 'results/palm_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
				+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)\
				+'_beta_'+str(beta)+ '_lam_val_'+str(lam)+'_rank_val_'+str(rank)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==0:
		# NMF settings with non-negativity constraints
		filename = 'results/palm_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
				+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)\
				+'_beta_'+str(beta)+ '_lam_val_'+str(lam)+'_rank_val_'+str(rank)+'_exp_option'+str(exp_option)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==1 and seed_exp==1:
		# without non-negativity constraints
		filename = 'results/palm_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
				+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)\
				+'_beta_'+str(beta)+ '_lam_val_'+str(lam)+'_rank_val_'+str(rank)+'_seed_exp_num_'+str(seed_exp_num)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==1:
		# NMF settings with non-negativity constraints
		filename = 'results/palm_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
				+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)\
				+'_beta_'+str(beta)+ '_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_exp_option'+str(exp_option)+'_seed_exp_num_'+str(seed_exp_num)+'.txt'
		# logging.info('Fileneme is '+ filename)
	else:
		pass

	if os.path.isfile(filename) and not force_exp:
		pass
	else:
		temp = main_func(A, U, Z, lam, fun_num=fun_num)
		train_rmse = [np.sqrt((main_func(A, U, Z, lam, fun_num=0)*2)/A.size)]


		func_vals = [temp]
		lyapunov_vals = [temp]
		print('PALM/iPALM fun val is '+ str(temp)+ ' iter ' + str(0))
		for i in range(max_iter):
			t_U,t_Z = make_update(U,Z, uL_est,lam, fun_num=fun_num, A=A, abs_fun_num=abs_fun_num,\
				breg_num=breg_num, U2=prev_U,Z2=prev_Z,beta=beta)
			prev_U = U
			prev_Z = Z
			U = t_U
			Z = t_Z
			gamma = 0
			temp = main_func(A, U, Z, lam, fun_num=fun_num)
			print('PALM/iPALM fun val is '+ str(temp)+ ' iter ' + str(i))


			rmse = (main_func(A, U, Z, lam, fun_num=0)*2)/A.size
			train_rmse = train_rmse + [rmse]

			if np.isnan(temp):
				raise
			if np.isnan(rmse):
				raise

			func_vals = func_vals + [temp]
			time_vals = time_vals + [time.time()]

		np.savetxt(filename,np.c_[func_vals, time_vals, train_rmse])
elif algo==3:
	# CoCaIn BPG-MF implementation
	# CoCaIn BPG-MF: CoCaIn BPG for Matrix Factorization

	lL_est_main = lL_est
	c_1 = 3
	c_2 = (np.linalg.norm(A))
	

	# Filenames creation
	if exp_option==1 and seed_exp==0:
		# without non-negativity constraints
		filename = 'results/cocain_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+\
					 '_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==0:
		# NMF settings with non-negativity constraints
		filename = 'results/cocain_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ \
					'_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_exp_option_'+str(exp_option)+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'.txt'
		# logging.info('Fileneme is '+ filename)
	if exp_option==1 and seed_exp==1:
		# without non-negativity constraints
		filename = 'results/cocain_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+\
					 '_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'_seed_exp_num_'+str(seed_exp_num)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==1:
		# NMF settings with non-negativity constraints
		filename = 'results/cocain_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ \
					'_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_exp_option_'+str(exp_option)+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'_seed_exp_num_'+str(seed_exp_num)+'.txt'
		# logging.info('Fileneme is '+ filename)
	else:
		pass

	# can ignore the following if, elif statements
	# if statement for force_exp which repeats the experiment if it 
	# cannot find the file.
	# elif is just to handle the automation script the beta argument
	# is used for iPALM, so BPG does not require this and so we just use
	# one value of beta to run BPG once and ignore other betas.
	# TODO: Remove beta and find a better way to handle this
	if os.path.isfile(filename) and not force_exp:
		pass
	elif beta >0:
		pass
	else:
		gamma_vals = [np.sqrt(uL_est/(uL_est+lL_est+1e-8))] # some initialization (can be 0)
		uL_est_vals = [uL_est]
		lL_est_vals = [lL_est]

		train_rmse = [np.sqrt((main_func(A, U, Z, lam, fun_num=0)*2)/A.size)]
		temp = main_func(A, U, Z, lam, fun_num=fun_num)

		func_vals = [temp]
		lyapunov_vals = [temp]

		print('CoCaIn BPG fun val is '+ str(temp)+ ' iter ' + str(0) )

		for i in range(max_iter):
			lL_est, y_U, y_Z, gamma = do_lb_search(A, U, Z, prev_U, prev_Z, lam, uL_est,lL_est=lL_est_main)
			
			# print('gamma ', gamma)
			prev_U = U
			prev_Z = Z
			uL_est, U, Z = do_ub_search(A, y_U,y_Z, uL_est)

			uL_est_vals = uL_est_vals + [uL_est]
			lL_est_vals = lL_est_vals + [lL_est]
			gamma_vals = gamma_vals + [gamma]

			temp = main_func(A, U, Z, lam, fun_num=fun_num)
			rmse = np.sqrt((main_func(A, U, Z, lam, fun_num=0)*2)/A.size)
			train_rmse = train_rmse + [rmse]

			if np.isnan(temp):
				raise
			if np.isnan(rmse):
				raise

			print('CoCaIn BPG fun val is '+ str(temp)+ ' iter ' + str(i) )
			# print('Lyapunov function is '+ str(((1/uL_est)*temp) +breg( U, Z, prev_U, prev_Z, \
			# 	breg_num=breg_num,c_1=c_1,c_2=c_2)))
			func_vals = func_vals + [temp]
			lyapunov_vals = lyapunov_vals + [((1/uL_est)*temp) +breg( U, Z, prev_U, prev_Z, \
				breg_num=breg_num,c_1=c_1,c_2=c_2)]
			time_vals = time_vals + [time.time()]
		print(filename)
		np.savetxt(filename,np.c_[func_vals,time_vals, lyapunov_vals, uL_est_vals, lL_est_vals, \
			gamma_vals, train_rmse])
elif algo==4:
	# BPG-MF-WB implementation
	# BPG-MF-WB: BPG With Backtracking

	lL_est_main = lL_est

	c_1 = 3
	c_2 = (np.linalg.norm(A))

	# Filenames creation
	if exp_option==1 and seed_exp==0:
		# without non-negativity constraints
		filename = 'results/bpg_mf_wb_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ \
					'_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==0:
		# NMF settings with non-negativity constraints
		filename = 'results/bpg_mf_wb_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ \
					'_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_exp_option_'+str(exp_option)+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'.txt'
		# logging.info('Fileneme is '+ filename)
	if exp_option==1 and seed_exp==1:
		# without non-negativity constraints
		filename = 'results/bpg_mf_wb_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ \
					'_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'_seed_exp_num_'+str(seed_exp_num)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==1:
		# NMF settings with non-negativity constraints
		filename = 'results/bpg_mf_wb_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ \
					'_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_exp_option_'+str(exp_option)+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'_seed_exp_num_'+str(seed_exp_num)+'.txt'
		# logging.info('Fileneme is '+ filename)
	else:
		pass

	# +'_seed_exp_num_'+str(seed_exp_num)
	# can ignore the following if, elif statements
	# if statement for force_exp which repeats the experiment if it 
	# cannot find the file.
	# elif is just to handle the automation script the beta argument
	# is used for iPALM, so BPG does not require this and so we just use
	# one value of beta to run BPG once and ignore other betas.
	# TODO: Remove beta and find a better way to handle this
	if os.path.isfile(filename) and not force_exp:
		pass
	elif beta >0:
		pass
	else:
		gamma_vals = [0]
		uL_est_vals = [uL_est]
		lL_est_vals = [lL_est]

		train_rmse = [np.sqrt((main_func(A, U, Z, lam, fun_num=0)*2)/A.size)]
		temp = main_func(A, U, Z, lam, fun_num=fun_num)

		func_vals = [temp]
		lyapunov_vals = [temp]
		for i in range(max_iter):
			gamma = 0

			uL_est, U, Z = do_ub_search(A, U, Z, uL_est)

			uL_est_vals = uL_est_vals + [uL_est]
			lL_est_vals = lL_est_vals + [lL_est]
			gamma_vals = gamma_vals + [gamma]
			prev_fun_val = temp
			temp = main_func(A, U, Z, lam, fun_num=fun_num)
			# if temp>prev_fun_val:
			# 	print('fun ', temp)
			# 	print('prev_fun ', prev_fun_val)
			# 	raise
			
			rmse = np.sqrt((main_func(A, U, Z, lam, fun_num=0)*2)/A.size)
			train_rmse = train_rmse + [rmse]

			if np.isnan(temp):
				raise
			if np.isnan(rmse):
				raise

			print('BPG-WB fun val is '+ str(temp)+ ' iter ' + str(i) + ' rmse '+ str(rmse))
			func_vals = func_vals + [temp]
			lyapunov_vals = lyapunov_vals + [((1/uL_est)*temp) +breg( U, Z, prev_U,\
			 prev_Z, breg_num=breg_num,c_1=c_1,c_2=c_2)]
			time_vals = time_vals + [time.time()]
		np.savetxt(filename,np.c_[func_vals,time_vals, lyapunov_vals, uL_est_vals, lL_est_vals,\
		 gamma_vals, train_rmse])
elif algo==6:
	# CoCaIn BPG-MF implementation (Heuristic for now so can ignore)
	# CoCaIn BPG-MF: CoCaIn BPG for Matrix Factorization

	lL_est_main = lL_est

	c_1 = 3
	c_2 = (np.linalg.norm(A))

	# Filenames creation
	if exp_option==1 and seed_exp==0:
		# without non-negativity constraints
		filename = 'results/cocain_warm_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+\
					 '_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'.txt'
		# logging.info('Fileneme is '+ filename)
	elif exp_option==2 and seed_exp==0:
		# NMF settings with non-negativity constraints
		filename = 'results/cocain_warm_mf_fun_name_'+str(fun_num)+'_dataset_option_'+str(dataset_option)\
					+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ \
					'_lam_val_'+str(lam)+'_rank_val_'+str(rank)\
					+'_exp_option_'+str(exp_option)+'_uL_est_'+str(uL_est)+'_lL_est_'+str(lL_est)+'.txt'
		# logging.info('Fileneme is '+ filename)
	else:
		pass

	# can ignore the following if, elif statements
	# if statement for force_exp which repeats the experiment if it 
	# cannot find the file.
	# elif is just to handle the automation script the beta argument
	# is used for iPALM, so BPG does not require this and so we just use
	# one value of beta to run BPG once and ignore other betas.
	# TODO: Remove beta and find a better way to handle this
	if os.path.isfile(filename) and not force_exp:
		pass
	elif beta >0:
		pass
	else:
		gamma_vals = [np.sqrt(uL_est/(uL_est+lL_est+1e-8))] # some initialization (can be 0)
		uL_est_vals = [uL_est]
		lL_est_vals = [lL_est]

		train_rmse = [np.sqrt((main_func(A, U, Z, lam, fun_num=0)*2)/A.size)]
		temp = main_func(A, U, Z, lam, fun_num=fun_num)

		func_vals = [temp]
		lyapunov_vals = [temp]

		for i in range(max_iter):
			lL_est, y_U, y_Z, gamma = do_lb_search(A, U, Z, prev_U, prev_Z, lam, uL_est,lL_est=lL_est,warm_option=True)
			prev_U = U
			prev_Z = Z
			uL_est, U, Z = do_ub_search(A, y_U,y_Z, uL_est,warm_option=True)

			uL_est_vals = uL_est_vals + [uL_est]
			lL_est_vals = lL_est_vals + [lL_est]
			gamma_vals = gamma_vals + [gamma]

			temp = main_func(A, U, Z, lam, fun_num=fun_num)
			rmse = np.sqrt((main_func(A, U, Z, lam, fun_num=0)*2)/A.size)
			train_rmse = train_rmse + [rmse]

			if np.isnan(temp):
				raise
			if np.isnan(rmse):
				raise

			print('CoCaIn WARM BPG fun val is '+ str(temp)+ ' iter ' + str(i) + ' RMSE ' + str(rmse))
			print('Lyapunov function is '+ str(((1/uL_est)*temp) +breg( U, Z, prev_U, prev_Z, \
				breg_num=breg_num,c_1=c_1,c_2=c_2)))
			func_vals = func_vals + [temp]
			lyapunov_vals = lyapunov_vals + [((1/uL_est)*temp) +breg( U, Z, prev_U, prev_Z, \
				breg_num=breg_num,c_1=c_1,c_2=c_2)]
			time_vals = time_vals + [time.time()]
		print(filename)
		np.savetxt(filename,np.c_[func_vals,time_vals, lyapunov_vals, uL_est_vals, lL_est_vals, \
			gamma_vals, train_rmse])
