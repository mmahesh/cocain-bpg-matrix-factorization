import numpy as np
np.random.seed(0)

def main_func(A, U, Z, lam, fun_num=1):
	# main functions
	if fun_num==0:
		# no-regularization
		return 0.5*(np.linalg.norm(A-np.matmul(U,Z))**2) 
	if fun_num==1:
		# L2-regularization
		return 0.5*(np.linalg.norm(A-np.matmul(U,Z))**2) \
			+ lam*0.5*(np.linalg.norm(U)**2) \
			+ lam*0.5*(np.linalg.norm(Z)**2)
	if fun_num==2:
		# L1-regularization
		return 0.5*(np.linalg.norm(A-np.matmul(U,Z))**2) \
		+ lam*(np.sum(np.abs(U))) + lam*(np.sum(np.abs(Z)))
	if fun_num==3:
		# No-regularization for some backward compatibility TODO.
		return 0.5*(np.linalg.norm(A -np.matmul(U,Z))**2) 

def grad(A, U, Z, lam, fun_num=1, option=1):
	# Gradients of smooth part of the function
	# Essentially function is f+g
	# f is non-smooth part
	# g is smooth part
	# here gradients of g is computed.

	# no-regularization 
	# option 1 gives all gradients
	# option 2 gives gradient with respect to U
	# option 3 gives gradient with respect to Z
	if fun_num in [0,1,2]:
		# grad u, grad z
		if option==1:
			return np.matmul(np.matmul(U,Z)-A, Z.T) , np.matmul(U.T, np.matmul(U,Z)-A) 
		elif option==2:
			return np.matmul(np.matmul(U,Z)-A, Z.T)
		elif option==3:
			return np.matmul(U.T, np.matmul(U,Z)-A) 
		else:
			pass



def abs_func(A, U, Z, U1, Z1, lam, abs_fun_num=1, fun_num=1):
	# Denote abs_func = f(x) + g(x^k) + <grad g(x^k), x-x^k>
	# x^k is the current iterate denoted by U1, Z1
	# This function is just to make the code easy to handle
	# There can be other efficient ways to implement TODO

	if abs_fun_num == 1:
		G0,G1 = grad(A, U1, Z1, lam, fun_num=fun_num)
		return main_func(A, U1, Z1, lam, fun_num=1) + np.sum(np.multiply(U-U1,G0)) \
		+ np.sum(np.multiply(Z-Z1,G1))
	if abs_fun_num == 2:
		G0,G1 = grad(A, U1, Z1, lam, fun_num=fun_num)
		return main_func(A, U1, Z1, lam, fun_num=fun_num)-lam*(np.sum(np.abs(U1))) - lam*(np.sum(np.abs(Z1)))\
		+ lam*(np.sum(np.abs(U))) + lam*(np.sum(np.abs(Z))) + np.sum(np.multiply(U-U1,G0)) + np.sum(np.multiply(Z-Z1,G1))
	if abs_fun_num == 3:
		G0,G1 = grad(A, U1, Z1, lam, fun_num=fun_num)
		return main_func(A, U1, Z1, lam, fun_num=fun_num)-lam*0.5*(np.linalg.norm(U1)**2) - lam*0.5*(np.linalg.norm(Z1)**2)\
		+ lam*0.5*(np.linalg.norm(U)**2) + lam*0.5*(np.linalg.norm(Z)**2) \
		+ np.sum(np.multiply(U-U1,G0)) + np.sum(np.multiply(Z-Z1,G1))

def make_update(U1, Z1,uL_est=1,lam=0,fun_num=1, abs_fun_num=1,breg_num=1, A=1, U2=1,Z2=1, beta=0.0,c_1=1,c_2=1,exp_option=1):
	# Main Update Step

	if breg_num ==2:
		# Calculates CoCaIn BPG-MF, BPG-MF, BPG-MF updates
		
		# Getting gradients to compute P^k, Q^k later
		grad_u, grad_z = grad(A, U1, Z1, lam, fun_num=0)
		grad_h_1_a = U1*(np.linalg.norm(U1)**2 + np.linalg.norm(Z1)**2)
		grad_h_1_b = Z1*(np.linalg.norm(U1)**2 + np.linalg.norm(Z1)**2)
		grad_h_2_a = U1
		grad_h_2_b = Z1
		sym_setting = 0
		
		if abs_fun_num == 3:
			# Code for No-Regularization  and L2 Regularization
			if exp_option==1:

				# Code for No-Regularization  and L2 Regularization
				# No-Regularization is equivalent to L2 Regularization with lam=0
				# compute P^k
				p_l = (1/uL_est)*grad_u - (c_1*grad_h_1_a + c_2*grad_h_2_a)
				# compute Q^k
				q_l = (1/uL_est)*grad_z - (c_1*grad_h_1_b + c_2*grad_h_2_b)
				if sym_setting == 0: #default option
					# solving cubic equation
					coeff = [c_1*(np.linalg.norm(p_l)**2 + np.linalg.norm(q_l)**2), 0,(c_2 + (lam/uL_est)), -1]
					temp_y = np.roots(coeff)[-1].real
					return (-1)*temp_y*p_l, (-1)*temp_y*q_l
				else:
					p_new = p_l + q_l.T
					coeff = [4*c_1*(np.linalg.norm(p_new)**2), 0,2*(c_2 + (lam/uL_est)), -1]
					temp_y = np.roots(coeff)[-1].real
					return (-1)*temp_y*p_new, (-1)*temp_y*(p_new.T)
			elif exp_option==2:
				# NMF case.
				# Code for No-Regularization  and L2 Regularization
				if sym_setting == 0:
					
					# compute P^k
					p_l = np.maximum(0,-(1/uL_est)*grad_u + (c_1*grad_h_1_a + c_2*grad_h_2_a))
					# compute Q^k
					q_l = np.maximum(0,-(1/uL_est)*grad_z + (c_1*grad_h_1_b + c_2*grad_h_2_b))
					
					# solving cubic equation
					temp_pnrm = np.sqrt((np.linalg.norm(p_l)**2 + np.linalg.norm(q_l)**2))/np.sqrt(2)
					# print('temp_pnrm '+ str(temp_pnrm))
					
					# technique to improve the numerical stability
					# same update anyway.
					coeff = [c_1*2, 0,(c_2 + (lam/uL_est)), -(temp_pnrm)]
					temp_y = np.roots(coeff)[-1].real
					
					return temp_y*p_l/temp_pnrm, temp_y*q_l/temp_pnrm
					
				else:
					temp_pl = -(1/uL_est)*grad_u + (c_1*grad_h_1_a + c_2*grad_h_2_a)
					temp_ql = -(1/uL_est)*grad_z + (c_1*grad_h_1_b + c_2*grad_h_2_b)
					# compute P^k
					p_new = np.maximum(0,temp_pl+temp_ql.T)
				
					# solving cubic equation
					coeff = [4*c_1*(np.linalg.norm(p_new)**2), 0,2*(c_2 + (lam/uL_est)), -1]
					temp_y = np.roots(coeff)[-1].real
					return temp_y*p_new, temp_y*(p_new.T)
			else:
				raise


		if abs_fun_num == 2:
			
			if exp_option==1:
				# L1 Regularization simple
				# compute P^k
				tp_l = (1/uL_est)*grad_u - (c_1*grad_h_1_a + c_2*grad_h_2_a)
				p_l = -np.maximum(0, np.abs(-tp_l)-lam*(1/uL_est))*np.sign(-tp_l)
				# compute Q^K
				tq_l = (1/uL_est)*grad_z - (c_1*grad_h_1_b + c_2*grad_h_2_b)
				q_l = -np.maximum(0, np.abs(-tq_l)-lam*(1/uL_est))*np.sign(-tq_l)
				# solving cubic equation
				coeff = [c_1*(np.linalg.norm(p_l)**2 + np.linalg.norm(q_l)**2), 0,(c_2), -1]
				temp_y = np.roots(coeff)[-1].real
				return (-1)*temp_y*p_l, (-1)*temp_y*q_l
			elif exp_option==2:
				# L1 Regularization NMF case

				# temporary matrices see update steps in the paper.
				nx = np.shape(grad_u)[0]
				ny = np.shape(grad_u)[1]
				temp_mat1 = np.outer(np.ones(nx),np.ones(ny))
				nx = np.shape(grad_z)[0]
				ny = np.shape(grad_z)[1]
				temp_mat2 = np.outer(np.ones(nx),np.ones(ny))

				# compute P^k
				tp_l = -(1/uL_est)*grad_u + (c_1*grad_h_1_a + c_2*grad_h_2_a) - (lam/uL_est)*(temp_mat1)
				p_l = np.maximum(0,tp_l)
				# compute Q^k
				tq_l = -(1/uL_est)*grad_z + (c_1*grad_h_1_b + c_2*grad_h_2_b) - (lam/uL_est)*(temp_mat2)
				q_l = np.maximum(0,tq_l)
				# solving cubic equation
				# print(np.linalg.norm(p_l)**2 + np.linalg.norm(q_l)**2)
				coeff = [c_1*(np.linalg.norm(p_l)**2 + np.linalg.norm(q_l)**2), 0,(c_2), -1]
				temp_y = np.roots(coeff)[-1].real
				return temp_y*p_l, temp_y*q_l
			else:
				pass
	
	if breg_num ==1:
		# Update steps for PALM and iPALM
		# Code for No-Regularization  and L2 Regularization
		if abs_fun_num == 3:
			# compute extrapolation
			U1 = U1+beta*(U1-U2)
			grad_u = grad(A, U1, Z1, lam, fun_num=fun_num, option=2)
			# compute Lipschitz constant
			L2 =  np.linalg.norm(np.mat(Z1) * np.mat(Z1.T))
			L2 = np.max([L2,1e-4])
			# print('L2 val '+ str(L2))
			if beta>0:
				# since we use convex regularizers
				# step-size is less restrictive
				step_size = (2*(1-beta)/(1+2*beta))*(1/	L2) 
			else:
				# from PALM paper 1.1 is just a scaling factor
				# can be set to any value >1.
				step_size = (1/(1.1*L2))

			# Update step for No-Regularization  and L2 Regularization
			U = ((U1 - step_size*grad_u))/(1+ step_size*lam)

			# compute extrapolation
			Z1 = Z1+beta*(Z1-Z2)
			grad_z = grad(A, U, Z1, lam, fun_num=fun_num, option=3)
			# compute Lipschitz constant
			L1 =  np.linalg.norm(np.mat(U.T) * np.mat(U))
			L1 = np.max([L1,1e-4])
			# print('L1 val '+ str(L1))
			if beta>0:
				# since we use convex regularizers
				# step-size is less restrictive
				step_size = (2*(1-beta)/(1+2*beta))*(1/	L1)
			else:
				# from PALM paper 1.1 is just a scaling factor
				# can be set to any value >1.
				step_size = 1/(1.1*L1)

			# Update step for No-Regularization  and L2 Regularization
			Z = ((Z1 - step_size*grad_z))/(1+ step_size*lam)
			return U,Z

		if abs_fun_num == 2:
			# Update steps for PALM and iPALM
			# Code for L1 Regularization

			# compute extrapolation
			U1 = U1+beta*(U1-U2)
			grad_u = grad(A, U1, Z1, lam, fun_num=fun_num, option=2)
			# compute Lipschitz constant
			L2 =  np.linalg.norm(np.mat(Z1) * np.mat(Z1.T))
			L2 = np.max([L2,1e-4])
			if beta>0:
				# since we use convex regularizers
				# step-size is less restrictive
				step_size = (2*(1-beta)/(1+2*beta))*(1/	L2)
			else:
				# from PALM paper 1.1 is just a scaling factor
				# can be set to any value >1.
				step_size = 1/(1.1*L2)

			# compute update step with U
			tU1 = ((U1 - step_size*grad_u))
			U = np.maximum(0, np.abs(tU1)-lam*(step_size))*np.sign(tU1)

			# compute extrapolation
			Z1 = Z1+beta*(Z1-Z2)
			grad_z = grad(A, U, Z1, lam, fun_num=fun_num, option=3)

			# compute Lipschitz constant
			L1 = np.linalg.norm(np.mat(U.T) * np.mat(U))
			L1 = np.max([L1,1e-4])
			if beta>0:
				# since we use convex regularizers
				# step-size is less restrictive
				step_size = (2*(1-beta)/(1+2*beta))*(1/	L1)
			else:
				# compute update step with U
				step_size = 1/(1.1*L1)

			# compute update step with z
			tZ1 = ((Z1 - step_size*grad_z))
			Z = np.maximum(0, np.abs(tZ1)-lam*(step_size))*np.sign(tZ1)

			return U,Z

def breg( U, Z, U1, Z1, breg_num=1, c_1=1,c_2=1):
	if breg_num==1:
		# Standard Euclidean distance
		temp =  0.5*(np.linalg.norm(U-U1)**2) + 0.5*(np.linalg.norm(Z-Z1)**2)
		if abs(temp) <= 1e-10:
			# to fix numerical issues
			temp = 0
		if temp<0:
			return 0
		return temp
	if breg_num==2:
		# New Bregman distance as in the paper
		# link: https://arxiv.org/abs/1905.09050
		grad_h_1_a = U1*(np.linalg.norm(U1)**2 + np.linalg.norm(Z1)**2)
		grad_h_1_b = Z1*(np.linalg.norm(U1)**2 + np.linalg.norm(Z1)**2)
		grad_h_2_a = U1
		grad_h_2_b = Z1
		temp_1 =  (0.25*((np.linalg.norm(U)**2 + np.linalg.norm(Z)**2)**2)) - (0.25*((np.linalg.norm(U1)**2 + np.linalg.norm(Z1)**2)**2))\
		-np.sum(np.multiply(U-U1,grad_h_1_a)) -np.sum(np.multiply(Z-Z1,grad_h_1_b)) 
		temp_2 = (0.5*((np.linalg.norm(U)**2 + np.linalg.norm(Z)**2))) - (0.5*((np.linalg.norm(U1)**2 + np.linalg.norm(Z1)**2)))\
		-np.sum(np.multiply(U-U1,grad_h_2_a)) -np.sum(np.multiply(Z-Z1,grad_h_2_b)) 
		if abs(temp_1) <= 1e-10:
			# to fix numerical issues
			temp_1 = 0
		if abs(temp_2) <= 1e-10:
			# to fix numerical issues
			temp_2 = 0

		if c_1*temp_1 + c_2*temp_2<0:
			# to fix numerical issues
			return 0

		return c_1*temp_1 + c_2*temp_2


