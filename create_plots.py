import numpy as np
import matplotlib as mpl
mpl.use('Agg')

pgf_with_custom_preamble = {
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    # "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
         "\\usepackage{unicode-math,amsmath,amssymb,amsthm}",  # unicode math setup
         ]
}
mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def makeup_for_plot(fig1):
	fig1.spines["top"].set_visible(False)    
	fig1.spines["bottom"].set_visible(True)    
	fig1.spines["right"].set_visible(False)    
	fig1.spines["left"].set_visible(True)
	fig1.get_xaxis().tick_bottom()    
	fig1.get_yaxis().tick_left()
	fig1.tick_params(axis="both", which="both", bottom="off", top="off",    
			labelbottom="on", left="off", right="off", labelleft="on",labelsize=12)
	grid_color = '#e3e3e3'
	grid_line_style= '--'
	fig1.grid(linestyle=grid_line_style,color=grid_color)
	return fig1

def do_tight_layout_for_fig(fig):
	fig.tight_layout()
	return fig

lr_vals = [0.1]

colors = ['red','green','c','m','y','orange','green','c','m','y','black','brown','orange','blue', 'black','blue','brown','red','orange','green','c','m','y','orange','green','c','m','y']


import argparse
parser = argparse.ArgumentParser(description='Plot Experiments')
parser.add_argument('--fun_num', '--fun_num', default=0,type=int,  dest='fun_num')

args = parser.parse_args()

fun_num = args.fun_num


my_markers = ['','','','','','','']


if fun_num == 0:
	# for L2 Regularization for U,Z and lam = 0
	files = {
		1: 'results/cocain_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5_uL_est_0.1_lL_est_0.1.txt',
		2: 'results/bpg_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5.txt',
		3: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.0_lam_val_0.0_rank_val_5.txt',
		4: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.2_lam_val_0.0_rank_val_5.txt',
		5: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.4_lam_val_0.0_rank_val_5.txt',
		6: 'results/bpg_mf_wb_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5_uL_est_0.1_lL_est_0.1.txt',
	}
if fun_num == 1:
	# for L2 Regularization for U,Z and lam = 1e-1
	files = {
		1: 'results/cocain_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5_uL_est_0.1_lL_est_0.1.txt',
		2: 'results/bpg_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5.txt',
		3: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.0_lam_val_0.1_rank_val_5.txt',
		4: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.2_lam_val_0.1_rank_val_5.txt',
		5: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.4_lam_val_0.1_rank_val_5.txt',
		6: 'results/bpg_mf_wb_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5_uL_est_0.1_lL_est_0.1.txt',

	}
if fun_num == 2:
	# for L1 Regularization for U,Z and 
	files = {
		1: 'results/cocain_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5_uL_est_0.1_lL_est_0.1.txt',
		2: 'results/bpg_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5.txt',
		3: 'results/palm_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_1_beta_0.0_lam_val_0.1_rank_val_5.txt',
		4: 'results/palm_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_1_beta_0.2_lam_val_0.1_rank_val_5.txt',
		5: 'results/palm_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_1_beta_0.4_lam_val_0.1_rank_val_5.txt',
		6: 'results/bpg_mf_wb_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5_uL_est_0.1_lL_est_0.1.txt',
	}


if fun_num == 3:
	# for L2 Regularization for U,Z and lam = 0 # exp_option_2
	files = {
		1: 'results/cocain_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5_exp_option_2_uL_est_0.1_lL_est_0.1.txt',
		2: 'results/bpg_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5_exp_option_2.txt',
		3: 'results/palm_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_1_beta_0.0_lam_val_0.0_rank_val_5_exp_option2.txt',
		4: 'results/palm_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_1_beta_0.2_lam_val_0.0_rank_val_5_exp_option2.txt',
		5: 'results/palm_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_1_beta_0.4_lam_val_0.0_rank_val_5_exp_option2.txt',
		6: 'results/bpg_mf_wb_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5_exp_option_2_uL_est_0.1_lL_est_0.1.txt',
	}
if fun_num == 4:
	# for L2 Regularization for U,Z and lam = 1e-1 # exp_option_2
	files = {
		1: 'results/cocain_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5_exp_option_2_uL_est_0.1_lL_est_0.1.txt',
		2: 'results/bpg_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5_exp_option_2.txt',
		3: 'results/palm_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_1_beta_0.0_lam_val_0.1_rank_val_5_exp_option2.txt',
		4: 'results/palm_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_1_beta_0.2_lam_val_0.1_rank_val_5_exp_option2.txt',
		5: 'results/palm_mf_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_1_beta_0.4_lam_val_0.1_rank_val_5_exp_option2.txt',
		6: 'results/bpg_mf_wb_fun_name_1_dataset_option_2_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5_exp_option_2_uL_est_0.1_lL_est_0.1.txt',
	}
if fun_num == 5:
	# for L1 Regularization for U,Z # exp_option_2
	files = {
		1: 'results/cocain_mf_fun_name_2_dataset_option_2_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5_exp_option_2_uL_est_0.1_lL_est_0.1.txt',
		2: 'results/bpg_mf_fun_name_2_dataset_option_2_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5_exp_option_2.txt',
		3: 'results/palm_mf_fun_name_2_dataset_option_2_abs_fun_num_2_breg_num_1_beta_0.0_lam_val_0.1_rank_val_5_exp_option2.txt',
		4: 'results/palm_mf_fun_name_2_dataset_option_2_abs_fun_num_2_breg_num_1_beta_0.2_lam_val_0.1_rank_val_5_exp_option2.txt',
		5: 'results/palm_mf_fun_name_2_dataset_option_2_abs_fun_num_2_breg_num_1_beta_0.4_lam_val_0.1_rank_val_5_exp_option2.txt',
		6: 'results/bpg_mf_wb_fun_name_2_dataset_option_2_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5_exp_option_2_uL_est_0.1_lL_est_0.1.txt',
	}


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1 = makeup_for_plot(ax1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2 = makeup_for_plot(ax2)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3 = makeup_for_plot(ax3)
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4 = makeup_for_plot(ax4)
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5 = makeup_for_plot(ax5)


label_font_size = 13
legend_font_size = 17
my_line_width = 2



labels_dict = {
	1: r"CoCaIn BPG-MF",
	6: r"BPG-MF-WB",
	2: r"BPG-MF",
	3: r"PALM",
	4: r"iPALM ($\beta = 0.2$)",
	5: r"iPALM ($\beta = 0.4$)",
}

nb_epoch = 1000
opt_vals= np.array([1,6,2,3,4,5])


color_count = 0

f_opt = 0


min_fun_val = np.inf
for i in opt_vals:
	file_name = files[i]
	try:
		best_train_objective_vals = np.loadtxt(file_name)[:,0]
		min_fun_val = np.nanmin([min_fun_val,np.min(best_train_objective_vals)])
		print(min_fun_val)
	except:
		pass

for i in opt_vals:
	file_name = files[i] 
	print(file_name)
	try:
		best_train_objective_vals = np.loadtxt(file_name)[:,0]
		best_time_vals = np.loadtxt(file_name)[:,1]
	except:
		best_train_objective_vals = np.loadtxt(file_name)

	ax4.loglog((np.arange(nb_epoch)+1),(best_train_objective_vals[:nb_epoch] - min_fun_val),\
				label=labels_dict[i],color=colors[color_count], linewidth=my_line_width,marker=my_markers[i-1])
	ax3.loglog((np.arange(nb_epoch)+1),(best_train_objective_vals[:nb_epoch] - min_fun_val)/(best_train_objective_vals[0] - min_fun_val),\
				label=labels_dict[i],color=colors[color_count], linewidth=my_line_width,marker=my_markers[i-1])
	ax5.loglog((np.arange(nb_epoch)+1),(best_train_objective_vals[:nb_epoch] - min_fun_val)/(best_train_objective_vals[0]),\
				label=labels_dict[i],color=colors[color_count], linewidth=my_line_width,marker=my_markers[i-1])

	ax1.plot((np.arange(nb_epoch)+1),(best_train_objective_vals[:nb_epoch]),\
			label=labels_dict[i],color=colors[color_count], linewidth=my_line_width,marker=my_markers[i-1])

	best_time_vals = best_time_vals - best_time_vals[0]
	best_time_vals[0]=1e-2
	temp_time_vals = np.cumsum(best_time_vals[:nb_epoch])
	ax2.loglog(temp_time_vals,  (best_train_objective_vals[:nb_epoch]),\
		label=labels_dict[i],color=colors[color_count], linewidth=my_line_width,marker=my_markers[i-1])
			

	color_count +=1

figure_name1 = 'figures/'+'func_vals_fun_num_'+str(fun_num)

# legends
ax1.legend(loc='upper right', fontsize=label_font_size)
ax2.legend(loc='upper right', fontsize=label_font_size)
ax3.legend(loc='upper right', fontsize=label_font_size)
ax4.legend(loc='upper right', fontsize=label_font_size)
ax5.legend(loc='upper right', fontsize=label_font_size)	


ax1.set_xlabel('Iterations (log scale)',fontsize=legend_font_size)
ax1.set_ylabel('Function value (log scale)',fontsize=legend_font_size)


do_tight_layout_for_fig(fig1)
fig1.savefig(figure_name1+'.png', dpi=fig1.dpi)
fig1.savefig(figure_name1+'.pdf', dpi=fig1.dpi)

ax2.set_xlabel('Time (log scale)',fontsize=legend_font_size)
ax2.set_ylabel('Function value (log scale)',fontsize=legend_font_size)

do_tight_layout_for_fig(fig2)
fig2.savefig(figure_name1+'_time_.png', dpi=fig2.dpi)
fig2.savefig(figure_name1+'_time_.pdf', dpi=fig2.dpi)

ax3.set_xlabel('Iterations (log scale)',fontsize=legend_font_size)
ax3.set_ylabel(r'$\frac{\Psi({\bf U^k},{\bf Z^k}) - v({\mathcal P})}{\Psi({\bf U^1},{\bf Z^1}) -v({\mathcal P})}$ (log scale)',fontsize=legend_font_size)

do_tight_layout_for_fig(fig3)
fig3.savefig(figure_name1+'_compare_optval1_.png', dpi=fig3.dpi)
fig3.savefig(figure_name1+'_compare_optval1_.pdf', dpi=fig3.dpi)

ax4.set_xlabel('Iterations (log scale)',fontsize=legend_font_size)
ax4.set_ylabel(r'$\Psi({\bf U^k},{\bf Z^k}) -v({\mathcal P})$ (log scale)',fontsize=legend_font_size)

do_tight_layout_for_fig(fig4)
fig4.savefig(figure_name1+'_compare_optval2_.png', dpi=fig4.dpi)
fig4.savefig(figure_name1+'_compare_optval2_.pdf', dpi=fig4.dpi)

ax5.set_xlabel('Iterations (log scale)',fontsize=legend_font_size)
ax5.set_ylabel(r'$\frac{\Psi({\bf U^k},{\bf Z^k}) -v({\mathcal P})}{\Psi({\bf U^1},{\bf Z^1})}$ (log scale)',fontsize=legend_font_size)

do_tight_layout_for_fig(fig5)
fig5.savefig(figure_name1+'_compare_optval3_.png', dpi=fig5.dpi)
fig5.savefig(figure_name1+'_compare_optval3_.pdf', dpi=fig5.dpi)

