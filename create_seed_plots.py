import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

pgf_with_custom_preamble = {
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    # "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
        # unicode math setup
        "\\usepackage{unicode-math,amsmath,amssymb,amsthm}",
    ]
}
mpl.rcParams.update(pgf_with_custom_preamble)


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
                     labelbottom="on", left="off", right="off", labelleft="on", labelsize=12)
    grid_color = '#e3e3e3'
    grid_line_style = '--'
    fig1.grid(linestyle=grid_line_style, color=grid_color)
    return fig1


def do_tight_layout_for_fig(fig):
    fig.tight_layout()
    return fig


lr_vals = [0.1]

colors = ['m', 'y', 'orange', 'red', 'green', 'c', ]

# 1 = red
# 6 = green
# 2 = c
# 3 = m
# 4 = y
# 5 = o

parser = argparse.ArgumentParser(description='Plot Experiments')
parser.add_argument('--fun_num', '--fun_num', default=0,
                    type=int,  dest='fun_num')

args = parser.parse_args()

fun_num = args.fun_num


my_markers = ['', '', '', '', '', '', '']


if fun_num == 0:
    # for L2 Regularization for U,Z and lam = 0
    files = {
        1: 'results/cocain_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5_uL_est_0.01_lL_est_0.01',
        2: 'results/bpg_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5',
        3: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.0_lam_val_0.0_rank_val_5',
        4: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.2_lam_val_0.0_rank_val_5',
        5: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.4_lam_val_0.0_rank_val_5',
        6: 'results/bpg_mf_wb_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.0_rank_val_5_uL_est_0.01_lL_est_0.01',
    }
if fun_num == 1:
    # for L2 Regularization for U,Z and lam = 1e-1
    files = {
        1: 'results/cocain_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5_uL_est_0.01_lL_est_0.01',
        2: 'results/bpg_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5',
        3: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.0_lam_val_0.1_rank_val_5',
        4: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.2_lam_val_0.1_rank_val_5',
        5: 'results/palm_mf_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_1_beta_0.4_lam_val_0.1_rank_val_5',
        6: 'results/bpg_mf_wb_fun_name_1_dataset_option_3_abs_fun_num_3_breg_num_2_lam_val_0.1_rank_val_5_uL_est_0.01_lL_est_0.01',

    }
if fun_num == 2:
    # for L1 Regularization for U,Z and
    files = {
        1: 'results/cocain_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5_uL_est_0.01_lL_est_0.01',
        2: 'results/bpg_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5',
        3: 'results/palm_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_1_beta_0.0_lam_val_0.1_rank_val_5',
        4: 'results/palm_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_1_beta_0.2_lam_val_0.1_rank_val_5',
        5: 'results/palm_mf_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_1_beta_0.4_lam_val_0.1_rank_val_5',
        6: 'results/bpg_mf_wb_fun_name_2_dataset_option_3_abs_fun_num_2_breg_num_2_lam_val_0.1_rank_val_5_uL_est_0.01_lL_est_0.01',
    }


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1 = makeup_for_plot(ax1)


label_font_size = 13
legend_font_size = 17
my_line_width = 2


num_seed_exps = 50

labels_dict = {
    1: r"CoCaIn BPG-MF",
    6: r"BPG-MF-WB",
    2: r"BPG-MF",
    3: r"PALM",
    4: r"iPALM ($\beta = 0.2$)",
    5: r"iPALM ($\beta = 0.4$)",
}

opt_vals = np.array([3, 4, 5, 1, 6, 2])

# ignoring 2 because BPG-MF always has bad performance as


color_count = 0

f_opt = 0

for i in opt_vals:
    file_name_temp = files[i]
    best_train_objective_vals = []
    for j in range(num_seed_exps):
        file_name = file_name_temp+'_seed_exp_num_'+str(j)+'.txt'
        best_train_objective_vals = best_train_objective_vals + \
            [np.loadtxt(file_name)[:, 0][-1]]

    ax1.hist((best_train_objective_vals[:num_seed_exps]), num_seed_exps,
             label=labels_dict[i], color=colors[color_count], width=0.5)

    color_count += 1

figure_name1 = 'seed_figures/'+'func_vals_fun_num_'+str(fun_num)

# legends
ax1.legend(loc='upper right', fontsize=label_font_size)

ax1.set_ylabel('Number of seeds', fontsize=legend_font_size)
ax1.set_xlabel('Function value', fontsize=legend_font_size)


do_tight_layout_for_fig(fig1)
fig1.savefig(figure_name1+'.png', dpi=fig1.dpi)
fig1.savefig(figure_name1+'.pdf', dpi=fig1.dpi)
