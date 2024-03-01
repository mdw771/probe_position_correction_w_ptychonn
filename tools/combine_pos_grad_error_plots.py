import glob
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
plt.viridis()

fname_baseline = 'pos_grad_error_history_baseline_pos_posCorr_1_clip_2.txt'
fname_calculated = 'pos_grad_error_history_calc_pos_posCorr_1_clip_2.txt'

dir_list_0 = glob.glob('test*')
dir_list = []
for d in dir_list_0:
    if os.path.isdir(d):
        dir_list.append(d)

print(dir_list)

for d in dir_list:
    if os.path.exists(os.path.join(d, fname_baseline)) and os.path.exists(os.path.join(d, fname_calculated)):
        losses_baseline = np.loadtxt(os.path.join(d, fname_baseline))
        losses_calculated = np.loadtxt(os.path.join(d, fname_calculated))

        # RMS
        losses_baseline = np.sqrt(losses_baseline)
        losses_calculated = np.sqrt(losses_calculated)

        fig, ax = plt.subplots(1, 1)
        plt.semilogy(losses_baseline, label='Nominal', linestyle='--')
        plt.semilogy(losses_calculated, label='Predicted')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('RMS-PPE (pixel)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'pos_error_comparison_baseline_calc.pdf'), transparent=True)
    else:
        print('No files found in {}.'.format(d))
