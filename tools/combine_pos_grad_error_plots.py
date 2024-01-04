import glob
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font',family='Times New Roman')
matplotlib.rcParams['font.size'] = 14
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

        fig = plt.figure()
        plt.semilogy(losses_baseline, label='Uncorrected')
        plt.semilogy(losses_calculated, label='Calculated')
        plt.xlabel('Epoch')
        plt.ylabel('Mean squared position gradient error (pixel)')
        plt.legend()
        plt.savefig(os.path.join(d, 'pos_error_comparison_baseline_calc.pdf'))
    else:
        print('No files found in {}.'.format(d))
