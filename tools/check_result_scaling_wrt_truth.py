import os
import glob
import re

import numpy as np


dir_list_0 = glob.glob('test*')
dir_list = []
for d in dir_list_0:
    if os.path.isdir(d):
        dir_list.append(d)

print(dir_list)

for d in dir_list:
    scan_idx = int(re.findall('\d+', d)[-1])
    fname_true = os.path.join(os.pardir, 'data', 'pos{}.csv'.format(scan_idx))
    fname_calculated = os.path.join(d, 'calc_pos_{}_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.csv'.format(scan_idx))
    pos_list_true = np.genfromtxt(fname_true, delimiter=',')
    pos_list_calculated = np.genfromtxt(fname_calculated, delimiter=',')
    pos_list_true = pos_list_true - pos_list_true[0]
    pos_list_calculated = pos_list_calculated - pos_list_calculated[0]
    print('{}: {}'.format(scan_idx, np.mean(pos_list_true[1:] / pos_list_calculated[1:], axis=0)))
