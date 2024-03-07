import glob
import os
import shutil
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
plt.viridis()


def parse_ptychoshelves_output(fname):
    res = {}
    f = h5py.File(fname, 'r')
    dset = f['object']
    obj = dset['real'][...] + 1j * dset['imag'][...]
    obj = np.transpose(obj)
    shrink_len = (20 / 1200 * np.array(obj.shape)).astype(int)
    obj = obj[shrink_len[0]:-shrink_len[0], shrink_len[1]:-shrink_len[1]]
    res['object'] = obj
    return res

def find_file_in_dir(dir, fname):
    for pardir, subdirs, files in os.walk(dir):
        if fname in files:
            return os.path.join(pardir, fname)
    return None


main_output_dir = 'outputs'
ptychoshelves_output_dir = 'outputs/ptychoshelves_outputs'

ps_output_dir_list = glob.glob(os.path.join(ptychoshelves_output_dir, 'test*'))

for ps_folder in ps_output_dir_list:
    scan_idx = int(re.findall('\d+', ps_folder)[-1])
    case_type = re.search('test(\d+)_(.+)', ps_folder).groups()[-1]
    ps_output_file = find_file_in_dir(ps_folder, 'Niter128.mat')
    print(ps_output_file)
    ps_output = parse_ptychoshelves_output(ps_output_file)
    obj = ps_output['object']

    dest_folder = os.path.join(main_output_dir, 'test{}'.format(scan_idx), 'ptychoshelves')
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.angle(obj), vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(dest_folder, 'recon_{}.pdf'.format(case_type)), transparent=True)
    plt.close()
