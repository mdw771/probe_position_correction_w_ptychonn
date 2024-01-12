import os
import glob
import re
import shutil
import collections

import numpy as np
import pandas as pd

def clean_data(arr):
    mask = arr < 0
    vals = arr[mask]
    vals = 32768 + (vals - -32768)
    arr[mask] = vals
    return arr

def load_and_preprocess_data(fname):
    f = np.load(fname)
    data = f['reciprocal']

    data = clean_data(data)
    data = data.astype('float32')
    if data.shape[-1] == 512:
        data = data[:, 128:-128, -128:128:-1]
    # data = np.fft.fftshift(data, axes=(-1, -2))  # This is needed for Tike; not sure for other libraries

    return data

def copy_and_rename(source_folder, dest_folder, source_fname, dest_fname=None):
    if dest_fname is None:
        dest_fname = source_fname
    shutil.copy(os.path.join(source_folder, source_fname), os.path.join(dest_folder, dest_fname))

def get_baseline_position(scan_idx):
    scaling_dict = collections.defaultdict(lambda: 1.0,
                                           {236: 0.5, 239: 0.5, 240: 0.25, 241: 0.25, 242: 0.25, 250: 0.5, 251: 0.5,
                                            252: 0.25, 253: 0.25})
    s = scaling_dict[scan_idx]
    probe_pos_list_baseline = np.genfromtxt('../data/pos221.csv', delimiter=',').astype('float32')
    probe_pos_list_baseline = probe_pos_list_baseline * s
    return probe_pos_list_baseline

def get_true_position(scan_idx):
    probe_pos_list_true = np.genfromtxt('../data/pos{}.csv'.format(scan_idx), delimiter=',').astype('float32')
    probe_pos_list_true = probe_pos_list_true
    return probe_pos_list_true

def save_position_list(pos, path):
    df = pd.DataFrame(pos)
    df.to_csv(path, header=False, index=False)

def copy_probe(path):
    probe_raw = np.load('../data/scan221_raw.npz')['probe']
    probe_raw = probe_raw[0] + 1j * probe_raw[1]
    np.save(path, probe_raw)

def copy_and_transform_data(scan_idx, source_folder, dest_folder):
    data = load_and_preprocess_data(os.path.join(source_folder, 'test{}.npz'.format(scan_idx)))
    np.save(os.path.join(dest_folder, 'diffraction.npy'), data)


top_dest_folder_name = 'packed_data_for_ptychoshelves'

if not os.path.exists(top_dest_folder_name):
    os.makedirs(top_dest_folder_name)

folder_list_0 = glob.glob('test*')
folder_list = []
for f in folder_list_0:
    if f != 'test257':
        folder_list.append(f)

for folder_name in folder_list:
    scan_idx = int(re.findall('\d+', folder_name)[-1])
    source_folder = folder_name

    for type in ['baseline', 'calculated', 'true']:
        dest_folder = os.path.join(top_dest_folder_name, folder_name + '_{}'.format(type))
        print('{} -> {}'.format(folder_name, dest_folder))
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        if type == 'calculated':
            copy_and_rename(source_folder, dest_folder,
                            'calc_pos_{}_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.csv'.format(scan_idx),
                            'positions.csv')
        elif type == 'baseline':
            save_position_list(get_baseline_position(scan_idx), os.path.join(dest_folder, 'positions.csv'))
        elif type == 'true':
            save_position_list(get_true_position(scan_idx), os.path.join(dest_folder, 'positions.csv'))
        copy_and_transform_data(scan_idx, '../data', dest_folder)
        copy_probe(os.path.join(dest_folder, 'probe.npy'))
        copy_and_rename('packed_data_for_ptychoshelves', dest_folder, 'parameters.toml')

