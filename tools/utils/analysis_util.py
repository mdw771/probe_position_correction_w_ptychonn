import glob
import os.path
import re
import collections

import numpy as np
import sklearn.neighbors


def get_scan_index_from_dir_name(dirname):
    return int(re.findall(r'\d+', dirname)[-1])


def get_true_position(scan_idx, psize_nm=8):
    probe_pos_list_true = np.genfromtxt('../data/pos{}.csv'.format(scan_idx), delimiter=',').astype('float32')
    probe_pos_list_true /= (psize_nm * 1e-9)
    return probe_pos_list_true


def get_baseline_position(scan_idx, psize_nm=8):
    scaling_dict = collections.defaultdict(lambda: 1.0,
                                           {236: 0.5, 239: 0.5, 240: 0.25, 241: 0.25, 242: 0.25, 250: 0.5, 251: 0.5,
                                            252: 0.25, 253: 0.25})
    s = scaling_dict[scan_idx]
    probe_pos_list_baseline = np.genfromtxt('../data/pos221.csv', delimiter=',').astype('float32') / (psize_nm * 1e-9) * s  # Baseline
    return probe_pos_list_baseline


def get_actual_position(folder, name_pattern=None, psize_nm=8):
    if name_pattern is None:
        name_pattern = 'calc_pos_*_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.csv'
    fname = glob.glob(os.path.join(folder, name_pattern))[0]
    probe_pos_list = np.genfromtxt(fname, delimiter=',').astype('float32')
    probe_pos_list /= (psize_nm * 1e-9)
    return probe_pos_list


def calculate_rms_ppe(actual_pos, true_pos):
    actual_grad = actual_pos[1:] - actual_pos[:-1]
    true_grad = true_pos[1:] - true_pos[:-1]
    return np.sqrt(np.mean(np.sum((actual_grad - true_grad) ** 2, axis=1)))


def calculate_absolute_pos_error(actual_pos, true_pos, return_stats=True):
    actual_pos = actual_pos - np.mean(actual_pos, axis=0)
    true_pos = true_pos - np.mean(true_pos, axis=0)
    squared_errors = np.sum((actual_pos - true_pos) ** 2, axis=1)
    error_dists = np.sqrt(squared_errors)
    rms_error = np.sqrt(np.mean(squared_errors))
    if return_stats:
        s = {'mean_error_dist': np.mean(error_dists),
             'max_error_dist': np.max(error_dists)
            }
        return rms_error, s
    else:
        return rms_error


def calculate_rms_ppe_n(actual_pos, true_pos, n=3):
    nn_engine = sklearn.neighbors.NearestNeighbors(n_neighbors=n + 1)
    nn_engine.fit(true_pos)
    nn_dist, nn_inds = nn_engine.kneighbors(true_pos)
    nn_inds = nn_inds[:, 1:]
    nn_inds_ext = nn_inds.reshape(-1)
    actual_pos_neighbors = np.take(actual_pos, nn_inds_ext, axis=0)
    actual_pos_rep = np.repeat(actual_pos, n, axis=0)
    actual_offsets = actual_pos_neighbors - actual_pos_rep
    true_pos_neighbors = np.take(true_pos, nn_inds_ext, axis=0)
    true_pos_rep = np.repeat(true_pos, n, axis=0)
    true_offsets = true_pos_neighbors - true_pos_rep
    res = np.sqrt(np.mean(np.sum((actual_offsets - true_offsets) ** 2, axis=1)))
    return res
