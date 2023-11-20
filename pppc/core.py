import os.path
import time
import collections

import numpy as np
import skimage.registration
import skimage.feature
import tifffile
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import sklearn.neighbors

import pppc
from pppc.configs import InferenceConfig
from pppc.reconstructor import PyTorchReconstructor
from pppc.io import create_data_file_handle
from pppc.position_list import ProbePositionList
from pppc.registrator import Registrator
from pppc.util import class_timeit


class PtychoNNProbePositionCorrector:

    def __init__(self, config_dict: InferenceConfig):
        self.config_dict = config_dict
        self.ptycho_reconstructor = PyTorchReconstructor(self.config_dict)
        self.dp_data_fhdl = None
        self.orig_probe_positions = None
        self.new_probe_positions = None
        self.n_dps = 0
        self.debug = self.config_dict['debug']
        self.registrator = None
        self.method = self.config_dict['method']

    def build(self):
        self.ptycho_reconstructor.build()
        if not self.config_dict['dp_data_file_handle']:
            self.dp_data_fhdl = create_data_file_handle(self.config_dict['dp_data_path'])
        else:
            self.dp_data_fhdl = self.config_dict['dp_data_file_handle']
        self.n_dps = self.dp_data_fhdl.num_dps

        self.orig_probe_positions = ProbePositionList(position_list=np.zeros([self.n_dps, 2]))
        if not self.config_dict['probe_position_list']:
            if self.config_dict['probe_position_data_path']:
                self.orig_probe_positions = ProbePositionList(file_path=self.config_dict['probe_position_data_path'],
                                                              unit=self.config_dict['probe_position_data_unit'],
                                                              psize_nm=self.config_dict['pixel_size_nm'])
        else:
            self.orig_probe_positions = self.config_dict['probe_position_list']
        self.new_probe_positions = self.orig_probe_positions.copy_with_zeros()

        self.registrator = Registrator(method='error_map', max_shift=7)

    def run(self):
        if self.method == 'serial':
            self.run_probe_position_correction_serial()
        elif self.method == 'collective':
            self.run_probe_position_correction_collective()
        else:
            raise ValueError('Correction method {} is not supported. '.format(self.method))

    def reconstruct_dp(self, ind=None, dp=None):
        if dp is None:
            dp = self.dp_data_fhdl.get_dp_by_consecutive_index(ind)
        obj_amp, obj_ph = self.ptycho_reconstructor.batch_infer(dp[np.newaxis, :, :])
        if self.debug:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(dp)
            ax[1].imshow(obj_ph[0])
            ax[2].imshow(obj_amp[0])
            plt.show()
        obj_amp, obj_ph = self.crop_center([obj_amp, obj_ph])
        return obj_amp, obj_ph

    def crop_center(self, image_list):
        if self.config_dict['central_crop'] is None:
            return image_list
        crop_shape = self.config_dict['central_crop']
        for i in range(len(image_list)):
            orig_shape = image_list[i].shape[1:]
            start_point = [(orig_shape[j] - crop_shape[j]) // 2 for j in range(2)]
            new_image = image_list[i][:,
                                      start_point[0]:start_point[0] + crop_shape[0],
                                      start_point[1]:start_point[1] + crop_shape[1]]
            image_list[i] = new_image
        return image_list

    def run_probe_position_correction_serial(self):
        """
        Run serial mode probe position correction.
        """
        previous_obj = self.reconstruct_dp(0)[1][0]
        for ind in trange(1, self.n_dps):
            current_obj = self.reconstruct_dp(ind)[1][0]
            offset = self.registrator.run(previous_obj, current_obj)
            if self.debug:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(previous_obj)
                ax[1].imshow(current_obj)
                plt.suptitle('Index {} - {}'.format(ind, ind - 1))
                plt.show()
                plt.tight_layout()
                print('Offset: {}'.format(offset))
            self._update_probe_position_list(ind, offset)
            previous_obj = current_obj
        self.new_probe_positions.plot()

    def run_probe_position_correction_collective(self):
        """
        Run collective probe position correction.
        """
        a_mat = []
        b_vec = []
        # A hash table used to keep track of already registered DP pairs. One can then check whether a pair has been
        # registered with O(1) time.
        registered_pair_hash_table = collections.defaultdict(lambda: None)
        nn_engine = sklearn.neighbors.NearestNeighbors(n_neighbors=4)
        nn_engine.fit(self.orig_probe_positions.array)
        nn_dists, nn_inds = nn_engine.kneighbors(self.orig_probe_positions.array)
        for i_dp, this_orig_pos in enumerate(tqdm(self.orig_probe_positions.array)):
            this_neighbors_inds = nn_inds[i_dp, 1:]
            current_obj = self.reconstruct_dp(dp=self.dp_data_fhdl.get_dp_by_raw_index(i_dp))[1][0]
            for ind_neighbor in this_neighbors_inds:
                # Check whether the current pair of DPs have already been registered.
                if registered_pair_hash_table['{}_{}'.format(ind_neighbor, i_dp)] is not None or \
                        registered_pair_hash_table['{}_{}'.format(i_dp, ind_neighbor)]:
                    continue
                # Otherwise, run registration and record the values.
                registered_pair_hash_table['{}_{}'.format(ind_neighbor, i_dp)] = 1
                neighbor_obj = self.reconstruct_dp(dp=self.dp_data_fhdl.get_dp_by_raw_index(ind_neighbor))[1][0]
                offset = self.registrator.run(neighbor_obj, current_obj)
                b_vec.append(np.array(offset))
                a_mat.append(self._generate_amat_row(i_dp, ind_neighbor))
        a_mat = np.stack(a_mat)
        b_vec = np.stack(b_vec)
        self.new_probe_positions.array = np.linalg.pinv(a_mat) @ b_vec
        self.new_probe_positions.plot()

    def _generate_amat_row(self, this_ind, neightbor_ind):
        a = np.zeros(self.n_dps)
        a[this_ind] = 1
        a[neightbor_ind] = -1
        return a

    def _update_probe_position_list(self, ind, offset):
        raw_ind = self.dp_data_fhdl.get_actual_indices_for_consecutive_index(ind, ravel=True)
        prev_raw_ind = self.dp_data_fhdl.get_actual_indices_for_consecutive_index(ind - 1, ravel=True)
        self.new_probe_positions.array[raw_ind] = self.new_probe_positions.array[prev_raw_ind] + offset

