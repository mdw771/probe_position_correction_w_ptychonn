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
from pppc.configs import InferenceConfigDict
from pppc.reconstructor import PyTorchReconstructor, VirtualReconstructor
from pppc.io import create_data_file_handle
from pppc.position_list import ProbePositionList
from pppc.registrator import Registrator
from pppc.util import class_timeit


class PtychoNNProbePositionCorrector:

    def __init__(self, config_dict: InferenceConfigDict):
        self.config_dict = config_dict
        if self.config_dict['ptycho_reconstructor'] is None:
            self.ptycho_reconstructor = PyTorchReconstructor(self.config_dict)
        else:
            self.ptycho_reconstructor = self.config_dict['ptycho_reconstructor']
        self.dp_data_fhdl = None
        self.orig_probe_positions = None
        self.new_probe_positions = None
        self.n_dps = 0
        self.debug = self.config_dict['debug']
        self.registrator = None
        self.method = self.config_dict['method']
        self.lmbda_collective = 1e-3
        self.a_mat = []
        self.b_vec = []

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

        self.registrator = Registrator(method=self.config_dict['registration_method'],
                                       max_shift=self.config_dict['max_shift'])

    def run(self):
        if self.method == 'serial':
            self.run_probe_position_correction_serial()
        elif self.method == 'collective':
            self.run_probe_position_correction_collective()
        else:
            raise ValueError('Correction method {} is not supported. '.format(self.method))

    def reconstruct_dp(self, ind=None, dp=None):
        if isinstance(self.ptycho_reconstructor, VirtualReconstructor):
            assert ind is not None, 'Since the reconstructor is a VirtualReconstructor, a index must be provided. '
            obj_amp, obj_ph = self.ptycho_reconstructor.batch_infer([ind])
            dp = np.zeros(self.dp_data_fhdl.dp_shape)
        else:
            if dp is None:
                dp = self.dp_data_fhdl.get_dp_by_consecutive_index(ind)
            obj_amp, obj_ph = self.ptycho_reconstructor.batch_infer(dp[np.newaxis, :, :])
        if self.debug:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(dp)
            ax[0].set_title('DP')
            ax[1].imshow(obj_ph[0])
            ax[1].set_title('Recon phase')
            ax[2].imshow(obj_amp[0])
            ax[2].set_title('Recon mag')
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
        previous_offset = np.array([0, 0])
        for ind in trange(1, self.n_dps):
            current_obj = self.reconstruct_dp(ind)[1][0]
            offset = self.registrator.run(previous_obj, current_obj)
            if self.registrator.get_status() == self.registrator.get_status_code('drift'):
                offset = previous_offset
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
            previous_offset = self.update_previous_offset(previous_offset, offset, i_iteration=ind)
        self.new_probe_positions.plot()

    def update_previous_offset(self, previous_offset, current_offset, i_iteration):
        """
        Update the running average of previous offsets, so that it is correct by momentum.

        :param previous_offset: np.ndarray.
        :param current_offset: np.ndarray.
        :param i_iteration: int. Current iteration.
        :return: np.ndarray.
        """
        beta = 0.5
        if i_iteration <= 1:
            previous_offset = current_offset
        else:
            previous_offset = beta * previous_offset + (1 - beta) * current_offset
        return previous_offset

    def run_probe_position_correction_collective(self):
        """
        Run collective probe position correction.
        """
        self.a_mat = []
        self.b_vec = []
        # A hash table used to keep track of already registered DP pairs. One can then check whether a pair has been
        # registered with O(1) time.
        registered_pair_hash_table = collections.defaultdict(lambda: None)
        nn_engine = sklearn.neighbors.NearestNeighbors(n_neighbors=self.config_dict['num_neighbors_collective'] + 1)
        nn_engine.fit(self.orig_probe_positions.array)
        nn_dists, nn_inds = nn_engine.kneighbors(self.orig_probe_positions.array)
        for i_dp, this_orig_pos in enumerate(tqdm(self.orig_probe_positions.array)):
            this_neighbors_inds = nn_inds[i_dp, 1:]
            # `ind` is also provided in case the reconstructor is a `VirtualReconstructor`.
            current_obj = self.reconstruct_dp(dp=self.dp_data_fhdl.get_dp_by_raw_index(i_dp), ind=i_dp)[1][0]
            for ind_neighbor in this_neighbors_inds:
                # Check whether the current pair of DPs have already been registered.
                if registered_pair_hash_table['{}_{}'.format(ind_neighbor, i_dp)] is not None or \
                        registered_pair_hash_table['{}_{}'.format(i_dp, ind_neighbor)]:
                    continue
                # Otherwise, run registration and record the values.
                registered_pair_hash_table['{}_{}'.format(ind_neighbor, i_dp)] = 1
                neighbor_obj = self.reconstruct_dp(dp=self.dp_data_fhdl.get_dp_by_raw_index(ind_neighbor),
                                                   ind=ind_neighbor)[1][0]
                offset = self.registrator.run(neighbor_obj, current_obj)
                if self.debug:
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(neighbor_obj)
                    ax[1].imshow(current_obj)
                    plt.suptitle('Index {} - {}'.format(i_dp, ind_neighbor))
                    plt.show()
                    plt.tight_layout()
                    print('Offset: {}'.format(offset))
                # We want to be more strict with collective mode. If a result is less confident, just skip it.
                if self.registrator.get_status() != self.registrator.get_status_code('ok'):
                    continue
                else:
                    self.b_vec.append(np.array(offset))
                    self.a_mat.append(self._generate_amat_row(i_dp, ind_neighbor))
        self.a_mat = np.stack(self.a_mat)
        self.b_vec = np.stack(self.b_vec)
        self.solve_linear_system(mode='residue')
        self.new_probe_positions.plot()

    def solve_linear_system(self, mode='residue'):
        a_mat = self.a_mat
        b_vec = self.b_vec
        if mode == 'residue':
            x0_vec = self.orig_probe_positions.array
            b_vec_damped = a_mat.T @ b_vec - a_mat.T @ (a_mat @ x0_vec)
            a_damped = a_mat.T @ a_mat + self.lmbda_collective * np.eye(a_mat.shape[1])
            delta_x_vec = np.linalg.inv(a_damped) @ b_vec_damped
            self.new_probe_positions.array = self.orig_probe_positions.array + delta_x_vec
        else:
            self.new_probe_positions.array = np.linalg.pinv(a_mat) @ b_vec

    def _generate_amat_row(self, this_ind, neightbor_ind):
        a = np.zeros(self.n_dps)
        a[this_ind] = 1
        a[neightbor_ind] = -1
        return a

    def _update_probe_position_list(self, ind, offset):
        raw_ind = self.dp_data_fhdl.get_actual_indices_for_consecutive_index(ind, ravel=True)
        prev_raw_ind = self.dp_data_fhdl.get_actual_indices_for_consecutive_index(ind - 1, ravel=True)
        self.new_probe_positions.array[raw_ind] = self.new_probe_positions.array[prev_raw_ind] + offset

