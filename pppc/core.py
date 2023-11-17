import os.path
import time

import numpy as np
import skimage.registration
from tqdm import trange
import matplotlib.pyplot as plt

import pppc
from pppc.configs import InferenceConfig
from pppc.reconstructor import PyTorchReconstructor
from pppc.io import create_data_file_handle
from pppc.position_list import ProbePositionList
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

    def build(self):
        self.ptycho_reconstructor.build()
        if not self.config_dict['dp_data_file_handle']:
            self.dp_data_fhdl = create_data_file_handle(self.config_dict['dp_data_path'])
        else:
            self.dp_data_fhdl = self.config_dict['dp_data_file_handle']
        self.n_dps = self.dp_data_fhdl.num_dps

        if self.config_dict['probe_position_data_path'] is not None:
            self.orig_probe_positions = ProbePositionList(file_path=self.config_dict['probe_position_data_path'],
                                                          unit=self.config_dict['probe_position_data_unit'],
                                                          psize_nm=self.config_dict['pixel_size_nm'])
            self.new_probe_positions = self.orig_probe_positions.copy_with_zeros()
        else:
            self.new_probe_positions = ProbePositionList(position_list=np.zeros([self.n_dps, 2]))

    def run(self):
        self.run_probe_position_correction_serial()

    def reconstruct_dp(self, ind):
        dp = self.dp_data_fhdl.get_dp_by_index(ind)
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
            offset = self.run_registration(previous_obj, current_obj)
            if self.debug:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(previous_obj)
                ax[1].imshow(current_obj)
                plt.show()
                print('Offset: {}'.format(offset))
            self.update_probe_position_list(ind, offset)
            previous_obj = current_obj
        self.new_probe_positions.plot()

    def run_registration(self, previous, current):
        offset = skimage.registration.phase_cross_correlation(previous, current, upsample_factor=10)
        return offset[0]

    def update_probe_position_list(self, ind, offset):
        raw_ind = self.dp_data_fhdl.get_actual_indices_for_consecutive_index(ind, ravel=True)
        prev_raw_ind = self.dp_data_fhdl.get_actual_indices_for_consecutive_index(ind - 1, ravel=True)
        self.new_probe_positions.array[raw_ind] = self.new_probe_positions.array[prev_raw_ind] + offset
