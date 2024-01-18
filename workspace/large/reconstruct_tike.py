import logging
import collections
import sys
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn/pppc')
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn')
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cupy
import scipy.ndimage as ndi
from skimage.transform import resize
import pandas as pd

import tike
import tike.ptycho
import tike.view

matplotlib.rc('font',family='Times New Roman')
matplotlib.rcParams['font.size'] = 14
plt.viridis()


class TikeReconstruction:

    def __init__(self, scan_idx, type, do_pos_corr=False, save_figs=False, record_intermediate_states=False):
        self.scan_idx = scan_idx
        self.type = type
        self.do_pos_corr = do_pos_corr
        self.save_figs = save_figs
        self.record_intermediate_states = record_intermediate_states
        self.probe_pos_history = []
        self.pos_error_history = []
        self.pos_grad_error_history = []
        self.recon_error_history = []
        self.scaling_dict = collections.defaultdict(lambda: 1.0,
                                                    {236: 0.5, 239: 0.5, 240: 0.25, 241: 0.25, 242: 0.25, 250: 0.5,
                                                     251: 0.5, 252: 0.25, 253: 0.25})
        self.psize_nm = 0
        self.data = None
        self.probe_pos_list = None
        self.probe = None
        self.psi = None

        self.position_options = None
        self.parameters = None
        self.result = None

        self.pos_corr_str = ''
        self.output_type_name_mapping = {'calculated': 'calc', 'true': 'true', 'baseline': 'baseline'}


    def build(self):
        np.random.seed(196)
        cupy.random.seed(196)
        self.build_data()
        self.build_options()
        self.build_parameters()
        self.build_strings()
        logging.basicConfig(level=logging.INFO)

    def build_data(self):
        f = np.load('data/test{}.npz'.format(scan_idx))
        data_raw = f['reciprocal']
        if self.type == 'calculated':
            probe_pos_list_raw = np.genfromtxt(
                'outputs/test{}/calc_pos_{}_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.csv'.format(scan_idx,
                                                                                                        scan_idx),
                delimiter=',').astype('float32')
            # probe_pos_list_raw *= 1.05
        elif self.type == 'true':
            probe_pos_list_raw = np.genfromtxt('data/pos{}.csv'.format(scan_idx), delimiter=',').astype('float32')
        elif self.type == 'baseline':
            s = self.scaling_dict[scan_idx]
            print('Baseline position scaled by {}.'.format(s))
            probe_pos_list_raw = np.genfromtxt('data/pos221.csv', delimiter=',').astype('float32') * s  # Baseline
        else:
            raise ValueError
        probe_raw = np.load('data/scan221_raw.npz')['probe']
        probe_raw = probe_raw[0] + 1j * probe_raw[1]
        probe_raw = probe_raw.astype('complex64')[None, :, :]

        print('data size: ', data_raw.shape)
        print('probe size: ', probe_raw.shape)

        # self.psize_nm = np.load('data/scan221_raw.npz')['pixelsize'] * 1e9
        self.psize_nm = 8

        # probe_pos_list = probe_pos_list_raw
        # probe_pos_list *= -1
        # probe_pos_list = probe_pos_list - np.min(probe_pos_list, axis=0)
        self.probe_pos_list = probe_pos_list_raw / (self.psize_nm * 1e-9)

        data = data_raw
        data = self.clean_data(data)
        data = data.astype('float32')
        # data = transform_data_for_ptychonn(data, target_shape=(128, 128), discard_len=(64, 64))
        # data = transform_data_for_ptychonn(data, target_shape=(128, 128), discard_len=(192, 192))
        if data.shape[-1] == 512:
            data = data[:, 128:-128, -128:128:-1]
        data = np.fft.fftshift(data, axes=(-1, -2))
        self.data = data

        probe = probe_raw
        # probe_real = transform_data_for_ptychonn(probe.real, target_shape=(256, 256), discard_len=None)
        # probe_imag = transform_data_for_ptychonn(probe.imag, target_shape=(256, 256), discard_len=None)
        # probe = probe_real + 1j * probe_imag
        probe = probe[np.newaxis, np.newaxis, :, :, :]
        probe = tike.ptycho.probe.add_modes_random_phase(probe, 1)
        self.probe = probe

        self.psi, self.probe_pos_list = tike.ptycho.object.get_padded_object(self.probe_pos_list, self.probe, extra=60)

    def build_options(self):
        self.position_options = None
        if self.do_pos_corr:
            self.position_options = tike.ptycho.PositionOptions(
                self.probe_pos_list,
                use_adaptive_moment=True,
                use_position_regularization=True,
                update_magnitude_limit=2,
                transform=tike.ptycho.position.AffineTransform(),
                optimize_scale=False  # Do NOT optimize global scale
            )

    def build_parameters(self):
        self.parameters = tike.ptycho.PtychoParameters(
            # Provide initial guesses for parameters that are updated
            probe=self.probe,
            scan=self.probe_pos_list,
            psi=self.psi,
            probe_options=tike.ptycho.ProbeOptions(),  # uses default settings for probe recovery
            object_options=tike.ptycho.ObjectOptions(
                # The object will be updated.
                use_adaptive_moment=True,  # smoothness constraint will use our provided setting
                # other object options will be default values
            ),
            position_options=self.position_options,
            algorithm_options=tike.ptycho.RpieOptions(
                num_iter=128 if not self.record_intermediate_states else 1,
                num_batch=7,
            ),
        )

    def build_strings(self):
        self.pos_corr_str = 'posCorr_1_clip_2' if self.do_pos_corr else 'posCorr_0'


    def run(self):
        if self.record_intermediate_states:
            # recon_true = np.angle(np.load('outputs/test{}/rpie_posCorr_0_pos_true.npy'.format(scan_idx)))
            # err_mask = get_error_calculation_mask(recon_true)
            for epoch in range(128):
                self.parameters = tike.ptycho.reconstruct(
                    data=self.data,
                    parameters=self.parameters,
                    num_gpu=1
                )
                current_probe_pos_list = self.parameters.scan
                self.probe_pos_history.append(current_probe_pos_list)
                # current_recon = np.angle(parameters.psi)
                # recon_error_history.append(calculate_recon_error(current_recon, recon_true, err_mask))

            self.result = self.parameters
        else:
            # returns an updated PtychoParameters object
            self.result = tike.ptycho.reconstruct(
                data=self.data,
                parameters=self.parameters,
                num_gpu=1
            )

    def visualize(self):
        self.plot_loss()
        self.plot_reconstruction()
        self.plot_path_comparison()
        # self.plot_probe_position_error_history()
        self.plot_probe_position_grad_error_history()
        # self.plot_reconstruction_error_history()
        self.save_refined_positions()

    def plot_loss(self):
        fig = plt.figure()
        avg_cost = np.mean(np.stack(self.result.algorithm_options.costs, axis=0), axis=1)
        plt.semilogy(avg_cost)
        if self.save_figs:
            if self.type == 'calculated':
                fig.savefig(os.path.join('outputs/test{}/loss_history_calc_pos_{}.pdf'.format(self.scan_idx, self.pos_corr_str)))
                np.savetxt(os.path.join('outputs/test{}/loss_history_calc_pos_{}.txt'.format(self.scan_idx, self.pos_corr_str)),
                           avg_cost)
            elif self.type == 'true':
                fig.savefig(os.path.join('outputs/test{}/loss_history_true_pos.pdf'.format(self.scan_idx)))
                np.savetxt(os.path.join('outputs/test{}/loss_history_true_pos.txt'.format(self.scan_idx)), avg_cost)
            elif self.type == 'baseline':
                fig.savefig(
                    os.path.join('outputs/test{}/loss_history_baseline_pos_{}.pdf'.format(self.scan_idx, self.pos_corr_str)))
                np.savetxt(
                    os.path.join('outputs/test{}/loss_history_baseline_pos_{}.txt'.format(self.scan_idx, self.pos_corr_str)),
                    avg_cost)
            else:
                raise ValueError
        else:
            plt.show()

    def plot_reconstruction(self):
        fig = plt.figure()
        im = plt.imshow(np.angle(self.result.psi), vmin=-1, vmax=1)
        plt.colorbar(im)
        if self.save_figs:
            if self.type == 'calculated':
                fig.savefig(
                    'outputs/test{}/rpie_{}_pos_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.pdf'.format(self.scan_idx,
                                                                                                            self.pos_corr_str))
                np.save(
                    'outputs/test{}/rpie_{}_pos_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.npy'.format(self.scan_idx,
                                                                                                            self.pos_corr_str),
                    self.result.psi)
            elif self.type == 'true':
                fig.savefig('outputs/test{}/rpie_posCorr_0_pos_true.pdf'.format(self.scan_idx))
                np.save('outputs/test{}/rpie_posCorr_0_pos_true.npy'.format(self.scan_idx), self.result.psi)
            elif self.type == 'baseline':
                fig.savefig('outputs/test{}/rpie_{}_pos_baseline.pdf'.format(self.scan_idx, self.pos_corr_str))
                np.save('outputs/test{}/rpie_{}_pos_baseline.npy'.format(self.scan_idx, self.pos_corr_str), self.result.psi)
            else:
                raise ValueError
        else:
            plt.show()

    def plot_path_comparison(self):
        if not self.type == 'calculated':
            return
        probe_pos_list_calc = self.probe_pos_list
        probe_pos_list_true = self.get_true_positions()
        probe_pos_list_refined = self.result.scan
        probe_pos_list_calc -= np.mean(probe_pos_list_calc, axis=0)
        probe_pos_list_true -= np.mean(probe_pos_list_true, axis=0)
        probe_pos_list_refined -= np.mean(probe_pos_list_refined, axis=0)
        # probe_pos_list_raw *= 1.05
        plt.figure()
        plt.scatter(probe_pos_list_calc[:, 1], probe_pos_list_calc[:, 0], s=1)
        plt.scatter(probe_pos_list_true[:, 1], probe_pos_list_true[:, 0], s=1)
        plt.scatter(probe_pos_list_refined[:, 1], probe_pos_list_refined[:, 0], s=1)
        plt.plot(probe_pos_list_calc[:, 1], probe_pos_list_calc[:, 0], linewidth=0.5, alpha=0.6, label='Calculated')
        plt.plot(probe_pos_list_true[:, 1], probe_pos_list_true[:, 0], linewidth=0.5, alpha=0.6, label='True')
        plt.plot(probe_pos_list_refined[:, 1], probe_pos_list_refined[:, 0], linewidth=0.5, alpha=0.6, label='Refined')
        plt.legend()
        if self.save_figs and self.do_pos_corr:
            plt.savefig(
                'outputs/test{}/comparison_path_plot_true_calc_refined_clip_2_collective_iter_2_nn_12_sw_1e-3_1e-2.pdf'.format(
                    self.scan_idx))
        elif not self.save_figs:
            plt.show()

    def save_refined_positions(self):
        if self.do_pos_corr:
            probe_pos_list_refined = self.result.scan
            probe_pos_list_refined -= np.mean(probe_pos_list_refined, axis=0)
            type_name = self.output_type_name_mapping[self.type]
            self.save_pos_to_csv(probe_pos_list_refined * (self.psize_nm * 1e-9),
                                 os.path.join('outputs', 'test{}'.format(self.scan_idx), 'refined_{}_pos.csv'.format(type_name)))

    def plot_probe_position_grad_error_history(self):
        if self.type != 'true' and self.do_pos_corr and self.record_intermediate_states:
            self.pos_grad_error_history = []
            probe_pos_list_true = self.get_true_positions()
            for i_epoch, this_pos_list in enumerate(self.probe_pos_history):
                self.pos_grad_error_history.append(self.calculate_pos_grad_error(this_pos_list, probe_pos_list_true))
            if self.save_figs:
                type_name = self.output_type_name_mapping[self.type]
                np.savetxt(os.path.join('outputs/test{}/pos_grad_error_history_{}_pos_{}.txt'.format(
                    self.scan_idx, type_name, self.pos_corr_str)),
                    self.pos_grad_error_history)

    def plot_probe_position_error_history(self):
        if self.type != 'true' and self.do_pos_corr and self.record_intermediate_states:
            self.pos_error_history = []
            probe_pos_list_true = self.get_true_positions()
            for i_epoch, this_pos_list in enumerate(self.probe_pos_history):
                this_pos_list = this_pos_list - np.mean(this_pos_list, axis=0)
                self.pos_error_history.append(np.mean(np.sum((this_pos_list - probe_pos_list_true) ** 2, axis=1)))
            if self.save_figs:
                type_name = self.output_type_name_mapping[type]
                np.savetxt(os.path.join('outputs/test{}/pos_error_history_{}_pos_{}.txt'.format(
                    self.scan_idx, type_name, self.pos_corr_str)),
                    self.pos_error_history)

    def plot_probe_position_history(self):
        pos_list_true = self.get_true_positions()
        plt.figure()
        plt.scatter(pos_list_true[:, 1], pos_list_true[:, 0])
        plt.plot(pos_list_true[:, 1], pos_list_true[:, 0])
        for i_epoch in range(len(self.probe_pos_history))[::32]:
            this_pos_list = self.probe_pos_history[i_epoch]
            plt.scatter(this_pos_list[:, 1], this_pos_list[:, 0], alpha=0.5)
            plt.plot(this_pos_list[:, 1], this_pos_list[:, 0], alpha=0.5)
        plt.show()


    def plot_reconstruction_error_history(self):
        if self.type != 'true' and self.do_pos_corr and self.record_intermediate_states:
            plt.figure()
            plt.semilogy(self.recon_error_history)
            plt.xlabel('Epoch')
            plt.ylabel('Reconstruction error')
            if self.save_figs:
                type_name = self.output_type_name_mapping[type]
                plt.savefig(os.path.join('outputs', 'test{}', 'recon_error_history_{}_pos_{}.pdf').format(
                    self.scan_idx, type_name, self.pos_corr_str))
                np.savetxt(os.path.join('outputs', 'test{}', 'recon_error_history_{}_pos_{}.txt').format(
                    self.scan_idx, type_name, self.pos_corr_str), self.recon_error_history)
            else:
                plt.show()

    def get_true_positions(self):
        probe_pos_list_true = np.genfromtxt('data/pos{}.csv'.format(self.scan_idx), delimiter=',').astype('float32')
        probe_pos_list_true = probe_pos_list_true / (self.psize_nm * 1e-9)
        return probe_pos_list_true

    def save_pos_to_csv(self, arr, filename):
        df = pd.DataFrame(arr)
        df.to_csv(filename, header=False, index=False)

    @staticmethod
    def clean_data(arr):
        mask = arr < 0
        vals = arr[mask]
        vals = 32768 + (vals - -32768)
        arr[mask] = vals
        return arr

    @staticmethod
    def get_error_calculation_mask(obj_true):
        mask = np.abs(obj_true) > 0
        mask = ndi.binary_closing(mask, np.ones([5, 5]))
        mask = ndi.binary_erosion(mask, np.ones([5, 5]), iterations=10)
        return mask

    @staticmethod
    def calculate_recon_error(obj_recon, obj_true, mask):
        return np.mean((obj_recon[mask] - obj_true[mask]) ** 2)

    @staticmethod
    def calculate_pos_grad_error(pos1, pos2):
        g1 = pos1[1:] - pos1[:-1]
        g2 = pos2[1:] - pos2[:-1]
        g_mse = np.mean(np.sum((g2 - g1) ** 2, axis=1))
        return g_mse

scan_indices = [233, 234, 235, 236, 239, 240, 241, 242, 244, 245, 246, 247, 250, 251, 252, 253]
# scan_indices = [246,]
# config_list = [('true', 0), ('baseline', 0), ('baseline', 1), ('calculated', 0), ('calculated', 1)]
config_list = [('baseline', 1), ('calculated', 1)]

for scan_idx in scan_indices:
    for type, pos_corr in config_list:
        reconstructor = TikeReconstruction(scan_idx=scan_idx,
                                           type=type,
                                           do_pos_corr=bool(pos_corr),
                                           save_figs=True,
                                           record_intermediate_states=True)
        reconstructor.build()
        reconstructor.run()
        reconstructor.visualize()
