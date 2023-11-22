import os
import itertools
import copy
import re

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from numpy.fft import fftn, fftshift
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from .model import *
from pppc.configs import TrainingConfigDict
from pppc.message_logger import logger
from pppc.util import set_all_random_seeds, get_gpu_memory


class PtychoNNTrainer:

    def __init__(self, config_dict: TrainingConfigDict):
        self.config_dict = config_dict
        self.dataset = self.config_dict['dataset']
        assert isinstance(self.dataset, Dataset)
        self.training_dataset = None
        self.validation_dataset = None
        self.validation_ratio = self.config_dict['validation_ratio']
        self.model = None
        self.model_class_handle = None
        self.training_dataloader = None
        self.validation_dataloader = None
        self.num_gpus = PtychoNNTrainer.get_num_devices()
        self.device = PtychoNNTrainer.get_device()
        self.batch_size = self.config_dict['batch_size_per_process'] * self.num_gpus
        self.learning_rate = self.config_dict['learning_rate_per_process'] * self.num_gpus
        self.num_epochs = self.config_dict['num_epochs']
        self.optimizer = None
        self.scheduler = None
        self.metric_dict = {}
        self.loss_criterion = None
        self.iterations_per_epoch = 0
        self.current_epoch = 0
        self.prediction_type = {'mag': True, 'phase': True}

        self.debug = self.config_dict['debug']
        self.verbose = True

    @staticmethod
    def get_num_devices():
        if not torch.cuda.is_available():
            return 1
        else:
            return torch.cuda.device_count()

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build(self):
        # When default device is set to `cuda`, DataLoader with `shuffle=True` would crash when yielding due to an
        # internal bug of PyTorch. Therefore, we set default device to `cpu` here and manually assign device to objects.
        torch.set_default_device('cpu')

        self.build_split_datasets()
        self.training_dataloader = DataLoader(self.training_dataset, shuffle=True, batch_size=self.batch_size)
        self.validation_dataloader = DataLoader(self.validation_dataset, shuffle=False, batch_size=self.batch_size)

        self.build_model()
        self.build_prediction_type()
        self.build_optimizer()
        self.build_scheduler()

        self.loss_criterion = nn.L1Loss()
        self.initialize_metric_dict()

    def run_training(self):
        for self.current_epoch in range(self.num_epochs):
            # Set model to train mode and run training
            self.model.train()
            self.run_trainig_epoch()

            # Switch model to eval mode and run validation
            self.model.eval()
            self.run_validation()

            if self.verbose:
                logger.info('Epoch: %d | FT  | Train Loss: %.5f | Val Loss: %.5f' % (
                    self.current_epoch, self.metric_dict['losses'][-1][0], self.metric_dict['val_losses'][-1][0]))
                logger.info('Epoch: %d | Amp | Train Loss: %.4f | Val Loss: %.4f' % (
                    self.current_epoch, self.metric_dict['losses'][-1][1], self.metric_dict['val_losses'][-1][1]))
                logger.info('Epoch: %d | Ph  | Train Loss: %.3f | Val Loss: %.3f' % (
                    self.current_epoch, self.metric_dict['losses'][-1][2], self.metric_dict['val_losses'][-1][2]))
                logger.info('Epoch: %d | Ending LR: %.6f ' % (self.current_epoch, self.metric_dict['lrs'][-1][0]))
        self.update_saved_model(filename='final_model.pth')

    def run_trainig_epoch(self):
        tot_loss = 0.0
        loss_amp = 0.0
        loss_ph = 0.0
        n_batches = 0
        for i, (ft_images, amps, phs) in enumerate(tqdm(self.training_dataloader, disable=(not self.verbose))):
            ft_images = ft_images.to(self.device)  # Move everything to device
            amps = amps.to(self.device)
            phs = phs.to(self.device)

            pred_amps, pred_phs = self.model(ft_images)  # Forward pass
            pred_amps, pred_phs = self.post_process(pred_amps, pred_phs)

            if self.debug:
                self.plot_images([ft_images[0, 0], amps[0, 0], phs[0, 0], pred_amps[0, 0], pred_phs[0, 0]])

            # Compute losses
            # Monitor amplitude loss
            loss_a = self.loss_criterion(pred_amps, amps) if self.prediction_type['mag'] else 0.0
            # Monitor phase loss but only within support (which may not be same as true amp)
            loss_p = self.loss_criterion(pred_phs, phs) if self.prediction_type['phase'] else 0.0
            loss = loss_a + loss_p  # Use equiweighted amps and phase

            # Zero current grads and do backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tot_loss += loss.detach().item()
            loss_amp += (loss_a.detach().item() if self.prediction_type['mag'] else 0.0)
            loss_ph += (loss_p.detach().item() if self.prediction_type['phase'] else 0.0)

            # Update the LR according to the schedule -- CyclicLR updates each batch
            self.scheduler.step()
            self.metric_dict['lrs'].append(self.scheduler.get_last_lr())

            n_batches += 1
        # Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
        self.metric_dict['losses'].append([tot_loss / n_batches,
                                           loss_amp / n_batches,
                                           loss_ph / n_batches])

    def run_validation(self):
        tot_val_loss = 0.0
        val_loss_amp = 0.0
        val_loss_ph = 0.0
        n_batches = 0
        for j, (ft_images, amps, phs) in enumerate(self.validation_dataloader):
            ft_images = ft_images.to(self.device)
            amps = amps.to(self.device)
            phs = phs.to(self.device)
            pred_amps, pred_phs = self.model(ft_images)  # Forward pass
            pred_amps, pred_phs = self.post_process(pred_amps, pred_phs, for_plotting=False)

            val_loss_a = self.loss_criterion(pred_amps, amps) if self.prediction_type['mag'] else 0.0
            val_loss_p = self.loss_criterion(pred_phs, phs) if self.prediction_type['phase'] else 0.0
            val_loss = val_loss_a + val_loss_p

            tot_val_loss += val_loss.detach().item()
            val_loss_amp += (val_loss_a.detach().item() if self.prediction_type['mag'] else 0.0)
            val_loss_ph += (val_loss_p.detach().item() if self.prediction_type['phase'] else 0.0)

            n_batches += 1
        self.metric_dict['val_losses'].append([tot_val_loss / n_batches,
                                               val_loss_amp / n_batches,
                                               val_loss_ph / n_batches])

        # Update saved model if val loss is lower
        if tot_val_loss / n_batches < self.metric_dict['best_val_loss']:
            logger.info("Saving improved model after Val Loss improved from %.5f to %.5f" % (
                self.metric_dict['best_val_loss'], tot_val_loss / n_batches))
            self.metric_dict['best_val_loss'] = tot_val_loss / n_batches
            self.metric_dict['best_val_loss_amp'] = val_loss_amp / n_batches
            self.metric_dict['best_val_loss_ph'] = val_loss_ph / n_batches
            self.metric_dict['epoch_best_val_loss'] = self.current_epoch
            self.update_saved_model(filename='best_model.pth')

    def initialize_metric_dict(self):
        self.metric_dict['losses'] = []
        self.metric_dict['val_losses'] = []
        self.metric_dict['lrs'] = []
        self.metric_dict['best_val_loss'] = np.inf
        self.metric_dict['best_val_loss_amp'] = np.inf
        self.metric_dict['best_val_loss_ph'] = np.inf
        self.metric_dict['epoch_best_val_loss'] = 0

    def build_split_datasets(self):
        train_idx, val_idx = train_test_split(list(range(len(self.dataset))), test_size=self.validation_ratio)
        self.training_dataset = Subset(self.dataset, train_idx)
        self.validation_dataset = Subset(self.dataset, val_idx)

    def build_optimizer(self):
        if isinstance(self.config_dict['optimizer'], str):
            if self.config_dict['optimizer'] == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = self.config_dict['optimizer'](self.model.parameters(), lr=self.learning_rate)

    def build_scheduler(self):
        self.iterations_per_epoch = (len(self.training_dataset) - len(self.validation_dataset)) / self.batch_size
        self.iterations_per_epoch = np.floor(self.iterations_per_epoch) + 1
        step_size = 6 * self.iterations_per_epoch
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.learning_rate / 10,
                                                           max_lr=self.learning_rate, step_size_up=step_size,
                                                           cycle_momentum=False, mode='triangular2')

    def build_model(self):
        if self.config_dict['model'] is None:
            self.model = PtychoNNModel()
        elif isinstance(self.config_dict['model'], nn.Module):
            self.model = self.config_dict['model']
        elif isinstance(self.config_dict['model'], (tuple, list)):
            self.model = self.config_dict['model'][0](**self.config_dict['model'][1])
        if self.device.type == 'cuda' and self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            logger.info('Using DataParallel with {} devices.'.format(self.num_gpus))
        self.model.to(self.device)

    def build_prediction_type(self):
        # Run some test data through the model.
        data, _, _ = self.dataset.__getitem__(0)
        data.to(self.device)
        pred_amp, pred_phase = self.model(data)
        if pred_amp is None:
            self.prediction_type['mag'] = False
        if pred_phase is None:
            self.prediction_type['phase'] = False

    def update_saved_model(self, filename='best_model.pth'):
        """
        Updates saved model if validation loss is minimum.
        """
        path = self.config_dict['model_save_dir']
        if not os.path.isdir(path):
            os.mkdir(path)
        dest_path = os.path.join(path, filename)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        torch.save(self.model.module.state_dict(), dest_path)

    def plot_training_history(self):
        self.plot_lr_history()
        self.plot_loss_history()
        plt.show()

    def plot_lr_history(self):
        batches = np.linspace(0, len(self.metric_dict['lrs']), len(self.metric_dict['lrs']) + 1)
        epoch_list = batches / self.iterations_per_epoch
        fig, ax = plt.subplots(1, 1)
        ax.plot(epoch_list[1:], self.metric_dict['lrs'], 'C3-')
        plt.grid()
        ax.set_ylabel("Learning rate")
        ax.set_xlabel("Epoch")

    def plot_loss_history(self):
        losses_arr = np.array(self.metric_dict['losses'])
        val_losses_arr = np.array(self.metric_dict['val_losses'])
        fig, ax = plt.subplots(3, sharex=True, figsize=(15, 8))
        ax[0].plot(losses_arr[:, 0], 'C3o-', label="Total Train loss")
        ax[0].plot(val_losses_arr[:, 0], 'C0o-', label="Total Val loss")
        ax[0].set(ylabel='Loss')
        ax[0].grid()
        ax[0].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
        ax[1].plot(losses_arr[:, 1], 'C3o-', label="Train Amp loss")
        ax[1].plot(val_losses_arr[:, 1], 'C0o-', label="Val Amp loss")
        ax[1].set(ylabel='Loss')
        ax[1].grid()
        ax[1].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
        ax[2].plot(losses_arr[:, 2], 'C3o-', label="Train Ph loss")
        ax[2].plot(val_losses_arr[:, 2], 'C0o-', label="Val Ph loss")
        ax[2].set(ylabel='Loss')
        ax[2].grid()
        ax[2].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        plt.tight_layout()
        plt.xlabel("Epochs")

    def plot_images(self, image_list):
        fig, ax = plt.subplots(1, len(image_list))
        for i, img in enumerate(image_list):
            try:
                ax[i].imshow(img)
            except TypeError:
                try:
                    img = img.cpu().numpy()
                    ax[i].imshow(img)
                except:
                    img = img.detach().cpu().numpy()
                    ax[i].imshow(img)
        plt.show()

    def run_testing(self, ind_list, dataset='train'):
        self.model.eval()
        dset = self.training_dataset if dataset == 'train' else self.validation_dataset
        dp_list, true_amp, true_ph = dset.__getitems__(ind_list)
        dp_list.to(self.device)
        pred_amp, pred_ph = self.model(dp_list)
        pred_amp, pred_ph = self.post_process(pred_amp, pred_ph, for_plotting=True)
        self.plot_test_results(dp_list, pred_amp, pred_ph, true_amp, true_ph)

    def post_process(self, pred_amps, pred_phs, for_plotting=False):
        pred_amps = torch.ones_like(pred_phs) if pred_amps is None else pred_amps
        pred_phs = torch.zeros_like(pred_amps) if pred_phs is None else pred_phs
        if not for_plotting:
            return pred_amps, pred_phs
        return_list = []
        for img in [pred_amps, pred_phs]:
            if img.requires_grad:
                img = img.detach()
            img = img.cpu().numpy()
            return_list.append(img)
        return return_list

    def plot_test_results(self, dp_list, pred_amp_list, pred_ph_list, true_amp_list, true_ph_list):
        n = dp_list.shape[0]
        fig, ax = plt.subplots(7, n, figsize=(4 * n, 15))
        plt.gcf().text(0.02, 0.85, "Input", fontsize=20)
        plt.gcf().text(0.02, 0.72, "True I", fontsize=20)
        plt.gcf().text(0.02, 0.6, "Predicted I", fontsize=20)
        plt.gcf().text(0.02, 0.5, "Difference I", fontsize=20)
        plt.gcf().text(0.02, 0.4, "True Phi", fontsize=20)
        plt.gcf().text(0.02, 0.27, "Predicted Phi", fontsize=20)
        plt.gcf().text(0.02, 0.17, "Difference Phi", fontsize=20)
        for i in range(n):
            self._plot_test_result(ax[0, i], np.log10(dp_list[i, 0, :, :] + 1))
            self._plot_test_result(ax[1, i], true_amp_list[i, 0, :, :])
            self._plot_test_result(ax[2, i], pred_amp_list[i, 0, :, :])
            self._plot_test_result(ax[3, i], true_amp_list[i, 0, :, :] - pred_amp_list[i, 0, :, :])
            self._plot_test_result(ax[4, i], true_ph_list[i, 0, :, :])
            self._plot_test_result(ax[5, i], pred_ph_list[i, 0, :, :])
            self._plot_test_result(ax[6, i], true_ph_list[i, 0, :, :] - pred_ph_list[i, 0, :, :])
        plt.show()

    def _plot_test_result(self, ax, img):
        im = ax.imshow(img)
        plt.colorbar(im, ax=ax, format='%.2f')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def load_checkpoint(self):
        state_dict = torch.load(self.config_dict['model_path'])

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)


class PtychoNNHyperparameterScanner:
    def __init__(self, scan_params_dict: dict, base_config_dict: TrainingConfigDict, keep_models_in_memory=False):
        """
        Hyperparameter scanner.

        :param scan_params_dict: dict. A dictionary of the parameters to be scanned. The keys of the dictionary
                                 should be from `TrainingConfigDict`, and the value should be a list of values to test.
        :param base_config_dict: TrainingConficDict. A baseline config dictionary.
        """
        self.scan_params_dict = scan_params_dict
        self.result_table = None
        self.n_params = len(self.scan_params_dict)
        self.metric_names = ['min_val_loss_total', 'min_val_loss_mag', 'min_val_loss_phase', 'epoch_min_val_loss']
        self.param_comb_list = None
        self.base_config_dict = base_config_dict
        # Might need to be a copy. Deepcopy is currently not done because of the H5py object in `dataset`.
        self.config_dict = self.base_config_dict
        self.trainer_list = []
        self.model_save_dir_prefix = self.base_config_dict['model_save_dir']
        self.keep_models_in_memory = keep_models_in_memory
        self.verbose = True

    def build_result_table(self):
        self.param_comb_list = list(itertools.product(*self.scan_params_dict.values()))
        dframe_dict = {}
        for i_param, param in enumerate(self.scan_params_dict.keys()):
            dframe_dict[param] = []
            for i_comb in range(len(self.param_comb_list)):
                v = self.param_comb_list[i_comb][i_param]
                v = self.convert_item_to_be_dataframe_compatible(v)
                dframe_dict[param].append(v)
        for metric in self.metric_names:
            dframe_dict[metric] = [0.0] * len(self.param_comb_list)
        self.result_table = pd.DataFrame(dframe_dict)

    def build(self, seed=123):
        if seed is not None:
            set_all_random_seeds(seed)
        self.build_result_table()

    def convert_item_to_be_dataframe_compatible(self, v):
        if isinstance(v, nn.Module):
            nv = v._get_name()
        elif isinstance(v, (tuple, list)) and issubclass(v[0], nn.Module):
            nv = v[0].__name__
            if len(v[1]) > 0:
                nv += '_' + self.convert_dict_to_string(v[1])
        else:
            nv = v
        return nv

    def modify_condig_dict(self, param_dict):
        for i, param in enumerate(param_dict.keys()):
            # if param == 'model':
            #     # For testing different models, the input in `scan_params_dict['model']` is supposed to be a list of
            #     # 2-tuples, where the first element is the class handle of the model, and the second element is a
            #     # dictionary of keyword arguments in the constructor of that class.
            #     self.config_dict[param] = param_dict[param][0](**param_dict[param][1])
            # else:
            self.config_dict[param] = param_dict[param]
        # Update save path.
        appendix = self.convert_dict_to_string(param_dict)
        self.config_dict['model_save_dir'] = self.model_save_dir_prefix + '_' + appendix

    @staticmethod
    def convert_string_to_camel_case(s):
        s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
        return ''.join([s[0].lower(), s[1:]])

    def convert_dict_to_string(self, d):
        s = ''
        for i, (k, v) in enumerate(d.items()):
            s += self.convert_string_to_camel_case(k)
            s += '_'
            s += str(self.convert_item_to_be_dataframe_compatible(v))
            if i < len(d) - 1:
                s += '_'
        return s

    def create_param_dict(self, config_val_list: list):
        d = {}
        for i in range(len(config_val_list)):
            param_name = list(self.scan_params_dict.keys())[i]
            d[param_name] = config_val_list[i]
        return d

    def run(self):
        for i_comb in tqdm(range(len(self.param_comb_list))):
            param_dict = self.create_param_dict(self.param_comb_list[i_comb])
            self.modify_condig_dict(param_dict)
            trainer = PtychoNNTrainer(self.config_dict)
            self.run_trainer(trainer)
            self.trainer_list.append(trainer)
            self.update_result_table(i_comb, trainer)
            self.cleanup()

    def run_trainer(self, trainer):
        trainer.verbose = False
        trainer.build()
        trainer.run_training()

    def plot_all_training_history(self):
        for i_comb in range(len(self.param_comb_list)):
            print('Training history for the following config - ')
            print(self.result_table.iloc[i_comb])
            trainer = self.trainer_list[i_comb]
            trainer.plot_training_history()

    def load_model_for_trainer(self, trainer):
        # Reinitialize with a brand new model object
        assert isinstance(trainer.config_dict['model'], (tuple, list)), \
            '`config_dict["model"]` should be a tuple of class handle and kwargs.'
        trainer.build()
        trainer.config_dict['model_path'] = os.path.join(trainer.config_dict['model_save_dir'], 'best_model.pth')
        trainer.load_checkpoint()

    def run_testing_for_all(self, indices, dataset='train'):
        """
        Run test for all trained models with selected samples and plot results.

        :param indices: tuple.
        :param dataset: str. Can be 'train' or 'validation'.
        """
        for i_comb in range(len(self.param_comb_list)):
            print('Testing results for the following config - ')
            print(self.result_table.iloc[i_comb])
            trainer = self.trainer_list[i_comb]
            if not self.keep_models_in_memory:
                # If the models of trainers were not kept in memory, load them back from hard drive.
                self.load_model_for_trainer(trainer)
            trainer.run_testing(indices, dataset=dataset)

    def update_result_table(self, i_comb, trainer):
        self.result_table.at[i_comb, 'min_val_loss_total'] = trainer.metric_dict['best_val_loss']
        self.result_table.at[i_comb, 'min_val_loss_mag'] = trainer.metric_dict['best_val_loss_amp']
        self.result_table.at[i_comb, 'min_val_loss_phase'] = trainer.metric_dict['best_val_loss_ph']
        self.result_table.at[i_comb, 'epoch_min_val_loss'] = trainer.metric_dict['epoch_best_val_loss']

    def cleanup(self):
        if self.verbose:
            get_gpu_memory(show=True)
        if not self.keep_models_in_memory:
            # Destroy model to save memory.
            del self.trainer_list[-1].model
            self.trainer_list[-1].model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
