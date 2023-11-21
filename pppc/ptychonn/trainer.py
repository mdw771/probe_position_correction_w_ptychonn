import os

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

from .model import PtychoNNModel
from pppc.configs import TrainingConfigDict
from pppc.message_logger import logger


class PtychoNNTrainer:

    def __init__(self, config_dict: TrainingConfigDict):
        self.config_dict = config_dict
        self.dataset = self.config_dict['dataset']
        assert isinstance(self.dataset, Dataset)
        self.training_dataset = None
        self.validation_dataset = None
        self.validation_ratio = self.config_dict['validation_ratio']
        self.model = None
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

        self.debug = self.config_dict['debug']

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
        self.build_optimizer()
        self.build_scheduler()

        self.loss_criterion = nn.L1Loss()
        self.initialize_metric_dict()

    def run_training(self):
        for epoch in range(self.num_epochs):
            # Set model to train mode and run training
            self.model.train()
            self.run_trainig_epoch()

            # Switch model to eval mode and run validation
            self.model.eval()
            self.run_validation()

            logger.info('Epoch: %d | FT  | Train Loss: %.5f | Val Loss: %.5f' % (
                epoch, self.metric_dict['losses'][-1][0], self.metric_dict['val_losses'][-1][0]))
            logger.info('Epoch: %d | Amp | Train Loss: %.4f | Val Loss: %.4f' % (
                epoch, self.metric_dict['losses'][-1][1], self.metric_dict['val_losses'][-1][1]))
            logger.info('Epoch: %d | Ph  | Train Loss: %.3f | Val Loss: %.3f' % (
                epoch, self.metric_dict['losses'][-1][2], self.metric_dict['val_losses'][-1][2]))
            logger.info('Epoch: %d | Ending LR: %.6f ' % (epoch, self.metric_dict['lrs'][-1][0]))

    def run_trainig_epoch(self):
        tot_loss = 0.0
        loss_amp = 0.0
        loss_ph = 0.0
        n_batches = 0
        for i, (ft_images, amps, phs) in enumerate(tqdm(self.training_dataloader)):
            ft_images = ft_images.to(self.device)  # Move everything to device
            amps = amps.to(self.device)
            phs = phs.to(self.device)

            pred_amps, pred_phs = self.model(ft_images)  # Forward pass

            if self.debug:
                self.plot_images([ft_images[0, 0], amps[0, 0], phs[0, 0], pred_amps[0, 0], pred_phs[0, 0]])

            # Compute losses
            loss_a = self.loss_criterion(pred_amps, amps)  # Monitor amplitude loss
            # Monitor phase loss but only within support (which may not be same as true amp)
            loss_p = self.loss_criterion(pred_phs, phs)
            loss = loss_a + loss_p  # Use equiweighted amps and phase

            # Zero current grads and do backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tot_loss += loss.detach().item()
            loss_amp += loss_a.detach().item()
            loss_ph += loss_p.detach().item()

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

            val_loss_a = self.loss_criterion(pred_amps, amps)
            val_loss_p = self.loss_criterion(pred_phs, phs)
            val_loss = val_loss_a + val_loss_p

            tot_val_loss += val_loss.detach().item()
            val_loss_amp += val_loss_a.detach().item()
            val_loss_ph += val_loss_p.detach().item()

            n_batches += 1
        self.metric_dict['val_losses'].append([tot_val_loss / n_batches,
                                               val_loss_amp / n_batches,
                                               val_loss_ph / n_batches])

        # Update saved model if val loss is lower
        if tot_val_loss / n_batches < self.metric_dict['best_val_loss']:
            logger.info("Saving improved model after Val Loss improved from %.5f to %.5f" % (
                self.metric_dict['best_val_loss'], tot_val_loss / n_batches))
            self.metric_dict['best_val_loss'] = tot_val_loss / n_batches
            self.update_saved_model()

    def initialize_metric_dict(self):
        self.metric_dict['losses'] = []
        self.metric_dict['val_losses'] = []
        self.metric_dict['lrs'] = []
        self.metric_dict['best_val_loss'] = np.inf

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
        self.model = PtychoNNModel()
        if self.device.type == 'cuda' and self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            logger.info('Using DataParallel with {} devices.'.format(self.num_gpus))
        self.model.to(self.device)

    def update_saved_model(self):
        """
        Updates saved model if validation loss is minimum.
        """
        path = self.config_dict['model_save_dir']
        if not os.path.isdir(path):
            os.mkdir(path)
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
        torch.save(self.model.module.state_dict(), os.path.join(path, 'best_model.pth'))

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
