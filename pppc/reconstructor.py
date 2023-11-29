import os.path
import warnings
from math import *
import glob

import numpy as np
import torch
import tifffile
import tqdm
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt

from pppc.helper import engine_build_from_onnx, mem_allocation, inference, crop_center
import pppc.configs
from pppc.ptychonn.model import PtychoNNModel
from pppc.position_list import ProbePositionList
from pppc.util import class_timeit
from pppc.io import *


class Reconstructor:
    def __init__(self, config_dict: pppc.configs.InferenceConfigDict):
        """
        Inference engine for PtychoNN.

        :param config_dict: dict. Configuration dictionary.
        """
        self.config_dict = config_dict
        self.device = None

    def build(self):
        if self.config_dict['cpu_only'] or (not torch.cuda.is_available()):
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')

    def batch_infer(self, x):
        pass


class VirtualReconstructor(Reconstructor):
    def __init__(self, config_dict: pppc.configs.InferenceConfigDict):
        super().__init__(config_dict)
        self.object_image_array = None

    def set_object_image_array(self, arr):
        self.object_image_array = arr

    def batch_infer(self, x):
        """
        Here x is supposed to be a list of indices for which the object images are to be retrieved.

        :param x: list[int].
        :return: np.ndarray.
        """
        a = np.take(self.object_image_array, indices=x, axis=0)
        return a, a


class PyTorchReconstructor(Reconstructor):
    def __init__(self, config_dict: pppc.configs.InferenceConfigDict):
        super().__init__(config_dict)
        self.model = None

    def build(self):
        super().build()
        self.build_model()

    def build_model(self):
        if self.config_dict['model'] is None:
            self.model = PtychoNNModel()
        else:
            self.model = self.config_dict['model'][0](**self.config_dict['model'][1])
        try:
            self.model.load_state_dict(torch.load(self.config_dict['model_path']))
            self.model.eval()
            if not self.config_dict['cpu_only']:
                self.model = self.model.cuda()
        except:
            warnings.warn('I was unable to locate the model. If this is desired (e.g., you want to override the '
                          'reconstructor object with a virtual reconstructor later for simulation), ignore this '
                          'message. Otherwise, check the path provided. ')

    def preprocess_data(self, x):
        assert isinstance(x, np.ndarray)
        x = x.astype(np.float32)
        x = x[:, np.newaxis, :, :]
        x = torch.tensor(x, requires_grad=False, device=self.device)
        return x

    def batch_infer(self, x):
        """
        Run batch inference.

        :param x: np.ndarray. Input diffraction pattern array of shape [batch_size, height, width].
        :return: (np.ndarray, np.ndarray). Reconstructed amplitude and phase, each of shape [batch_size, height, width].
        """
        x = self.preprocess_data(x)
        pred_amp, pred_phase = self.model(x)
        pred_amp = pred_amp.detach().cpu().numpy()[:, 0, :, :]
        pred_phase = pred_phase.detach().cpu().numpy()[:, 0, :, :]
        return pred_amp, pred_phase


class ONNXTensorRTReconstructor(Reconstructor):
    def __init__(self, config_dict: pppc.configs.InferenceConfigDict):
        """
        Inference engine for PtychoNN.

        :param config_dict: dict. Configuration dictionary.
        """
        super().__init__(config_dict)
        self.onnx_mdl = None

        # Buffers
        self.trt_hin = None
        self.trt_din = None
        self.trt_hout = None
        self.trt_dout = None

        self.trt_engine = None
        self.trt_stream = None
        self.trt_context = None
        self.context = None

    def build(self):
        import pycuda.autoinit
        self.context = pycuda.autoinit.context
        self.onnx_mdl = self.config_dict['onnx_mdl']
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl)
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()

    def batch_infer(self, x):
        """
        Run batch inference.

        :param x:
        :return:
        """
        # in_mb = self.tq_diff.get()
        bsz, ny, nx = x.shape
        # frm_id_list = self.frm_id_q.get()
        np.copyto(self.trt_hin, x.astype(np.float32).ravel())
        pred = np.array(inference(self.trt_context, self.trt_hin, self.trt_hout,
                                  self.trt_din, self.trt_dout, self.trt_stream))

        pred = pred.reshape([bsz, ny, nx])
        return pred


class DatasetInferencer:
    def __init__(self, inference_dict: pppc.configs.InferenceConfigDict):
        self.config_dict = inference_dict
        self.dp_data_file_handle = None
        self.inference_batch_size = self.config_dict['batch_size']
        self.reconstructor = None

    def build(self):
        self.dp_data_file_handle = self.config_dict['dp_data_file_handle']
        self.reconstructor = PyTorchReconstructor(self.config_dict)
        self.reconstructor.build()

    def run(self):
        pbar = tqdm.tqdm(total=self.dp_data_file_handle.num_dps, position=0, leave=True)
        i_start = 0
        i_end = min(i_start + self.inference_batch_size, self.dp_data_file_handle.num_dps)
        while i_start < self.dp_data_file_handle.num_dps:
            # TODO: make this compatible with 4D data array by creating a get indices method in data file handle.
            data_arr = self.dp_data_file_handle.array[i_start:i_end]
            pred_amp, pred_ph = self.reconstructor.batch_infer(data_arr)
            self.write_tiffs(pred_ph, start_index=i_start, name_prefix='pred_phase')
            self.write_tiffs(pred_amp, start_index=i_start, name_prefix='pred_amp')
            i_start = i_end
            i_end = min(i_start + self.inference_batch_size, self.dp_data_file_handle.num_dps)
            pbar.update(i_end - i_start)
        pbar.close()

    def write_tiffs(self, arr, start_index=0, name_prefix='pred_phase'):
        if not os.path.exists(self.config_dict['prediction_output_path']):
            os.makedirs(self.config_dict['prediction_output_path'])
        if arr.ndim == 2:
            tifffile.imwrite(os.path.join(self.config_dict['prediction_output_path'],
                                          '{}_{}.tiff'.format(name_prefix, start_index)),
                             arr)
        else:
            for i in range(arr.shape[0]):
                ind = start_index + i
                tifffile.imwrite(os.path.join(self.config_dict['prediction_output_path'],
                                              '{}_{}.tiff'.format(name_prefix, ind)),
                                 arr[i])


class TileStitcher:

    def __init__(self, config_dict: pppc.configs.InferenceConfigDict):
        self.config_dict = config_dict
        self.images = None
        self.image_stitched = None
        self.position_list = None
        self.flip_lr = False
        self.name_prefix = 'pred_phase_'

    def build(self):
        self.build_position_list()
        self.build_image_array()

    def build_position_list(self):
        if self.config_dict['probe_position_list'] is not None:
            self.position_list = self.config_dict['probe_position_list']
        else:
            self.position_list = ProbePositionList(file_path=self.config_dict['probe_position_data_path'],
                                                   unit=self.config_dict['probe_position_data_unit'],
                                                   psize_nm=self.config_dict['pixel_size_nm'],
                                                   convert_to_pixel=False)

    def build_image_array(self):
        if len(self.config_dict['prediction_output_path']) > 5 and \
                self.config_dict['prediction_output_path'][-5:] == '.tiff':
            self.images = tifffile.imread(self.config_dict['prediction_output_path'])
        else:
            flist = glob.glob(os.path.join(self.config_dict['prediction_output_path'], '{}*'.format(self.name_prefix)))
            n = len(flist)
            self.images = []
            prefix = self.find_true_prefix([os.path.basename(x) for x in flist[:2]])
            for i in range(n):
                img = tifffile.imread(
                    os.path.join(self.config_dict['prediction_output_path'], '{}{}.tiff'.format(prefix, i)))
                self.images.append(img)
            self.images = np.stack(self.images)
        if self.config_dict['central_crop'] is not None:
            self.images = crop_center(self.images, self.config_dict['central_crop'])

    def run(self):
        """
        By Tao Zhou.
        """
        pos = self.position_list.array
        data = self.images
        psize_m = self.position_list.psize_nm * 1e-9
        pos_x = pos[:, 1] * -1
        pos_y = pos[:, 0] * -1
        x = np.arange(pos_x.min() - 0.5e-6, pos_x.max() + 0.5e-6, psize_m)
        y = np.arange(pos_y.min() - 0.5e-6, pos_y.max() + 0.5e-6, psize_m)

        self.image_stitched = np.zeros((y.shape[0], x.shape[0]))
        cnt = np.copy(self.image_stitched)
        cnt1 = cnt + 1

        xx = np.arange(self.images.shape[2]) * psize_m
        xx -= xx.mean()
        yy = np.arange(self.images.shape[1]) * psize_m
        yy -= yy.mean()

        for i in tqdm.trange(data.shape[0], position=0, leave=True):
            xxx = xx - pos_x[i]
            yyy = yy - pos_y[i]
            img = np.fliplr(data[i, :, :]) if self.flip_lr else data[i, :, :]
            find_pha = interpolate.interp2d(xxx[:], yyy[:], img, kind='linear', fill_value=0)
            # find_pha = interpolate.RegularGridInterpolator((yyy[:],xxx[:]),data[i,:,:],\
            #                                               method='linear', fill_value=0, bounds_error=False)
            # crop out 40 pixels on each end, you can try 32, 16...
            tmp = find_pha(x, y)
            cnt += tmp != 0
            self.image_stitched += tmp
        self.image_stitched = (self.image_stitched / np.maximum(cnt, cnt1))[:, ::-1]

    def find_true_prefix(self, name_list):
        s = ''
        for i in range(len(name_list[0])):
            flag = True
            for fname in name_list:
                if s + name_list[0][i] not in fname:
                    flag = False
                    break
            if flag:
                s = s + name_list[0][i]
            else:
                break
        return s
