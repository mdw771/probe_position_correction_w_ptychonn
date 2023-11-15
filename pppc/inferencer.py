import numpy as np
import torch

from pppc.helper import engine_build_from_onnx, mem_allocation, inference
import pppc.configs
from pppc.ptychonn.model import PtychoNNModel


class Inferencer:
    def __init__(self, config_dict: pppc.configs.InferenceConfig):
        """
        Inference engine for PtychoNN.

        :param config_dict: dict. Configuration dictionary.
        """
        self.config_dict = config_dict

    def build(self):
        pass

    def batch_infer(self, x):
        pass
        

class PyTorchInferencer(Inferencer):
    def __init__(self, config_dict: pppc.configs.InferenceConfig):
        super().__init__(config_dict)
        self.model = None

    def build(self):
        self.model = PtychoNNModel()
        self.model.load_state_dict(torch.load(self.config_dict['model_path']))
        self.model.eval()
        self.model = self.model.cuda()

    def preprocess_data(self, x):
        assert(isinstance(x, np.ndarray))
        x = x.astype(np.float32)
        x = x[:, np.newaxis, :, :]
        x = torch.tensor(x, requires_grad=False, device='cuda')
        return x

    def batch_infer(self, x):
        x = self.preprocess_data(x)
        pred_amp, pred_phase = self.model(x)
        return pred_amp.detach().cpu(), pred_phase.detach().cpu()


class ONNXTensorRTInferencer(Inferencer):
    def __init__(self, config_dict: pppc.configs.InferenceConfig):
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


