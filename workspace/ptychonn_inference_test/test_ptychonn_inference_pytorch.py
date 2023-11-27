import sys
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn/pppc')

import numpy as np
import matplotlib.pyplot as plt

import pppc
from pppc.message_logger import logger


if __name__ == '__main__':
    config = pppc.configs.InferenceConfigDict()
    config['model_path'] = '../../trained_models/ptychoNN_CNN_encoder_decoder_pytorch_statedict.pth'
    data = np.load('data/20191008_39_diff_sample.npy')
    data = data.reshape([data.shape[0] * data.shape[1], *data.shape[2:]])
    logger.info(data.shape)

    inferencer = pppc.reconstructor.PyTorchReconstructor(config)
    inferencer.build()
    pred_amp, pred_phase = inferencer.batch_infer(data)

    fig, axes = plt.subplots(2, 2)
    for i in range(pred_phase.shape[0]):
        axes[i // 2][i % 2].imshow(pred_phase[i])
    fig.suptitle('Phase')
    fig, axes = plt.subplots(2, 2)
    for i in range(pred_amp.shape[0]):
        axes[i // 2][i % 2].imshow(pred_amp[i])
    fig.suptitle('Amplitude')
    plt.show()
