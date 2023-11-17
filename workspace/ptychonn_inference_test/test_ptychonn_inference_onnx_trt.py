import sys
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn/pppc')

import numpy as np
import matplotlib.pyplot as plt

import pppc
from pppc.message_logger import logger


if __name__ == '__main__':
    config = pppc.configs.InferenceConfig()
    config['onnx_mdl'] = '../../trained_models/ptychoNN_8.onnx'

    data = np.load('data/20191008_39_diff_sample.npy')
    data = data.reshape([data.shape[0] * data.shape[1], *data.shape[2:]])[:, 64:192, 64:192]
    data = np.concatenate([data, data], axis=0)
    logger.info(data.shape)

    inferencer = pppc.reconstructor.ONNXTensorRTReconstructor(config)
    inferencer.build()
    pred = inferencer.batch_infer(data)

    fig, axes = plt.subplots(2, 4)
    for i in range(pred.shape[0]):
        axes[i // 4][i % 4].imshow(pred[i])
    plt.show()
