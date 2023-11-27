import collections

import torch

class InferenceConfigDict(collections.defaultdict):

    def __init__(self, *args, **kwargs):
        super().__init__(lambda: None)
        self['batch_size'] = 1
        # Path to a trained PtychoNN model.
        self['model_path'] = None
        # The model. Should be a tuple(nn.Module, kwargs): the first element of the tuple is the class handle of a
        # model class, and the second is a dictionary of keyword arguments. The model will be instantiated using these.
        # This value is used to instantiate a model object, whose weights are overwritten with those read from
        # `model_path`. The provided model class and arguments must match the model being loaded.
        self['model'] = None
        self['dp_data_file_path'] = None
        # Used as an alternative to `dp_data_file_path`. Should be a `DataFileHandle` object.
        self['dp_data_file_handle'] = None
        # A ProbePositionList object. If None, `probe_position_data_path` must be provided.
        self['probe_position_list'] = None
        self['probe_position_data_path'] = None
        self['probe_position_data_unit'] = None
        self['pixel_size_nm'] = None
        # Patch size used for image registration. If smaller than the reconstructed object size, a patch will
        # be cropped from the center.
        self['central_crop'] = None
        # Path to save PtychoNN prediction results.
        self['prediction_output_path'] = None
        # Method for correction. Can be 'serial' or 'collective'
        self['method'] = 'collective'
        self['max_shift'] = 7
        self['cpu_only'] = False
        self['debug'] = None


class TrainingConfigDict(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda: None)
        self['batch_size_per_process'] = 64
        self['num_epochs'] = 60
        self['learning_rate_per_process'] = 1e-3
        self['optimizer'] = 'adam'  # String of optimizer name or the handle of a subclass of torch.optim.Optimizer
        self['model_save_dir'] = '.'  # Directory to save trained models
        # The model. The three options are:
        # (1) None: the model will be instantiated with the default model class.
        # (2) A object of nn.Module: the model object will be used as provided.
        # (3) tuple(nn.Module, kwargs): the first element of the tuple is the class handle of a model class, and the
        #     second is a dictionary of keyword arguments. The model will be instantiated using these.
        self['model'] = None


class PtychoNNTrainingConfigDict(TrainingConfigDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['height'] = 256
        self['width'] = 256
        self['num_lines_for_training'] = 100  # Number of lines used for training
        self['num_lines_for_testing'] = 60  # Number of lines used for testing
        self['num_lines_for_validation'] = 805  # Number of lines used for testing
        self['dataset'] = None  # A torch.Dataset object
        self['validation_ratio'] = 0.003  # Ratio of validation set out of the entire dataset
