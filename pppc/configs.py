import collections
import json
import os

import torch


class ConfigDict(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda: None)

    def __str__(self):
        s = ''
        for key in self.keys():
            s += '{}: {}\n'.format(key, self[key])
        return s

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    def get_serializable_dict(self):
        d = {}
        for key in self.keys():
            v = self[key]
            if not self.__class__.is_jsonable(v):
                if isinstance(v, (tuple, list)):
                    v = '_'.join([str(x) for x in v])
                else:
                    v = str(v)
            d[key] = v
        return d

    def dump_to_json(self, filename):
        try:
            f = open(filename, 'w')
            d = self.get_serializable_dict()
            json.dump(d, f)
            f.close()
        except:
            print('Failed to dump json.')

    def load_from_json(self, filename):
        """
        This function only overwrites entries contained in the JSON file. Unspecified entries are unaffected.
        """
        f = open(filename, 'r')
        d = json.load(f)
        for key in d.keys():
            self[key] = d[key]
        f.close()

class InferenceConfigDict(ConfigDict):

    def __init__(self, *args, **kwargs):
        super().__init__()
        # ===== PtychoNN configs =====
        self['batch_size'] = 1
        # Path to a trained PtychoNN model.
        self['model_path'] = None
        # The model. Should be a tuple(nn.Module, kwargs): the first element of the tuple is the class handle of a
        # model class, and the second is a dictionary of keyword arguments. The model will be instantiated using these.
        # This value is used to instantiate a model object, whose weights are overwritten with those read from
        # `model_path`. The provided model class and arguments must match the model being loaded.
        self['model'] = None
        # Path to save PtychoNN prediction results.
        self['prediction_output_path'] = None
        self['cpu_only'] = False

        # ===== Image registration configs =====
        self['registration_method'] = 'error_map'
        # Method for detecting outlier matches for SIFT. Can be "trial_error", "kmeans", "isoforest", "ransac".
        self['sift_outlier_removal_method'] = 'kmeans'
        # The length of the near-boundary region of the image. When doing SIFT registration, if a matching pair of
        # keypoints involve points in this region, it will be discarded. However, if all matches (after outlier removal)
        # are near-boundary, they are used as they are. This operation is less aggressive than `central_crop`.
        self['sift_border_exclusion_length'] = 16
        # Image downsampling before registration.
        self['registration_downsample'] = 1
        # Hybrid registration algorithms
        self['hybrid_registration_algs'] = ['error_map_multilevel', 'error_map_expandable', 'sift']
        # Hybrid registration tolerances
        self['hybrid_registration_tols'] = [0.15, 0.3, 0.3]

        # ===== General configs =====
        self['dp_data_file_path'] = None
        # Used as an alternative to `dp_data_file_path`. Should be a `DataFileHandle` object.
        self['dp_data_file_handle'] = None
        # A ProbePositionList object used for finding nearest neighbors in collective mode.
        # If None, `probe_position_data_path` must be provided.
        self['probe_position_list'] = None
        self['probe_position_data_path'] = None
        self['probe_position_data_unit'] = None
        self['pixel_size_nm'] = None
        # Baseline positions. Used by ProbePositionCorrectorChain when the serial mode result is bad.
        self['baseline_position_list'] = None
        # Patch size used for image registration. If smaller than the reconstructed object size, a patch will
        # be cropped from the center.
        self['central_crop'] = None
        # Method for correction. Can be 'serial' or 'collective'
        self['method'] = 'collective'
        self['max_shift'] = 7
        # Number of neighbors in collective registration
        self['num_neighbors_collective'] = 3
        self['offset_estimator_order'] = 1
        self['offset_estimator_beta'] = 0.5
        self['smooth_constraint_weight'] = 1e-2
        self['random_seed'] = 123
        self['debug'] = None


class TrainingConfigDict(ConfigDict):
    def __init__(self, *args, **kwargs):
        super().__init__()
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
        self['loss_function'] = None  # Can be None (default to L1Loss) or a Callable.
