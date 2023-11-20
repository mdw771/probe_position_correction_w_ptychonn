import collections


class InferenceConfig(collections.defaultdict):

    def __init__(self):
        super().__init__(lambda: None)
        self['batch_size'] = 1
        # Path to a trained PtychoNN model.
        self['model_path'] = None
        self['dp_data_file_path'] = None
        # Used as an alternative to `dp_data_file_path`. Should be a `DataFileHandle` object.
        self['dp_data_file_handle'] = None
        self['probe_position_data_path'] = None
        self['probe_position_data_unit'] = None
        self['pixel_size_nm'] = None
        # Patch size used for image registration. If smaller than the reconstructed object size, a patch will
        # be cropped from the center.
        self['central_crop'] = None
        # Method for correction. Can be 'serial' or 'collective'
        self['method'] = 'collective'
        self['debug'] = None
