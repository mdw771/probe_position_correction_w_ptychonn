import collections


class InferenceConfig(collections.defaultdict):

    def __init__(self):
        super().__init__(lambda: None)
        self['batch_size'] = 1

