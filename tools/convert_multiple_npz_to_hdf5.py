import argparse

import numpy as np
import tqdm
import glob
import os

import h5py
import numpy
import tqdm


class NPZ2HDF5Converter:

    def __init__(self, file_list, output_fname='data.h5', total_shape_dict=None):
        self.file_list = file_list
        self.output_fname = output_fname
        self.f_h5 = None
        self.total_shape_dict = {}
        self.dtype_dict = {}
        self.all_keys = []
        self.data_keys = ['real', 'reciprocal']
        self.total_shape_dict = total_shape_dict

    def get_total_shape_and_dtype(self):
        print('Checking data...')
        f = np.load(self.file_list[0], 'r')
        self.all_keys = list(f.keys())
        for key in f.keys():
            self.dtype_dict[key] = f[key].dtype
        if self.total_shape_dict is None:
            self.total_shape_dict = {}
            for key in self.data_keys:
                self.total_shape_dict[key] = [0] * len(f[key].shape)
            for fname in tqdm.tqdm(self.file_list):
                f = np.load(fname, 'r')
                for key in self.data_keys:
                    self.total_shape_dict[key][0] += f[key].shape[0]
                    self.total_shape_dict[key][1:] = f[key].shape[1:]
        for key in self.data_keys:
            print('Total shape for {}: {}'.format(key, self.total_shape_dict[key]))
        for key in self.all_keys:
            print('Dtype for {}: {}'.format(key, self.dtype_dict[key]))

    def initialize_hdf5(self):
        self.get_total_shape_and_dtype()
        self.f_h5 = h5py.File(self.output_fname, 'w')
        grp_meta = self.f_h5.create_group('metadata')
        grp_meta.create_dataset('source_files', shape=(len(self.file_list, )), dtype='S10')
        grp_data = self.f_h5.create_group('data')
        for key in self.data_keys:
            grp_data.create_dataset(key, shape=self.total_shape_dict[key], dtype=self.dtype_dict[key])
        grp_data.create_group('positions')
        grp_data.create_group('probes')

    def run(self):
        self.initialize_hdf5()
        offset_dict = {}
        for key in self.data_keys:
            offset_dict[key] = 0
        for i, fname in enumerate(tqdm.tqdm(self.file_list)):
            f = np.load(fname, 'r')
            self.f_h5['metadata/source_files'][i] = os.path.splitext(os.path.basename(fname))[0]
            for key in self.data_keys:
                arr = f[key]
                offset = offset_dict[key]
                self.f_h5['data/{}'.format(key)][offset:offset + arr.shape[0]] = arr
                offset_dict[key] += arr.shape[0]
            self.f_h5['data/positions'].create_dataset(os.path.splitext(os.path.basename(fname))[0], data=f['position'])
            self.f_h5['data/probes'].create_dataset(os.path.splitext(os.path.basename(fname))[0], data=f['probe'])
        self.f_h5.close()

    @staticmethod
    def generate_file_list_from_dir(source_dir, exclude_list=()):
        flist = glob.glob(os.path.join(source_dir, '*.npz'))
        final_flist = []
        for f in flist:
            if os.path.basename(f) not in exclude_list:
                final_flist.append(f)
        return final_flist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default=None)
    parser.add_argument('--source_dir', default=None)
    parser.add_argument('--exclude_list', default=None)
    parser.add_argument('--output_filename', default='data.h5')
    args = parser.parse_args()

    exclude_list = args.exclude_list
    if exclude_list is not None:
        exclude_list = exclude_list.split(',')
    else:
        exclude_list = ()
    if args.filename is not None:
        file_list = [args.filename]
    else:
        file_list = NPZ2HDF5Converter.generate_file_list_from_dir(args.source_dir, exclude_list)
    converter = NPZ2HDF5Converter(file_list, args.output_filename,
                                  total_shape_dict={'real': [34596, 128, 128], 'reciprocal': [34596, 512, 512]})
    converter.run()
