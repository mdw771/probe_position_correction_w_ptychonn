"""
Generate HDF5-formatted datasets for training or testing.
"""
import glob
import re
import os
import shutil
import logging

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.interpolate
import tifffile

logger = logging
logger.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)


class DatasetGenerator:

    def __init__(self, data_path, label_path, output_path='./data_train.h5', target_shape=(128, 128),
                 target_psize_nm=20, standardize_data=True,
                 data_dtype='float32', label_dtype='complex64', transform_func=None, transform_func_kwargs=None,
                 *args, **kwargs):
        self.data_path = data_path
        self.label_path = label_path
        self.output_path = output_path
        self.target_shape = target_shape
        self.target_psize_nm = target_psize_nm
        self.data_shape = [0, *self.target_shape]
        self.data_dtype = data_dtype
        self.label_dtype = label_dtype
        self.standardize_data = standardize_data
        self.f_out = None
        self.transform_func = transform_func
        self.transform_func_kwargs = {} if transform_func_kwargs is None else transform_func_kwargs
        self.z_data = 0
        self.z_label = 0
        self.debug_dir = os.path.join(os.path.dirname(self.output_path), 'dataset_generator_debug')
        self.debug = False

    def initialize_output_file(self):
        self.f_out = h5py.File(self.output_path, 'w')
        grp = self.f_out.create_group('data')
        dset = grp.create_dataset('real', shape=tuple(self.data_shape), maxshape=tuple(self.data_shape),
                                  dtype=self.label_dtype)
        dset = grp.create_dataset('reciprocal', shape=tuple(self.data_shape), maxshape=tuple(self.data_shape),
                                  dtype=self.data_dtype)
        grp = self.f_out.create_group('metadata')
        dset = grp.create_dataset('psize_nm', data=np.array([self.target_psize_nm]))

    def get_data_storage_space(self):
        """
        Estimate the storage space consumed by diffraction data, returned in GB.
        :return:
        """
        return self.calculate_storage_space(self.data_shape, self.data_dtype)

    def calculate_storage_space(self, shape, dtype):
        return np.prod(shape) * np.dtype(dtype).itemsize / 1e9

    def resize_3d_arr(self, arr, scaling_factor):
        """
        Resize a 3D array in the last 2 dimensions, keeping array size.
        """
        z = np.arange(arr.shape[0])
        y = np.arange(arr.shape[1])
        x = np.arange(arr.shape[2])
        offset_y = (y[0] + y[-1]) / 2
        offset_x = (x[0] + x[-1]) / 2
        y_new = (y - offset_y) / scaling_factor[0] + offset_y
        x_new = (x - offset_x) / scaling_factor[1] + offset_x
        interp = scipy.interpolate.RegularGridInterpolator((z, y, x), arr, bounds_error=False, fill_value=0)
        zz, yy, xx = np.meshgrid(z, y_new, x_new, indexing='ij')
        pts = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
        vals = interp(pts)
        vals = vals.reshape(arr.shape)
        return vals

    def rescale_diffraction_data_to_match_pixel_size(self, data, original_psize, new_psize):
        return self.resize_3d_arr(data, [new_psize / original_psize] * 2)

    def write_metadata(self):
        self.f_out['metadata/psize_nm'][0] = self.target_psize_nm

    def write_diffraction_data(self, data):
        self.f_out['data/reciprocal'][self.z_data:self.z_data + data.shape[0]] = data
        self.z_data += data.shape[0]

    def write_label(self, label):
        self.f_out['data/real'][self.z_label:self.z_label + label.shape[0]] = label
        self.z_label += label.shape[0]

    def finish_output_file(self):
        self.f_out['data/reciprocal'].resize([self.z_data, *self.target_shape])
        self.f_out['data/real'].resize([self.z_label, *self.target_shape])
        self.f_out.close()
        print('Actual storage space for diffraction data: {} GB.'.format(
            self.calculate_storage_space([self.z_data, *self.target_shape], self.data_dtype)))
        print('Actual storage space for labels: {} GB.'.format(
            self.calculate_storage_space([self.z_label, *self.target_shape], self.label_dtype)))
        print('Final number of diffraction patterns: {}.'.format(self.z_data))


class BNPDatasetGenerator(DatasetGenerator):

    def __init__(self, data_path, label_path, mda_path, output_path='./data_train.h5',
                 target_shape=(256, 256), target_psize_nm=20, standardize_data=True,
                 transform_func=None, transform_func_kwargs=None,
                 *args, **kwargs):
        """
        For generating datasets from Bionanoprobe's ptychography data.

        :param data_path: str. Path to the folder that contains diffraction data.
        :param label_path: str. Path to the folder that contains reconstructed object functions (phases).
        :param transform_func: Callable. A function that transform a 3D array into the desired shape.
        """
        super().__init__(data_path, label_path, output_path, target_shape, target_psize_nm,
                         standardize_data=standardize_data,
                         transform_func=transform_func,
                         transform_func_kwargs=transform_func_kwargs,
                         *args, **kwargs)
        self.mda_path = mda_path
        self.scan_indices = []

    def build_scan_indices(self):
        data_indices = []
        flist = glob.glob(os.path.join(self.data_path, '*.h5'))
        for f in flist:
            res = re.search(r'fly(\d+)_(\d+)', f)
            if res is not None:
                scan_idx = res.groups()[0]
                data_indices.append(int(scan_idx))
        data_indices = np.unique(data_indices)
        # Indices should also have labels.
        filtered_indices = []
        for ind in data_indices:
            if len(glob.glob(os.path.join(self.label_path, 'fly{:03}'.format(ind), 'roi0_Ndp*'))) > 0:
                filtered_indices.append(ind)
        self.scan_indices = filtered_indices

    def build_data_shape(self):
        self.data_shape = np.array([0, *self.target_shape]).astype(int)
        for ind in self.scan_indices:
            scan_files = glob.glob(os.path.join(self.data_path, '*fly{:04}*.h5'.format(ind)))
            for fname in scan_files:
                f = h5py.File(fname, 'r')
                self.data_shape[0] += f['entry/data/data'].shape[0]
        print('Total data shape = {}'.format(self.data_shape))
        print('Estimated storage space for diffraction data: {} GB.'.format(self.get_data_storage_space()))

    def build(self):
        self.build_scan_indices()
        self.build_data_shape()
        self.initialize_output_file()

    def get_probe_positions_from_mda(self, fname, offset=(0, 0)):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
        import readMDA
        mda_data = readMDA.readMDA(fname)
        y_pos = np.array(mda_data[1].p[0].data)
        x_pos = np.array(mda_data[2].p[0].data)[0]
        xx, yy = np.meshgrid(y_pos, x_pos)
        # Reverse x on even rows to form snake pattern
        for i in range(1, xx.shape[0], 2):
            xx[i, :] = xx[i, ::-1]
        yy = yy - yy[0, 0] + offset[0]
        xx = xx - xx[0, 0] + offset[1]
        y = yy.reshape(-1)
        x = xx.reshape(-1)
        return y, x

    def run(self):
        z = 0
        for scan_ind in self.scan_indices:
            scan_files = glob.glob(os.path.join(self.data_path, '*fly{:04}*.h5'.format(scan_ind)))
            for fname in scan_files:
                # Write data
                f_data = h5py.File(fname, 'r')
                dat = self.transform_func(f_data['entry/data/data'][...], **self.transform_func_kwargs)
                self.f_out['data/reciprocal'][z:z + dat.shape[0]] = dat
                # Write label
                y_pos, x_pos = self.get_probe_positions_from_mda(
                    os.path.join(self.mda_path, 'bnp_fly{:04}.mda'.format(scan_ind))
                )


class FromPtychoShelvesDatasetGenerator(DatasetGenerator):

    def __init__(self, data_path, label_path, output_path='./data_train.h5',
                 target_shape=(256, 256), target_psize_nm=20, standardize_data=True,
                 standardize_data_across_samples=False,
                 transform_func=None, transform_func_kwargs=None,
                 *args, **kwargs):
        """
        Generate data from the HDF5 files used by PtychoShelves.

        :param data_path: str.
        :param label_path: str.
        :param output_path: str.
        :param target_shape: str.
        :param transform_func: Callable.
        :param transform_func_kwargs: dict.
        """
        super().__init__(data_path, label_path, output_path, target_shape, target_psize_nm,
                         standardize_data=standardize_data,
                         transform_func=transform_func,
                         transform_func_kwargs=transform_func_kwargs,
                         *args, **kwargs)
        self.scan_indices = []
        self.standardize_data_across_samples = standardize_data_across_samples
        assert self.data_path == self.label_path, 'data_path and label_path for PtychoShelves data should be the same.'

    def build_scan_indices(self):
        data_indices = []
        flist = glob.glob(os.path.join(self.data_path, 'fly*'))
        for f in flist:
            res = re.search(r'fly(\d+)', f)
            if res is not None:
                scan_idx = res.groups()[0]
                data_indices.append(int(scan_idx))
        data_indices = np.unique(data_indices)
        # Indices should also have labels.
        filtered_indices = []
        for ind in data_indices:
            if len(glob.glob(os.path.join(self.label_path, 'fly{:03}'.format(ind), 'roi0_Ndp*'))) > 0:
                filtered_indices.append(ind)
        self.scan_indices = filtered_indices

    def build_data_shape(self):
        self.data_shape = np.array([0, *self.target_shape]).astype(int)
        for ind in self.scan_indices:
            scan_files = glob.glob(os.path.join(self.data_path, 'fly{:03}'.format(ind), 'data_roi0_Ndp*_dp.hdf5'))
            fname = scan_files[0]
            f = h5py.File(fname, 'r')
            self.data_shape[0] += f['dp'].shape[0]
        print('Total data shape = {}'.format(self.data_shape))
        print('Estimated storage space for diffraction data: {} GB.'.format(self.get_data_storage_space()))

    def build(self):
        self.z_data = 0
        self.z_label = 0
        self.build_scan_indices()
        self.build_data_shape()
        self.initialize_output_file()
        if self.debug and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def choose_h5_file(self, scan_folder, type='dp', return_size=False):
        """
        If there are multiple HDF5 files in the scan folder, choose the one associated with reconstructions.
        :return: str.
        """
        recon_folder = glob.glob(os.path.join(scan_folder, 'roi0_*'))[0]
        s = int(re.findall(r'\d+', recon_folder)[-1])
        h5s = glob.glob(os.path.join(scan_folder, 'data_roi0_Ndp{}_{}.hdf5'.format(s, type)))[0]
        if return_size:
            return h5s, s
        return h5s

    def crop_data_to_target_shape(self, arr):
        offset = (np.array(arr.shape[1:]) - self.target_shape) // 2
        arr = arr[:, offset[0]:offset[0] + self.target_shape[0], offset[1]:offset[1] + self.target_shape[1]]
        return arr

    def convert_x_pos_to_snake_pattern(self, pos_x, nx):
        i = 0
        ir = 0
        while i < len(pos_x):
            if ir % 2 == 1:
                pos_x[i:i + nx] = pos_x[i:i + nx][::-1]
            i += nx
            ir += 1
        return pos_x

    def convert_diffraction_data_to_snake_pattern(self, data, nx):
        i = 0
        ir = 0
        while i < len(data):
            if ir % 2 == 1:
                data[i:i + nx] = data[i:i + nx][::-1, :, :]
            i += nx
            ir += 1
        return data

    def get_parameters(self, scan_idx):
        scan_folder = os.path.join(self.data_path, 'fly{:03}'.format(scan_idx))
        f_param, s = self.choose_h5_file(scan_folder, type='para', return_size=True)
        f_param = h5py.File(f_param, 'r')
        psize_m = f_param['dx'][0]
        pos_x_m = f_param['ppX'][...]
        pos_y_m = f_param['ppY'][...]
        nx = f_param['N_scan_x'][0]
        ny = f_param['N_scan_y'][0]
        pos_x_m = self.convert_x_pos_to_snake_pattern(pos_x_m, nx)
        params = {}
        params['psize_nm'] = psize_m * 1e9
        params['pos_px'] = np.stack([pos_y_m, pos_x_m], axis=-1) / psize_m
        params['probe_size'] = s
        params['nx'] = nx
        params['indices_to_keep'] = list(range(len(pos_y_m)))
        return params

    def clean_data(self, data):
        data_mean = np.mean(data, axis=0)
        # data_std = np.std(data, axis=0)
        data[data > data_mean * 80] = 0
        return data

    def get_diffraction_data(self, scan_idx, params):
        scan_folder = os.path.join(self.data_path, 'fly{:03}'.format(scan_idx))
        f_dp = self.choose_h5_file(scan_folder, type='dp')
        f_dp = h5py.File(f_dp, 'r')
        logger.info('Loading diffraction data...')
        dat = f_dp['dp'][...]
        psize_nm = params['psize_nm']
        logger.info('Transforming diffraction data...')
        if not np.allclose(dat.shape[1:], self.target_shape):
            orig_shape = dat.shape[1:]
            dat = self.crop_data_to_target_shape(dat)
            psize_nm = psize_nm / np.mean(self.target_shape / np.array(orig_shape))
        dat = self.rescale_diffraction_data_to_match_pixel_size(dat, psize_nm, self.target_psize_nm)
        dat = self.convert_diffraction_data_to_snake_pattern(dat, params['nx'])
        dat = dat[params['indices_to_keep']]
        if self.debug:
            tifffile.imwrite(os.path.join(self.debug_dir, 'data_raw_{:03}.tiff'.format(scan_idx)), dat)
        dat = self.clean_data(dat)
        if self.standardize_data:
            dat = (dat - np.mean(dat, axis=(1, 2), keepdims=True)) / (np.std(dat, axis=(1, 2), keepdims=True) + 1e-3)
        if self.standardize_data_across_samples:
            dat = dat - np.mean(dat, axis=0)
        if self.debug:
            tifffile.imwrite(os.path.join(self.debug_dir, 'data_{:03}.tiff'.format(scan_idx)), dat)
        return dat

    def get_reconstruction_filename(self, scan_idx):
        """
        Find the path to the reconstruction *.mat file with the largest number of iterations.
        """
        scan_folder = os.path.join(self.data_path, 'fly{:03}'.format(scan_idx))
        _, s = self.choose_h5_file(scan_folder, type='dp', return_size=True)
        roi_folder = os.path.join(scan_folder, 'roi0_Ndp{}'.format(s))
        try:
            recon_folder = glob.glob(os.path.join(roi_folder, 'ML*Ndp{}*'.format(s)))[0]
        except:
            print('Was looking for ML*Ndp{}*, but only the following are available:'.format(s))
            print(glob.glob(os.path.join(roi_folder, 'ML*Ndp{}*'.format(s))))
            raise ValueError
        mat_files = glob.glob(os.path.join(recon_folder, 'Niter*.mat'))
        ns = [int(re.findall(r'\d+', x)[-1]) for x in mat_files]
        ns.sort()
        n = ns[-1]
        recon_file = os.path.join(recon_folder, 'Niter{}.mat'.format(n))
        return recon_file

    def get_object_function_from_mat_file(self, filename):
        try:
            f = h5py.File(filename, 'r')
            dset = f['object']
            obj = dset['real'][...] + 1j * dset['imag'][...]
        except:
            obj = scipy.io.loadmat(filename)['object']
        # obj = np.transpose(obj)
        return obj

    def process_positions(self, scan_idx, params):
        recon_file = self.get_reconstruction_filename(scan_idx)
        obj = np.angle(self.get_object_function_from_mat_file(recon_file))
        pos = params['pos_px']
        offset_pos = self.offset_positions(pos, obj.shape)
        # Only keep positions that have enough overlap.
        filtered_pos, pos_indices_to_keep = self.filter_positions(offset_pos, params, obj.shape)
        params['pos_px'] = filtered_pos
        params['indices_to_keep'] = pos_indices_to_keep
        return params

    def get_labels(self, scan_idx, params):
        scan_folder = os.path.join(self.data_path, 'fly{:03}'.format(scan_idx))
        recon_file = self.get_reconstruction_filename(scan_idx)
        obj = self.get_object_function_from_mat_file(recon_file)
        pos = params['pos_px']
        logger.info('Extracting object tiles...')
        labels_ph = self.extract_object_tiles(np.angle(obj), pos, params)
        labels_mag = self.extract_object_tiles(np.abs(obj), pos, params)
        labels = labels_mag * np.exp(1j * labels_ph)
        if self.debug:
            tifffile.imwrite(os.path.join(self.debug_dir, 'labels_ph_{:03}.tiff'.format(scan_idx)), np.angle(labels))
            tifffile.imwrite(os.path.join(self.debug_dir, 'labels_mag_{:03}.tiff'.format(scan_idx)), np.abs(labels))
        return labels

    def extract_object_tiles(self, obj, pos, params):
        orig_psize_nm = params['psize_nm']
        new_psize_nm = self.target_psize_nm
        radius = np.array(self.target_shape) / 2
        radius = radius * new_psize_nm / orig_psize_nm
        query_pts = []
        for this_pos in pos:
            y = np.linspace(this_pos[0] - radius[0], this_pos[0] + radius[0], self.target_shape[0])
            x = np.linspace(this_pos[1] - radius[1], this_pos[1] + radius[1], self.target_shape[1])
            xx, yy = np.meshgrid(x, y)
            pts = np.stack([yy, xx], axis=-1).reshape([-1, 2])
            query_pts.append(pts)
        query_pts = np.concatenate(query_pts, axis=0)
        grid_y = np.arange(obj.shape[0])
        grid_x = np.arange(obj.shape[1])
        interp = scipy.interpolate.RegularGridInterpolator((grid_y, grid_x), obj, bounds_error=False, fill_value=0)
        vals = interp(query_pts)
        vals = vals.reshape([len(pos), *self.target_shape])
        return vals

    def filter_positions(self, pos, params, obj_shape, extra_factor=320):
        """
        Only keep positions that have full overlap.
        """
        e = int(extra_factor / params['psize_nm'])
        s = params['probe_size']
        sy = s + e
        ey = obj_shape[0] - s - e
        sx = s + e
        ex = obj_shape[1] - s - e
        mask = (pos[:, 0] > sy) * (pos[:, 0] < ey) * (pos[:, 1] > sx) * (pos[:, 1] < ex)
        indices_to_keep = np.where(mask)[0]
        pos = pos[indices_to_keep]
        return pos, indices_to_keep

    def offset_positions(self, pos, obj_shape):
        """
        Offset probe positions such that they directly correspond to pixel coordinates in the image.
        """
        # span_y = pos[:, 0].max() - pos[:, 0].min()
        # span_x = pos[:, 1].max() - pos[:, 1].min()
        # offset = (np.array(obj_shape) - np.array([span_y, span_x])) / 2
        offset = np.array([obj_shape[0] / 2 + pos[:, 0].min(), obj_shape[1] / 2 + pos[:, 1].min()])
        pos = pos - pos[0] + offset
        return pos

    def run(self):
        self.write_metadata()

        # self.scan_indices = [68]
        for i_scan, scan_idx in enumerate(self.scan_indices):
            logger.info('({}/{}) Scan index = {}'.format(i_scan + 1, len(self.scan_indices), scan_idx))
            params = self.get_parameters(scan_idx)
            params = self.process_positions(scan_idx, params)
            if len(params['indices_to_keep']) == 0:
                logging.warning('No positions are left after filtering.')
                continue
            data = self.get_diffraction_data(scan_idx, params)
            labels = self.get_labels(scan_idx, params)
            self.write_diffraction_data(data)
            self.write_label(labels)
        self.finish_output_file()

if __name__ == '__main__':
    def transform(img, target_shape=(128, 128), center=(268, 503)):
        half_target_shape = [target_shape[i] // 2 for i in range(2)]
        img = img[:,
                  center[0] - half_target_shape[0]:center[0] - half_target_shape[0] + target_shape[0],
                  center[1] - half_target_shape[1]:center[1] - half_target_shape[1] + target_shape[1]]
        return img

    target_shape = (128, 128)
    gen = FromPtychoShelvesDatasetGenerator(data_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/train/results/ML_recon',
                              label_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/train/results/ML_recon',
                              output_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/train/data_train_batchnorm.h5',
                              target_shape=target_shape,
                              target_psize_nm=20,
                              transform_func=transform,
                              transform_func_kwargs={'target_shape': target_shape},
                              standardize_data=True,
                              standardize_data_across_samples=True)
    gen.debug = True
    gen.build()
    gen.run()