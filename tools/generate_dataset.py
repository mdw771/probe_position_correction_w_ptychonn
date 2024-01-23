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

from pppc.io import save_positions_to_csv

logger = logging
logger.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)


class ReconDataNotFoundError(Exception):
    pass


class DatasetGenerator:

    def __init__(self, data_path, label_path, output_path='./data_train.h5', target_shape=(128, 128),
                 target_psize_nm=20, standardize_data=True, standardize_labels=True,
                 data_dtype='float32', label_dtype='float32', transform_func=None, transform_func_kwargs=None,
                 mode='train',
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
        self.standardize_labels = standardize_labels
        self.f_out = None
        self.transform_func = transform_func
        self.transform_func_kwargs = {} if transform_func_kwargs is None else transform_func_kwargs
        self.z_data = 0
        self.z_label = 0
        self.mode = mode
        self.debug_dir = os.path.join(os.path.dirname(self.output_path), 'dataset_generator_debug')
        self.debug = False

    def initialize_output_file(self):
        self.f_out = h5py.File(self.output_path, 'w')
        grp = self.f_out.create_group('data')
        dset = grp.create_dataset('real_phase', shape=tuple(self.data_shape), maxshape=tuple(self.data_shape),
                                  dtype=self.label_dtype)
        dset = grp.create_dataset('real_magnitude', shape=tuple(self.data_shape), maxshape=tuple(self.data_shape),
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

    @staticmethod
    def resize_3d_arr(arr, scaling_factor):
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

    @staticmethod
    def rescale_diffraction_data_to_match_pixel_size(data, original_psize, new_psize):
        return DatasetGenerator.resize_3d_arr(data, [new_psize / original_psize] * 2)

    @staticmethod
    def crop_data_to_target_shape(arr, target_shape):
        offset = (np.array(arr.shape[1:]) - target_shape) // 2
        arr = arr[:, offset[0]:offset[0] + target_shape[0], offset[1]:offset[1] + target_shape[1]]
        return arr

    def write_metadata(self):
        self.f_out['metadata/psize_nm'][0] = self.target_psize_nm

    def write_diffraction_data(self, data):
        self.f_out['data/reciprocal'][self.z_data:self.z_data + data.shape[0]] = data
        self.z_data += data.shape[0]

    def write_label(self, label_ph, label_mag):
        self.f_out['data/real_phase'][self.z_label:self.z_label + label_ph.shape[0]] = label_ph
        self.f_out['data/real_magnitude'][self.z_label:self.z_label + label_mag.shape[0]] = label_mag
        self.z_label += label_ph.shape[0]

    def write_positions(self, params):
        pos = params['pos_m']
        self.f_out['metadata'].create_dataset('pos_m', data=pos)

    def finish_output_file(self):
        self.f_out['data/reciprocal'].resize([self.z_data, *self.target_shape])
        self.f_out['data/real_phase'].resize([self.z_label, *self.target_shape])
        self.f_out['data/real_magnitude'].resize([self.z_label, *self.target_shape])
        self.f_out.close()
        print('Actual storage space for diffraction data: {} GB.'.format(
            self.calculate_storage_space([self.z_data, *self.target_shape], self.data_dtype)))
        print('Actual storage space for labels: {} GB.'.format(
            self.calculate_storage_space([self.z_label, *self.target_shape], self.label_dtype)))
        print('Final number of diffraction patterns: {}.'.format(self.z_data))


class BNPDatasetGenerator(DatasetGenerator):

    def __init__(self, data_path, label_path, mda_path, output_path='./data_train.h5',
                 target_shape=(256, 256), target_psize_nm=20, standardize_data=True, standardize_labels=True,
                 transform_func=None, transform_func_kwargs=None, mode='train',
                 *args, **kwargs):
        """
        For generating datasets from Bionanoprobe's ptychography data.

        :param data_path: str. Path to the folder that contains diffraction data.
        :param label_path: str. Path to the folder that contains reconstructed object functions (phases).
        :param transform_func: Callable. A function that transform a 3D array into the desired shape.
        """
        super().__init__(data_path, label_path, output_path, target_shape, target_psize_nm,
                         standardize_data=standardize_data, standardize_labels=standardize_labels,
                         transform_func=transform_func,
                         transform_func_kwargs=transform_func_kwargs,
                         mode=mode,
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
                 subtract_data_mean=False, standardize_labels_across_samples=True,
                 transform_func=None, transform_func_kwargs=None, mode='train',
                 transform_positions_to_snake_path=True,
                 allow_using_reconstructions_with_different_psize=False,
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
                         standardize_labels=standardize_labels_across_samples,
                         transform_func=transform_func,
                         transform_func_kwargs=transform_func_kwargs,
                         mode=mode,
                         *args, **kwargs)
        self.scan_indices = []
        self.transform_positions_to_snake_path = transform_positions_to_snake_path
        self.subtract_data_mean = subtract_data_mean
        self.allow_using_reconstructions_with_different_psize = allow_using_reconstructions_with_different_psize
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

    @staticmethod
    def get_snake_patter_reordering_indices(pos):
        inds = []
        y_pos = pos[:, 0]
        pos_grad = y_pos[1:] - y_pos[:-1]
        pos_grad = np.abs(pos_grad)
        new_line_ind = np.where(pos_grad > 0.5)[0] + 1
        new_line_ind = np.concatenate([[0], new_line_ind, [len(pos)]])
        row = 0
        while True:
            if row % 2 == 0:
                inds += list(range(new_line_ind[row], new_line_ind[row + 1]))
            else:
                inds += list(range(new_line_ind[row + 1] - 1, new_line_ind[row] - 1, -1))
            if row == len(new_line_ind) - 2:
                break
            row += 1
        return inds

    def get_parameters(self, scan_idx):
        scan_folder = os.path.join(self.data_path, 'fly{:03}'.format(scan_idx))
        f_param, s = self.choose_h5_file(scan_folder, type='para', return_size=True)
        f_param = h5py.File(f_param, 'r')
        psize_m = f_param['dx'][0]
        pos_x_m = f_param['ppX'][...]
        pos_y_m = f_param['ppY'][...]
        nx = f_param['N_scan_x'][0]
        ny = f_param['N_scan_y'][0]
        params = {}
        params['nominal_psize_nm'] = psize_m * 1e9
        params['psize_nm'] = params['nominal_psize_nm']
        params['pos_m'] = np.stack([pos_y_m, pos_x_m], axis=-1)
        params['pos_px'] = params['pos_m'] / psize_m
        params['reordering_indices'] = list(range(len(pos_y_m)))
        if self.transform_positions_to_snake_path:
            params['reordering_indices'] = self.get_snake_patter_reordering_indices(params['pos_px'])
            params['pos_m'] = np.take(params['pos_m'], params['reordering_indices'], axis=0)
            params['pos_px'] = np.take(params['pos_px'], params['reordering_indices'], axis=0)
        params['nominal_probe_size'] = s
        params['probe_size'] = s
        params['nx'] = nx
        params['indices_to_keep'] = list(range(len(pos_y_m)))
        return params

    def clean_data(self, data, do_percentile_thresholding=True):
        data_mean = np.mean(data, axis=0)
        data[data > data_mean * 80] = 0
        if do_percentile_thresholding:
            thrsh1 = np.percentile(data, 99.99999)
            thrsh2 = np.percentile(data, 99.99)
            if thrsh1 > thrsh2 * 4:
                thrsh = np.percentile(data, 99.9995)
                data[data > thrsh] = 0
        return data

    def clean_data_std(self, data, sigma_multiple=10):
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        mask = (data > data_mean - sigma_multiple * data_std) * (data > data_mean + sigma_multiple * data_std)
        data[mask] = np.tile(data_mean, [data.shape[0], 1, 1])[mask]
        return data

    def get_diffraction_data(self, scan_idx, params):
        scan_folder = os.path.join(self.data_path, 'fly{:03}'.format(scan_idx))
        f_dp = self.choose_h5_file(scan_folder, type='dp')
        f_dp = h5py.File(f_dp, 'r')
        logger.info('Loading diffraction data...')
        dat = f_dp['dp'][...]
        psize_nm = params['nominal_psize_nm']
        logger.info('Transforming diffraction data...')
        if not np.allclose(dat.shape[1:], self.target_shape):
            orig_shape = dat.shape[1:]
            dat = self.crop_data_to_target_shape(dat, self.target_shape)
            psize_nm = psize_nm / np.mean(self.target_shape / np.array(orig_shape))
        dat = self.rescale_diffraction_data_to_match_pixel_size(dat, psize_nm, self.target_psize_nm)
        if self.transform_positions_to_snake_path:
            dat = np.take(dat, params['reordering_indices'], axis=0)
        dat = dat[params['indices_to_keep']]
        if self.debug:
            tifffile.imwrite(os.path.join(self.debug_dir, 'data_raw_{:03}.tiff'.format(scan_idx)), dat)
        dat = self.clean_data(dat)
        if self.debug:
            tifffile.imwrite(os.path.join(self.debug_dir, 'data_cleaned_{:03}.tiff'.format(scan_idx)), dat)
        if self.subtract_data_mean:
            dat = dat - np.mean(dat, axis=0)
        if self.standardize_data:
            # dat = (dat - np.mean(dat, axis=(1, 2), keepdims=True)) / (np.std(dat, axis=(1, 2), keepdims=True) + 1e-3)
            dat = (dat - np.mean(dat)) / np.std(dat)
        dat = self.clean_data_std(dat, sigma_multiple=20)
        if self.debug:
            tifffile.imwrite(os.path.join(self.debug_dir, 'data_{:03}.tiff'.format(scan_idx)), dat)
        return dat

    def get_probe(self, scan_idx, params):
        recon_file, params = self.get_reconstruction_filename(scan_idx, params,
                                                      relax_name_consistency_requirement=(
                                                              self.mode == 'test' or
                                                              self.allow_using_reconstructions_with_different_psize)
                                                      )
        probe = self.get_probe_function_from_mat_file(recon_file)
        probe = np.transpose(probe, [2, 3, 0, 1])
        orig_arr_shape = probe.shape
        probe = np.reshape(probe, [-1, *probe.shape[-2:]])
        psize_nm = params['psize_nm']
        if not np.allclose(probe.shape[1:], self.target_shape):
            orig_shape = probe.shape[1:]
            probe_real = ndi.zoom(probe.real, [1, target_shape[0] / orig_shape[0], target_shape[1] / orig_shape[1]])
            probe_imag = ndi.zoom(probe.imag, [1, target_shape[0] / orig_shape[0], target_shape[1] / orig_shape[1]])
            probe = probe_real + 1j * probe_imag
            psize_nm = psize_nm / np.mean(self.target_shape / np.array(orig_shape))
        probe = self.rescale_diffraction_data_to_match_pixel_size(probe, psize_nm, self.target_psize_nm)
        probe = np.reshape(probe, [*orig_arr_shape[:2], *self.target_shape])
        probe = np.transpose(probe, [2, 3, 0, 1])
        if self.debug:
            np.save(os.path.join(self.debug_dir, 'probe_{:03}.npy'.format(scan_idx)), probe)

    def get_reconstruction_filename(self, scan_idx, params, relax_name_consistency_requirement=False):
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
            print(glob.glob(os.path.join(roi_folder, 'ML*')))
            if relax_name_consistency_requirement:
                recon_folder = glob.glob(os.path.join(roi_folder, 'ML*'))[0]
                print('Using {} instead.'.format(recon_folder))
                s = int(re.findall(r'Ndp(\d+)', recon_folder)[-1])
                print('DP size for this reconstruction is {}.'.format(s))
                if params['nominal_probe_size'] == params['probe_size']:
                    new_psize = params['psize_nm'] * (params['nominal_probe_size'] / s)
                    print('Modifying probe size, pixel size, and positions in pixel:')
                    print('    probe_size {} -> {}'.format(params['probe_size'], s))
                    print('    psize_nm {} -> {}'.format(params['psize_nm'], new_psize))
                    params['probe_size'] = s
                    params['psize_nm'] = new_psize
                    params['pos_px'] = params['pos_m'] / (new_psize * 1e-9)
            else:
                raise ReconDataNotFoundError
        mat_files = glob.glob(os.path.join(recon_folder, 'Niter*.mat'))
        ns = [int(re.findall(r'\d+', x)[-1]) for x in mat_files]
        ns.sort()
        n = ns[-1]
        recon_file = os.path.join(recon_folder, 'Niter{}.mat'.format(n))
        return recon_file, params


    def get_probe_function_from_mat_file(self, filename):
        try:
            f = h5py.File(filename, 'r')
            dset = f['probe']
            a = dset['real'][...] + 1j * dset['imag'][...]
        except:
            a = scipy.io.loadmat(filename)['probe']
        return a

    def get_object_function_from_mat_file(self, filename):
        try:
            f = h5py.File(filename, 'r')
            dset = f['object']
            obj = dset['real'][...] + 1j * dset['imag'][...]
        except:
            obj = scipy.io.loadmat(filename)['object']
        # obj = np.transpose(obj)
        return obj

    def process_positions(self, scan_idx, params, filter_positions=True):
        recon_file, params = self.get_reconstruction_filename(scan_idx, params,
                                                      relax_name_consistency_requirement=(
                                                              self.mode == 'test' or
                                                              self.allow_using_reconstructions_with_different_psize)
                                                      )
        obj = np.angle(self.get_object_function_from_mat_file(recon_file))
        pos = params['pos_px']
        offset_pos = self.offset_positions(pos, obj.shape)
        if filter_positions:
            # Only keep positions that have enough overlap.
            filtered_pos, pos_indices_to_keep = self.filter_positions(offset_pos, params, obj.shape)
        else:
            filtered_pos = offset_pos
            pos_indices_to_keep = np.arange(len(filtered_pos)).astype(int)
        params['pos_px'] = filtered_pos
        params['indices_to_keep'] = pos_indices_to_keep
        if self.debug:
            save_positions_to_csv(pos, os.path.join(self.debug_dir, 'raw_pos_{:03}.csv'.format(scan_idx)))
            save_positions_to_csv(filtered_pos, os.path.join(self.debug_dir, 'pos_{:03}.csv'.format(scan_idx)))
        return params

    def get_labels(self, scan_idx, params):
        scan_folder = os.path.join(self.data_path, 'fly{:03}'.format(scan_idx))
        recon_file, params = self.get_reconstruction_filename(scan_idx, params,
                                                      relax_name_consistency_requirement=(
                                                         self.mode == 'test' or
                                                         self.allow_using_reconstructions_with_different_psize)
                                                     )
        obj = self.get_object_function_from_mat_file(recon_file)
        pos = params['pos_px']
        logger.info('Extracting object tiles...')
        labels_ph = self.extract_object_tiles(np.angle(obj), pos, params)
        if self.standardize_labels:
            labels_ph = (labels_ph - np.mean(labels_ph)) / np.std(labels_ph) * 0.1
        labels_mag = self.extract_object_tiles(np.abs(obj), pos, params)
        if self.standardize_labels:
            labels_mag = (labels_mag - np.mean(labels_mag)) / np.std(labels_mag)
        if self.debug:
            tifffile.imwrite(os.path.join(self.debug_dir, 'obj_ph{:03}.tiff'.format(scan_idx)), np.angle(obj))
            tifffile.imwrite(os.path.join(self.debug_dir, 'labels_ph_{:03}.tiff'.format(scan_idx)), labels_ph)
            tifffile.imwrite(os.path.join(self.debug_dir, 'labels_mag_{:03}.tiff'.format(scan_idx)), labels_mag)
        return labels_ph, labels_mag

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

    def run(self, mode='train'):
        self.write_metadata()
        if self.mode == 'train':
            self.run_generate_training_data()
        elif self.mode == 'test':
            self.run_generate_test_data()
        self.finish_output_file()

    def run_generate_training_data(self):
        # self.scan_indices = [66]
        for i_scan, scan_idx in enumerate(self.scan_indices):
            try:
                logger.info('({}/{}) Scan index = {}'.format(i_scan + 1, len(self.scan_indices), scan_idx))
                params = self.get_parameters(scan_idx)
                params = self.process_positions(scan_idx, params)
                if len(params['indices_to_keep']) == 0:
                    logging.warning('No positions are left after filtering.')
                    continue
                if self.debug:
                    try:
                        self.get_probe(scan_idx, params)
                    except:
                        print('Could not get probe function for scan index {}.'.format(scan_idx))
                data = self.get_diffraction_data(scan_idx, params)
                labels_ph, labels_mag = self.get_labels(scan_idx, params)
                self.write_diffraction_data(data)
                self.write_label(labels_ph, labels_mag)
            except ReconDataNotFoundError:
                print('Skipping scan index {}.'.format(scan_idx))

    def run_generate_test_data(self):
        for i_scan, scan_idx in enumerate(self.scan_indices):
            logger.info('({}/{}) Scan index = {}'.format(i_scan + 1, len(self.scan_indices), scan_idx))
            params = self.get_parameters(scan_idx)
            params = self.process_positions(scan_idx, params, filter_positions=False)
            if len(params['indices_to_keep']) == 0:
                logging.warning('No positions are left after filtering.')
                continue
            if self.debug:
                self.get_probe(scan_idx, params)
            data = self.get_diffraction_data(scan_idx, params)
            self.write_diffraction_data(data)
            self.write_positions(params)
            try:
                labels_ph, labels_mag = self.get_labels(scan_idx, params)
                self.write_label(labels_ph, labels_mag)
            except:
                print('Could not generate ground truths for test data.')

if __name__ == '__main__':
    def transform(img, target_shape=(128, 128), center=(268, 503)):
        half_target_shape = [target_shape[i] // 2 for i in range(2)]
        img = img[:,
                  center[0] - half_target_shape[0]:center[0] - half_target_shape[0] + target_shape[0],
                  center[1] - half_target_shape[1]:center[1] - half_target_shape[1] + target_shape[1]]
        return img

    target_shape = (128, 128)
    # gen = FromPtychoShelvesDatasetGenerator(data_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/train/results/ML_recon',
    #                                         label_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/train/results/ML_recon',
    #                                         output_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/train/data_train_std_meanSub_data_std_labels_large.h5',
    #                                         # output_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/train/test.h5',
    #                                         target_shape=target_shape,
    #                                         target_psize_nm=20,
    #                                         transform_func=transform,
    #                                         transform_func_kwargs={'target_shape': target_shape},
    #                                         standardize_data=True,
    #                                         # standardize_data=False,
    #                                         subtract_data_mean=True,
    #                                         standardize_labels_across_samples=True,
    #                                         allow_using_reconstructions_with_different_psize=True,
    #                                         transform_positions_to_snake_path=False)
    gen = FromPtychoShelvesDatasetGenerator(data_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/test/results/ML_recon',
                                            label_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/test/results/ML_recon',
                                            output_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/test/data_085.h5',
                                            # output_path='/data/programs/probe_position_correction_w_ptychonn/workspace/bnp/data/train/data_77.h5',
                                            target_shape=target_shape,
                                            target_psize_nm=20,
                                            transform_func=transform,
                                            transform_func_kwargs={'target_shape': target_shape},
                                            standardize_data=True,
                                            # standardize_data=False,
                                            subtract_data_mean=True,
                                            standardize_labels_across_samples=True,
                                            mode='test',
                                            transform_positions_to_snake_path=True)
    gen.debug = True
    gen.build()
    gen.run()
