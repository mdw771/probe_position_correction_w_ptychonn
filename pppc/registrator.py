import copy
import warnings

import numpy as np
import skimage.registration
import skimage.feature
import sklearn.cluster
import sklearn.ensemble
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from pppc.message_logger import logger


class Registrator:

    def __init__(self, method='error_map', max_shift=None, random_seed=123, **kwargs):
        self.method = method
        self.max_shift = max_shift
        self.random_seed = random_seed
        self.algorithm_dict = {'error_map': ErrorMapRegistrationAlgorithm,
                               'phase_correlation': PhaseCorrelationRegistrationAlgorithm,
                               'sift': SIFTRegistrationAlgorithm,
                               'hybrid': HybridRegistrationAlgorithm}
        self.algorithm = self.algorithm_dict[method](max_shift=self.max_shift, **kwargs)
        self.algorithm.random_seed = self.random_seed

    def run(self, previous, current):
        """
        Run registration and get offset. The returned offset is the supposed *probe position difference* of the
        current object relative to the previous one, which is opposite to the offset between the images.

        :param previous: object image of the previous scan point.
        :param current: object image of the current scan point.
        :return: np.ndarray.
        """
        offset = self.algorithm.run(previous, current)
        return offset

    def get_status(self):
        return self.algorithm.status

    def get_status_code(self, key):
        return self.algorithm.status_dict[key]


class RegistrationAlgorithm:
    def __init__(self, tol=0.3, *args, **kwargs):
        self.status_dict = {'ok': 0, 'questionable': 1, 'bad': 2}
        self.status = 0
        self.random_seed = 123
        self.tol = tol
        self.debug = False

    def run(self, previous, current, *args, **kwargs):
        pass

    def check_offset_quality(self, prev, curr, offset, update_status=True, tol=0.3):
        image_offset = -offset
        prev_shifted = np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(prev), image_offset)).real

        sy, sx, ey, ex = self.calculate_metric_region_for_shifted_image(offset, prev.shape)
        error = np.mean((prev_shifted[sy:ey, sx:ex] - curr[sy:ey, sx:ex]) ** 2)
        if error > tol and update_status:
            logger.info('Large error after applying offset ({}).'.format(error))
            self.status = self.status_dict['bad']
            # fig, ax = plt.subplots(1, 5, figsize=(13, 3))
            # ax[0].imshow(prev); ax[0].grid('both')
            # ax[1].imshow(curr); ax[1].grid('both')
            # ax[2].imshow(prev_shifted); ax[2].grid('both')
            # ax[3].imshow(prev_shifted - curr); ax[3].grid('both')
            # if isinstance(self, ErrorMapRegistrationAlgorithm):
            #     ax[4].imshow(self.error_map)
            # plt.suptitle('{} {}'.format(str(self), offset))
            # plt.tight_layout()
            # plt.show()
        else:
            # There should be enough variance in the analysis region for the result to be reliable.
            std_roi = np.std(curr[sy:ey, sx:ex])
            if std_roi < 0.2:
                logger.info('Error is low ({}) but variance is also low within the ROI ({}).'.format(error, std_roi))
                self.status = self.status_dict['questionable']
                fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                ax[0].imshow(prev_shifted[sy:ey, sx:ex])
                ax[1].imshow(curr[sy:ey, sx:ex])
                plt.show()
                print(np.std(curr[sy:ey, sx:ex]))
                print(offset)
        return error

    def calculate_metric_region_for_shifted_image(self, offset, image_shape):
        image_offset = -np.array(offset)
        sy, sx = 0, 0
        ey, ex = image_shape
        if image_offset[0] > 0:
            sy = int(np.ceil(image_offset[0]))
        elif image_offset[0] < 0:
            ey = image_shape[0] - int(np.ceil(-image_offset[0]))
        if image_offset[1] > 0:
            sx = int(np.ceil(image_offset[1]))
        elif image_offset[1] < 0:
            ex = image_shape[1] - int(np.ceil(-image_offset[1]))
        return sy, sx, ey, ex

class HybridRegistrationAlgorithm(RegistrationAlgorithm):
    def __init__(self, algs=('error_map_multilevel', 'error_map_expandable', 'sift'), tols=(0.08, 0.3, 0.3),
                 max_shift=7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alg_list = []
        self.alg_names = algs
        for i, alg in enumerate(algs):
            if alg == 'error_map_multilevel':
                self.alg_list.append(
                    ErrorMapRegistrationAlgorithm(subpixel=False, max_shift=max_shift, n_levels=3, tol=tols[i]))
            elif alg == 'error_map_expandable':
                self.alg_list.append(
                    ErrorMapRegistrationAlgorithm(subpixel=False, max_shift=max_shift, n_levels=1, tol=tols[i]))
            elif alg == 'sift':
                self.alg_list.append(
                    SIFTRegistrationAlgorithm(outlier_removal_method='trial_error', boundary_exclusion_length=16,
                                              tol=tols[i]))

    def run(self, previous, current, *args, **kwargs):
        offset = None
        for i in range(len(self.alg_list)):
            offset = self.alg_list[i].run(previous, current, *args, **kwargs)
            self.status = self.alg_list[i].status
            if self.status == self.status_dict['ok']:
                return offset
            if i < len(self.alg_list) - 1:
                logger.info('Switching to {}...'.format(self.alg_names[i + 1]))
        return offset
        ###
        # self.error_map_alg.debug = True
        # self.error_map_alg.run(previous, current, *args, **kwargs)
        # self.error_map_alg.debug = False
        ###

class PhaseCorrelationRegistrationAlgorithm(RegistrationAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, previous, current, *args, **kwargs):
        previous -= np.mean(previous)
        current -= np.mean(current)
        offset = skimage.registration.phase_cross_correlation(current, previous, upsample_factor=10)
        offset = offset[0]
        offset = -offset
        return offset


class ErrorMapRegistrationAlgorithm(RegistrationAlgorithm):
    def __init__(self, max_shift=7, subpixel=True, n_levels=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_shift = max_shift
        self.starting_max_shift = min(self.max_shift, 30)
        self.max_shift_multilevel = 40
        self.drift_threshold = 60
        self.subpixel = subpixel
        self.error_map = None
        self.n_levels = n_levels

    def run(self, previous, current, *args, **kwargs):
        if self.n_levels == 1:
            return self.run_expandable(previous, current, *args, **kwargs)
        else:
            return self.run_multilevel(previous, current, *args, **kwargs)

    def run_multilevel(self, previous, current, *args, **kwargs):
        self.status = self.status_dict['ok']
        offset = None
        last_level_offset = None
        for level in range(self.n_levels, 0, -1):
            level_scaler = 2 ** (level - 1)
            if level > 1:
                level_previous = ndi.zoom(previous, 1. / level_scaler)
                level_current = ndi.zoom(current, 1. / level_scaler)
            else:
                level_previous = previous
                level_current = current
            current_max_shift = self.max_shift_multilevel // level_scaler
            if level == self.n_levels:
                offset, coeffs, min_error = self.run_with_current_max_shift(level_previous, level_current,
                                                                            current_max_shift)
            else:
                y_range = np.clip(
                    [np.round(-last_level_offset[0] * 2) - 4, np.round(-last_level_offset[0] * 2) + 4],
                    a_min=-current_max_shift, a_max=current_max_shift).astype(int)
                x_range = np.clip(
                    [np.round(-last_level_offset[1] * 2) - 4, np.round(-last_level_offset[1] * 2) + 4],
                    a_min=-current_max_shift, a_max=current_max_shift).astype(int)
                offset, coeffs, min_error = self.run_with_current_max_shift(level_previous, level_current,
                                                                            y_range=y_range, x_range=x_range)
            self.check_fitting_result(coeffs, min_error)
            last_level_offset = offset
        self.check_offset(offset)
        self.check_offset_quality(previous, current, offset, tol=self.tol)
        return offset

    def run_expandable(self, previous, current, *args, **kwargs):
        self.status = self.status_dict['ok']
        offset = None
        current_max_shift = self.starting_max_shift
        offset = [np.inf, np.inf]
        while (self.status in [self.status_dict['ok'], self.status_dict['questionable']] and
               current_max_shift <= self.max_shift):
            offset, coeffs, min_error = self.run_with_current_max_shift(previous, current,
                                                                        current_max_shift)
            self.check_fitting_result(coeffs, min_error)
            if self.status == self.status_dict['ok']:
                break
            if current_max_shift == self.max_shift:
                current_max_shift += 10
            else:
                current_max_shift = min(current_max_shift + 10, self.max_shift)
                if self.subpixel:
                    logger.info(
                        'Result failed quality check, so I am increasing max shift to {}. (offset = {}, a = {}, '
                        'b = {}, min_error = {})'.format(current_max_shift, offset, *coeffs[:2], min_error))
                else:
                    logger.info('Result failed quality check, so I am increasing max shift to {}. (offset = {}, '
                                'min_error = {})'.format(current_max_shift, offset, min_error))
        self.check_offset(offset)
        self.check_offset_quality(previous, current, offset, tol=self.tol)
        return offset

    def run_with_current_max_shift(self, previous, current, max_shift=None, y_range=None, x_range=None):
        if max_shift is not None:
            y_range = [-max_shift, max_shift]
            x_range = [-max_shift, max_shift]
        len_range = [y_range[1] - y_range[0] + 1, x_range[1] - x_range[0] + 1]
        result_table = np.zeros([len_range[0] * len_range[1], 3])
        self.error_map = np.zeros(len_range)

        # Calculate the range within which error is to be calculated.
        sy = max(self.calculate_metric_region_for_shifted_image([y_range[0], 0], previous.shape)[0],
                 self.calculate_metric_region_for_shifted_image([y_range[1], 0], previous.shape)[0])
        sx = max(self.calculate_metric_region_for_shifted_image([0, x_range[0]], previous.shape)[1],
                 self.calculate_metric_region_for_shifted_image([0, x_range[1]], previous.shape)[1])
        ey = min(self.calculate_metric_region_for_shifted_image([y_range[0], 0], previous.shape)[2],
                 self.calculate_metric_region_for_shifted_image([y_range[1], 0], previous.shape)[2])
        ex = min(self.calculate_metric_region_for_shifted_image([0, x_range[0]], previous.shape)[3],
                 self.calculate_metric_region_for_shifted_image([0, x_range[1]], previous.shape)[3])

        i = 0
        for dy in range(y_range[0], y_range[1] + 1):
            for dx in range(x_range[0], x_range[1] + 1):
                previous_r = np.roll(np.roll(previous, dy, axis=0), dx, axis=1)
                im1 = previous_r[sy:ey, sx:ex]
                im2 = current[sy:ey, sx:ex]
                err = skimage.metrics.mean_squared_error(im1, im2)
                # err = skimage.metrics.structural_similarity(im1, im2, data_range=6.28)
                result_table[i] = dy, dx, err
                self.error_map[dy - y_range[0], dx - x_range[0]] = err
                i += 1
        result_table = np.stack(result_table)
        min_error = np.min(result_table[:, 2])
        if self.subpixel:
            offset, coeffs = self.__class__._fit_quadratic_peak_in_error_map(result_table, return_coeffs=True)
            # Probe position offset is opposite to image offset.
            offset = -np.array(offset)
            return offset, coeffs, min_error
        else:
            i_min_error = np.argmin(result_table[:, 2])
            offset = result_table[i_min_error, :2]
            offset = -np.array(offset)

            if self.debug:
                print(offset)
                fig, ax = plt.subplots(1, 4, figsize=(11, 3))
                ax[0].imshow(previous); ax[0].grid('both')
                ax[1].imshow(current); ax[1].grid('both')
                ax[2].imshow(np.roll(np.roll(previous, -int(offset[0]), axis=0), -int(offset[1]), axis=1)); ax[2].grid('both')
                ax[3].imshow(self.error_map); ax[3].grid('both')
                plt.tight_layout()
                plt.show()

            return offset, None, min_error

    def check_fitting_result(self, coeffs, min_error):
        if self.subpixel:
            a, b = coeffs[:2]
            if a < 1e-3 or b < 1e-3:
                self.status = self.status_dict['questionable']
                return
        if min_error > 0.3:
            self.status = self.status_dict['questionable']
            return
        self.status = self.status_dict['ok']

    def check_offset(self, offset):
        if np.count_nonzero(abs(np.array(offset)) > self.drift_threshold):
            self.status = self.status_dict['bad']
            logger.info('Offset magnitude is very large ({}), which might be unreliable. '.format(offset))
        else:
            self.status = self.status_dict['ok']

    @staticmethod
    def _fit_quadratic_peak_in_error_map(result_table, window_size=5, return_coeffs=False):
        rad = window_size // 2
        shift_range_y = [int(np.round(result_table[:, 0].min())), int(np.round(result_table[:, 0].max()))]
        shift_range_x = [int(np.round(result_table[:, 1].min())), int(np.round(result_table[:, 1].max()))]
        shift_range = [shift_range_y, shift_range_x]
        map_shape = [shift_range_y[1] - shift_range_y[0] + 1, shift_range_x[1] - shift_range_x[0] + 1]
        error_map = result_table[:, 2].reshape(map_shape)
        min_error_shift = result_table[np.argmin(result_table[:, 2])][:2]
        min_loc = [int(np.round(min_error_shift[i])) - shift_range[i][0] for i in range(2)]
        window_offset = [min_loc[0] - rad, min_loc[1] - rad]
        if window_offset[0] < 0:
            window_offset[0] = 0
        if window_offset[0] + window_size > map_shape[0]:
            window_offset[0] = map_shape[0] - window_size
        if window_offset[1] < 0:
            window_offset[1] = 0
        if window_offset[1] + window_size > map_shape[1]:
            window_offset[1] = map_shape[1] - window_size
        window = error_map[window_offset[0]:window_offset[0] + window_size,
                 window_offset[1]:window_offset[1] + window_size]
        a_mat = np.zeros([window.size, 6])
        y, x = np.mgrid[:window.shape[0], :window.shape[1]]
        y, x = y.reshape(-1), x.reshape(-1)
        a_mat[:, 0] = y ** 2
        a_mat[:, 1] = x ** 2
        a_mat[:, 2] = y
        a_mat[:, 3] = x
        a_mat[:, 4] = x * y
        a_mat[:, 5] = 1
        b_vec = window.reshape(-1)
        x_vec = np.linalg.pinv(a_mat) @ b_vec
        a, b, c, d, e, f = x_vec
        y_min = -(2 * b * c - d * e) / (4 * a * b - e ** 2)
        x_min = -(2 * a * d - c * e) / (4 * a * b - e ** 2)
        y_min += (window_offset[0] + shift_range_y[0])
        x_min += (window_offset[1] + shift_range_x[0])
        if abs(y_min) > 100 or abs(x_min) > 100:
            warnings.warn('A suspiciously large offset was detected ({}, {}). If this does not seem normal, the search '
                          'range is likely too small which caused the result to diverge. Adjust the search range above '
                          'the largest possible offset. \nSolved quadratic coefficients: {}.'.format(
                y_min, x_min, x_vec))
        if return_coeffs:
            return (y_min, x_min), (a, b, c, d, e, f)
        return y_min, x_min


class SIFTRegistrationAlgorithm(RegistrationAlgorithm):
    def __init__(self, *args, outlier_removal_method='kmeans', initial_crop_ratio=1, boundary_exclusion_length=16,
                 downsample=1, **kwargs):
        """
        SIFT registration.

        :param outlier_removal_method: str. Can be 'trial_error', 'kmeans', 'isoforest', 'ransac'.
        :param boundary_exclusion_length: int. The length of the near-boundary region of the image. When doing
               SIFT registration, if a matching pair of keypoints involve points in this region, it will be discarded.
               However, if all matches (after outlier removal) are near-boundary, they are used as they are.
        :param downsample: int. Downsampling ratio.
        """
        super().__init__(*args, **kwargs)
        self.outlier_removal_method = outlier_removal_method
        self.initial_crop_ratio = initial_crop_ratio
        self.boundary_exclusion_length = boundary_exclusion_length
        self.downsample = downsample

    def find_keypoints(self, previous, current):
        feature_extractor = skimage.feature.SIFT(n_octaves=8, upsampling=1, n_scales=3, sigma_in=0.8, sigma_min=1.2)
        feature_extractor.detect_and_extract(previous)
        keypoints_prev = feature_extractor.keypoints
        descriptors_prev = feature_extractor.descriptors
        feature_extractor.detect_and_extract(current)
        keypoints_curr = feature_extractor.keypoints
        descriptors_curr = feature_extractor.descriptors
        return keypoints_prev, keypoints_curr, descriptors_prev, descriptors_curr

    def find_matches(self, keypoints_prev, keypoints_curr, descriptors_prev, descriptors_curr):
        matches = skimage.feature.match_descriptors(descriptors_prev, descriptors_curr)
        matched_points_prev = np.take(keypoints_prev, matches[:, 0], axis=0)
        matched_points_curr = np.take(keypoints_curr, matches[:, 1], axis=0)
        return matches, matched_points_prev, matched_points_curr

    def run(self, previous, current, *args, **kwargs):
        if self.downsample > 1:
            previous = ndi.zoom(previous, 1. / self.downsample)
            current = ndi.zoom(current, 1. / self.downsample)
        matched_points_prev = []
        matched_points_curr = []
        matches = []
        if self.initial_crop_ratio < 1:
            has_enough_matches = False
            is_full_size = False
            crop_len = [int(previous.shape[i] * (1 - self.initial_crop_ratio)) for i in range(2)]
            cropped_previous = previous[crop_len[0]:-crop_len[0], crop_len[1]:-crop_len[1]]
            cropped_current = current[crop_len[0]:-crop_len[0], crop_len[1]:-crop_len[1]]
            while not has_enough_matches:
                self.status = self.status_dict['ok']
                keypoints_prev, keypoints_curr, descriptors_prev, descriptors_curr = self.find_keypoints(
                    cropped_previous, cropped_current)
                matches, matched_points_prev, matched_points_curr = self.find_matches(keypoints_prev, keypoints_curr,
                                                                                      descriptors_prev, descriptors_curr)
                if len(matches) < 2 and not is_full_size:
                    logger.info('Could not find enough matches with crop ratio {}. Using full-sized images '
                                'instead...'.format(self.initial_crop_ratio))
                    cropped_previous = previous
                    cropped_current = current
                    is_full_size = True
                else:
                    has_enough_matches = True
        else:
            self.status = self.status_dict['ok']
            keypoints_prev, keypoints_curr, descriptors_prev, descriptors_curr = self.find_keypoints(previous, current)
            matches, matched_points_prev, matched_points_curr = self.find_matches(keypoints_prev, keypoints_curr,
                                                                                  descriptors_prev, descriptors_curr)

        # fig, ax = plt.subplots(1, 1)
        # skimage.feature.plot_matches(ax, previous, current, keypoints_prev, keypoints_curr, matches)
        # plt.title('Before')
        # plt.show()

        # Remove outliers
        majority_inds, outlier_removal_result = self.find_majority_pairs(matched_points_prev, matched_points_curr,
                                                                         prev_image=previous, current_image=current)
        matched_points_prev = matched_points_prev[majority_inds]
        matched_points_curr = matched_points_curr[majority_inds]
        matches = matches[majority_inds]

        # Remove near-edge pairs. If all pairs are near-edge, just use them as they are.
        # matches_0 = copy.copy(matches)
        non_remote_inds, ss = self.find_non_remote_pairs(matched_points_prev, matched_points_curr, previous.shape,
                                                         boundary_len=self.boundary_exclusion_length)
        matched_points_prev = matched_points_prev[non_remote_inds]
        matched_points_curr = matched_points_curr[non_remote_inds]
        matches = matches[non_remote_inds]

        # Just calculate the averaged offset since we only want translation
        offset = np.mean(matched_points_prev - matched_points_curr, axis=0)

        # self.check_clustering_quality(outlier_removal_result)
        self.check_offset_quality(previous, current, offset, tol=self.tol)
        # fig, ax = plt.subplots(1, 1)
        # skimage.feature.plot_matches(ax, previous, current, keypoints_prev, keypoints_curr, matches)
        # plt.title('After')
        # plt.show()

        # if ss:
        #     fig, ax = plt.subplots(1, 1)
        #     skimage.feature.plot_matches(ax, previous, current, keypoints_prev, keypoints_curr, matches_0)
        #     plt.title('Before')
        #     plt.show()
        #     fig, ax = plt.subplots(1, 1)
        #     skimage.feature.plot_matches(ax, previous, current, keypoints_prev, keypoints_curr, matches)
        #     plt.title('After')
        #     plt.show()

        if self.downsample > 1:
            offset = offset * self.downsample

        return offset

    def run_affine(self, previous, current, *args, **kwargs):
        self.status = self.status_dict['ok']
        keypoints_prev, keypoints_curr, descriptors_prev, descriptors_curr = self.find_keypoints(previous, current)
        matches, matched_points_prev, matched_points_curr = self.find_matches(keypoints_prev, keypoints_curr,
                                                                              descriptors_prev, descriptors_curr)
        # fig, ax = plt.subplots(1, 1)
        # skimage.feature.plot_matches(ax, previous, current, keypoints_prev, keypoints_curr, matches)
        # plt.title('Before')
        # plt.show()

        # Remove outliers
        majority_inds, outlier_removal_result = self.find_majority_pairs(matched_points_prev, matched_points_curr,
                                                                         prev_image=previous, current_image=current)
        matched_points_prev = matched_points_prev[majority_inds]
        matched_points_curr = matched_points_curr[majority_inds]
        matches = matches[majority_inds]

        affine_tform = self.estimate_affine_transform(matched_points_curr, matched_points_prev)

        # if np.abs(affine_tform[1, 2]) > 5:
        #     print(affine_tform)
        #     print(np.mean(matched_points_prev - matched_points_curr, axis=0))
        #     fig, ax = plt.subplots(1, 1)
        #     skimage.feature.plot_matches(ax, previous, current, keypoints_prev, keypoints_curr, matches)
        #     plt.title('After')
        #     plt.show()

        # affine_tform = skimage.transform.estimate_transform('euclidean', matched_points_prev, matched_points_curr)
        return affine_tform

    def estimate_affine_transform(self, matched_points_curr, matched_points_prev):
        mat_curr = np.stack([matched_points_curr[:, 0], matched_points_curr[:, 1],
                             np.ones(matched_points_curr.shape[0])], axis=1)
        mat_prev = np.stack([matched_points_prev[:, 0], matched_points_prev[:, 1],
                             np.ones(matched_points_prev.shape[0])], axis=1)
        a_mat = np.linalg.pinv(mat_curr) @ mat_prev
        return a_mat.T
    
    def find_non_remote_pairs(self, matched_points_prev, matched_points_curr, image_shape, boundary_len=10):
        status = False
        mask_in_prev_y = np.logical_and(matched_points_prev[:, 0] > boundary_len,
                                        matched_points_prev[:, 0] < (image_shape[0] - boundary_len))
        mask_in_prev_x = np.logical_and(matched_points_prev[:, 1] > boundary_len,
                                        matched_points_prev[:, 1] < (image_shape[1] - boundary_len))
        mask_in_curr_y = np.logical_and(matched_points_curr[:, 0] > boundary_len,
                                        matched_points_curr[:, 0] < (image_shape[0] - boundary_len))
        mask_in_curr_x = np.logical_and(matched_points_curr[:, 1] > boundary_len,
                                        matched_points_curr[:, 1] < (image_shape[1] - boundary_len))
        mask_final = np.logical_and(
            np.logical_and(
                np.logical_and(mask_in_curr_x, mask_in_prev_x),
                mask_in_curr_y),
            mask_in_prev_y
        )
        inds = np.where(mask_final)[0]
        if len(inds) == 0:
            inds = np.arange(len(matched_points_prev)).astype('int')
        else:
            if len(matched_points_prev) > len(inds):
                status = True
        return inds, status
        
    def find_majority_pairs(self, matched_points_prev, matched_points_curr,
                            prev_image=None, current_image=None, *args, **kwargs):
        if self.outlier_removal_method == 'kmeans':
            majority_inds, res = self.find_majority_pairs_kmeans(matched_points_prev, matched_points_curr)
        elif self.outlier_removal_method == 'isoforest':
            majority_inds, res = self.find_majority_pairs_isoforest(matched_points_prev, matched_points_curr)
        elif self.outlier_removal_method == 'ransac':
            majority_inds, res = self.find_majority_pairs_ransac(matched_points_prev, matched_points_curr,
                                                                 *args, **kwargs)
        elif self.outlier_removal_method == 'trial_error':
            majority_inds, res = self.find_majority_pairs_trial_error(matched_points_prev, matched_points_curr,
                                                                      prev_image, current_image,
                                                                      *args, **kwargs)
        else:
            raise ValueError('{} is not a valid method. '.format(self.outlier_removal_method))
        return majority_inds, res

    def find_majority_pairs_isoforest(self, matched_points_prev, matched_points_curr):
        """
        Find inlying matches, and return the indices.

        :param matched_points_prev: np.ndarray.
        :param matched_points_curr: np.ndarray.
        :return: np.ndarray.
        """
        shift_vectors = matched_points_curr - matched_points_prev
        isoforest = sklearn.ensemble.IsolationForest(n_estimators=10, random_state=self.random_seed)
        res = isoforest.fit_predict(shift_vectors)
        majority_inds = np.where(res == 1)[0]
        if len(majority_inds) == 0:
            majority_inds = list(range(shift_vectors.shape[0]))
            self.status = self.status_dict['questionable']
        return majority_inds, res

    def find_majority_pairs_kmeans(self, matched_points_prev, matched_points_curr):
        """
        Cluster the shift vectors of matched point pairs into 2 clusters, and return the indices of the majority ones.

        :param matched_points_prev: np.ndarray.
        :param matched_points_curr: np.ndarray.
        :return: np.ndarray.
        """
        kmeans = sklearn.cluster.KMeans(n_clusters=min(2, matched_points_curr.shape[0]), n_init='auto',
                                        random_state=self.random_seed)
        shift_vectors = matched_points_curr - matched_points_prev
        res = kmeans.fit(shift_vectors)
        majority_cluster_ind = np.argmax(np.unique(res.labels_, return_counts=True)[1])
        majority_inds = np.where(res.labels_ == majority_cluster_ind)[0]
        return majority_inds, res

    def find_majority_pairs_trial_error(self, matched_points_prev, matched_points_curr, prev_image, current_image,
                                        error_threshold=0.5):
        offset_list = matched_points_prev - matched_points_curr
        good_inds = []
        error_list = []
        for i, offset in enumerate(offset_list[:min(10, offset_list.shape[0])]):
            # If the offset is too large, the computed error would be inaccurate because there are too few pixels
            # after excluding wrap-around regions. Thus it should be skipped.
            if np.alltrue(np.abs(offset) < 0.6 * np.array(prev_image.shape)):
                error = self.check_offset_quality(prev_image, current_image, offset, update_status=False)
                if error < error_threshold:
                    good_inds.append(i)
                    error_list.append(error)
        if len(good_inds) > 0:
            temp_points_prev = matched_points_prev[good_inds]
            temp_points_curr = matched_points_curr[good_inds]
            good_inds_2, _ = self.find_majority_pairs_ransac(temp_points_prev, temp_points_curr,
                                                             n_iters=1, sigma=3)
            if len(good_inds_2) > 0:
                good_inds = list(np.array(good_inds)[np.array(good_inds_2)])
            else:
                good_inds = []
        if len(good_inds) == 0:
            logger.info('Trial-error did not return any good candidates, thus switching to KMeans.')
            self.status = self.status_dict['questionable']
            good_inds, res = self.find_majority_pairs_kmeans(matched_points_prev, matched_points_curr)
        return good_inds, error_list

    def find_majority_pairs_ransac(self, matched_points_prev, matched_points_curr, n_iters=4, sigma=3.0):
        """
        Find inlying matches, and return the indices.

        :param matched_points_prev: np.ndarray.
        :param matched_points_curr: np.ndarray.
        :return: np.ndarray.
        """
        shift_vectors = matched_points_curr - matched_points_prev
        inlier_inds = list(range(shift_vectors.shape[0]))
        last_inliner_inds = inlier_inds
        inlier_shift_vectors = shift_vectors.copy()
        for i_iter in range(n_iters):
            std_vec = np.std(inlier_shift_vectors, axis=0)
            mean_vec = np.median(inlier_shift_vectors, axis=0)
            lb_vec = mean_vec - sigma * np.clip(std_vec, a_min=1e-3, a_max=5)
            ub_vec = mean_vec + sigma * np.clip(std_vec, a_min=1e-3, a_max=5)
            mask = np.logical_and(shift_vectors > lb_vec, shift_vectors < ub_vec)
            inlier_inds = np.where(np.alltrue(mask, axis=1))[0]
            if len(inlier_inds) == 0:
                inlier_inds = last_inliner_inds
                inlier_shift_vectors = shift_vectors[inlier_inds]
                break
            inlier_shift_vectors = shift_vectors[inlier_inds]
            last_inliner_inds = inlier_inds
        return inlier_inds, inlier_shift_vectors

    def check_clustering_quality(self, kmeans_result):
        # If the majority cluster does not dominate, the result is less confident.
        labels, counts = np.unique(kmeans_result.labels_, return_counts=True)
        counts = np.sort(counts)
        if len(counts) > 1 and (counts[-1] - counts[-2]) / counts[-2] < 0.3:
            self.status = self.status_dict['questionable']
            logger.info('Non-dominating majority cluster: {} vs {}'.format(counts[-1], counts[-2]))
            return
        if counts[-1] < 4:
            self.status = self.status_dict['questionable']
            logger.info('Too few majority matches ({}).'.format(counts[-1]))
            return

