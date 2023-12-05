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
                               'sift': SIFTRegistrationAlgorithm}
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
    def __init__(self, *args, **kwargs):
        self.status_dict = {'ok': 0, 'bad_fit': 1, 'drift': 2}
        self.status = 0
        self.random_seed = 123

    def run(self, previous, current, *args, **kwargs):
        pass


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
    def __init__(self, max_shift=7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_shift = max_shift
        self.starting_max_shift = min(self.max_shift, 30)
        self.drift_threshold = 60

    def run(self, previous, current, *args, **kwargs):
        self.status = self.status_dict['ok']
        current_max_shift = self.starting_max_shift
        offset = [np.inf, np.inf]
        while (self.status in [self.status_dict['ok'], self.status_dict['bad_fit']] and
               current_max_shift <= self.max_shift):
            offset, coeffs, min_error = self.run_with_current_max_shift(previous, current, current_max_shift)
            self.check_fitting_result(coeffs, min_error)
            if self.status == self.status_dict['ok']:
                break
            if current_max_shift == self.max_shift:
                current_max_shift += 10
            else:
                current_max_shift = min(current_max_shift + 10, self.max_shift)
                logger.info('Result failed quality check, so I am increasing max shift to {}. (offset = {}, a = {}, '
                            'b = {}, min_error = {})'.format(current_max_shift, offset, *coeffs[:2], min_error))
        self.check_offset(offset)
        if self.status == self.status_dict['drift']:
            logger.info('Offset magnitude is very large ({}), which might be unreliable. '.format(offset))
        return offset

    def run_with_current_max_shift(self, previous, current, max_shift):
        result_table = np.zeros([(2 * max_shift + 1) ** 2, 3])
        i = 0
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                previous_r = np.roll(np.roll(previous, dy, axis=0), dx, axis=1)
                im1 = previous_r[max_shift:-max_shift, max_shift:-max_shift]
                im2 = current[max_shift:-max_shift, max_shift:-max_shift]
                # result_table[i] = dy, dx, skimage.metrics.structural_similarity(im1, im2, data_range=6.28)
                result_table[i] = dy, dx, skimage.metrics.mean_squared_error(im1, im2)
                i += 1
        result_table = np.stack(result_table)
        min_error = np.min(result_table[:, 2])
        offset, coeffs = self.__class__._fit_quadratic_peak_in_error_map(result_table, return_coeffs=True)
        # Probe position offset is opposite to image offset.
        offset = -np.array(offset)
        return offset, coeffs, min_error

    def check_fitting_result(self, coeffs, min_error):
        a, b = coeffs[:2]
        if a < 1e-3 or b < 1e-3:
            self.status = self.status_dict['bad_fit']
            return
        if min_error > 0.5:
            self.status = self.status_dict['bad_fit']
            return
        self.status = self.status_dict['ok']

    def check_offset(self, offset):
        if np.count_nonzero(abs(np.array(offset)) > self.drift_threshold):
            self.status = self.status_dict['drift']
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
    def __init__(self, *args, outlier_removal_method='kmeans', **kwargs):
        """
        SIFT registration.

        :param outlier_removal_method: str. Can be 'kmeans', 'isoforest', 'ransac'.
        """
        super().__init__(*args, **kwargs)
        self.outlier_removal_method = outlier_removal_method

    def run(self, previous, current, *args, **kwargs):
        self.status = self.status_dict['ok']
        feature_extractor = skimage.feature.SIFT()
        feature_extractor.detect_and_extract(previous)
        keypoints_prev = feature_extractor.keypoints
        descriptors_prev = feature_extractor.descriptors
        feature_extractor.detect_and_extract(current)
        keypoints_curr = feature_extractor.keypoints
        descriptors_curr = feature_extractor.descriptors
        matches = skimage.feature.match_descriptors(descriptors_prev, descriptors_curr)
        matched_points_prev = np.take(keypoints_prev, matches[:, 0], axis=0)
        matched_points_curr = np.take(keypoints_curr, matches[:, 1], axis=0)

        # fig, ax = plt.subplots(1, 1)
        # skimage.feature.plot_matches(ax, previous, current, keypoints_prev, keypoints_curr, matches)

        # Remove outliers
        majority_inds, outlier_removal_result = self.find_majority_pairs(matched_points_prev, matched_points_curr)
        matched_points_prev = matched_points_prev[majority_inds]
        matched_points_curr = matched_points_curr[majority_inds]
        matches = matches[majority_inds]

        # Just calculate the averaged offset since we only want translation
        offset = np.mean(matched_points_prev - matched_points_curr, axis=0)

        # self.check_clustering_quality(outlier_removal_result)
        self.check_offset_quality(previous, current, offset)
        # fig, ax = plt.subplots(1, 1)
        # skimage.feature.plot_matches(ax, previous, current, keypoints_prev, keypoints_curr, matches)
        # plt.show()

        return offset

    def find_majority_pairs(self, matched_points_prev, matched_points_curr, *args, **kwargs):
        if self.outlier_removal_method == 'kmeans':
            majority_inds, res = self.find_majority_pairs_kmeans(matched_points_prev, matched_points_curr)
        elif self.outlier_removal_method == 'isoforest':
            majority_inds, res = self.find_majority_pairs_isoforest(matched_points_prev, matched_points_curr)
        elif self.outlier_removal_method == 'ransac':
            majority_inds, res = self.find_majority_pairs_ransac(matched_points_prev, matched_points_curr,
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
            self.status = self.status_dict['bad_fit']
            logger.info('Non-dominating majority cluster: {} vs {}'.format(counts[-1], counts[-2]))
            return
        if counts[-1] < 4:
            self.status = self.status_dict['bad_fit']
            logger.info('Too few majority matches ({}).'.format(counts[-1]))
            return

    def check_offset_quality(self, prev, curr, offset):
        image_offset = -offset
        prev_shifted = np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(prev), image_offset)).real

        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(prev)
        # ax[0, 1].imshow(curr)
        # ax[1, 0].imshow(prev_shifted)
        # ax[1, 1].imshow(curr)
        # plt.show()

        sy, sx = 0, 0
        ey, ex = prev.shape
        if image_offset[0] > 0:
            sy = int(np.ceil(image_offset[0]))
        elif image_offset[0] < 0:
            ey = prev.shape[0] - int(np.ceil(-image_offset[0]))
        if image_offset[1] > 0:
            sx = int(np.ceil(image_offset[1]))
        elif image_offset[1] < 0:
            ex = prev.shape[1] - int(np.ceil(-image_offset[1]))
        error = np.mean((prev_shifted[sy:ey, sx:ex] - curr[sy:ey, sx:ex]) ** 2)
        if error > 0.3:
            logger.info('Large error after applying offset ({}).'.format(error))
            self.status = self.status_dict['drift']
