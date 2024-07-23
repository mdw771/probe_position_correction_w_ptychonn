import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage.registration
import tifffile
import scipy.ndimage as ndi
import xommons.analysis

import os
import sys


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
plt.viridis()


def comapre_recons(images, labels, linestyles=('--', ':', None)):
    assert labels[0] == 'Nominal refined', 'The first image should be the baseline.'
    common_shape = [min([x.shape[0] for x in images]), min([x.shape[1] for x in images])]
    images = [x[:common_shape[0], :common_shape[1]] for x in images]
    
    images_shifted = [images[0]]
    for i in range(1, len(images)):
        shift = skimage.registration.phase_cross_correlation(images[i], images[0], upsample_factor=100)[0]
        print(shift)
        if labels[i] == 'Predicted unrefined':
            shift[0] += 25
        elif labels[i] == 'Predicted refined':
            shift[0] += 20
        image_shifted = np.real(np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(images[i]), shift)))
        images_shifted.append(image_shifted)
        
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(images[i] - images[0])
        ax[1].imshow(images_shifted[i] - images[0])
        ax[2].plot(images_shifted[i][300, :])
        ax[2].plot(images_shifted[0][300, :])
        
    plt.show()
        
    frc_lists = []
    r_list = None
    for i in range(len(images)):
        frc_list, r_list = xommons.analysis.fourier_ring_correlation(
            images_shifted[i][:, :images_shifted[i].shape[1] // 2], 
            np.fliplr(images_shifted[i][:, images_shifted[i].shape[1] // 2:])
        )
        frc_lists.append(frc_list)
    half_bit_threshold = xommons.analysis.get_half_bit_threshold_curve(len(r_list))
    
    fig, ax = plt.subplots(1, 1)
    for i in range(len(images)):
        ax.plot(r_list, frc_lists[i], label=labels[i])
    ax.plot(r_list, half_bit_threshold, label='1/2-bit threshold')
    plt.legend()
    plt.show()
    
    psd_lists = []
    f = None
    for i in range(len(images)):
        psd_list, f = xommons.analysis.calculate_radial_psd(images_shifted[i], log=False)
        psd_lists.append(psd_list)
        
    fig, ax = plt.subplots(1, 1)
    for i in range(len(images)):
        ax.semilogy(f, psd_lists[i], label=labels[i], color='gray', linestyle=linestyles[i])
    plt.xlabel('Spatial frequency (pixel$^{-1}$)')
    plt.ylabel('Radial power spectrum density')
    plt.legend(frameon=False)

    fig_recons, ax_recons = plt.subplots(1, len(images_shifted))
    slicer = [slice(common_shape[0] // 3 * 2 - 30, common_shape[0] // 3 * 2 + 80),
              slice(common_shape[1] // 2 - 50, common_shape[1] // 2 + 50),]
    for i in range(len(images_shifted)):
        ax_recons[i].imshow(images_shifted[i][*slicer])
        ax_recons[i].set_title(labels[i])


    plt.show()
    # plt.savefig(os.path.join(folder, 'psd_comparison.pdf'))


folder = 'test0_unscaled_reduced_4_22/nn_8_tol_3e-4_4e-4_variableTol_newErrorMap_unscaled'

f_baseline = os.path.join(folder, 'rpie_posCorr_1_clip_2_pos_baseline.tiff')
f_calculated = os.path.join(folder, 'rpie_posCorr_0_pos_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.tiff')
f_calculated_refined = os.path.join(folder, 'rpie_posCorr_1_clip_2_pos_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.tiff')

recon_baseline = tifffile.imread(f_baseline)
recon_calculated = tifffile.imread(f_calculated)
recon_calculated_refined = tifffile.imread(f_calculated_refined)

comapre_recons([recon_baseline, recon_calculated, recon_calculated_refined], ['Nominal refined', 'Predicted unrefined', 'Predicted refined'])
