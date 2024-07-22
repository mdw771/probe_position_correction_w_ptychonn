import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib
import tifffile


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 24}
plt.rc('font', **fontProperties)
plt.viridis()


folder_list = ['test0_unscaled_reduced_4_22/nn_8_tol_3e-4_4e-4_variableTol_newErrorMap_unscaled']
for folder in folder_list:
    flist = glob.glob(os.path.join(folder, 'rpie*.tiff'))
    for f in flist:
        img = tifffile.imread(f)
        img[np.isnan(img)] = 0
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, vmin=-0.9, vmax=0.5)
        #ax.tick_params(axis='both', which='major', labelsize=30)
        #ax.set_xticks(list(range(0, img.shape[1], 300)))
        #ax.set_yticks(list(range(0, img.shape[1], 300)))
        ax.set_xticks([])
        ax.set_yticks([])
        scalebar = AnchoredSizeBar(ax.transData,
                           100, '100 pixels', 'lower left',
                           pad=0.3,
                           sep=5,
                           color='black',
                           frameon=False,
                           size_vertical=10)

        ax.add_artist(scalebar)
        #plt.show()
        plt.savefig(os.path.join(os.path.splitext(f)[0] + '.pdf'), transparent=True)
        plt.close()

