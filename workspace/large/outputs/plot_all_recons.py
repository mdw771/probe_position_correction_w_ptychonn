import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib
import matplotlib.font_manager as fm
import tifffile


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
plt.viridis()


folder_list = glob.glob('test*')
for folder in folder_list:
    flist = glob.glob(os.path.join(folder, 'rpie*.npy'))
    for f in flist:
        img = np.load(f)
        img = np.angle(img)
        img[np.isnan(img)] = 0
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, vmin=-1, vmax=1)
        #ax.tick_params(axis='both', which='major', labelsize=30)
        #ax.set_xticks(list(range(0, img.shape[1], 300)))
        #ax.set_yticks(list(range(0, img.shape[1], 300)))
        ax.set_xticks([])
        ax.set_yticks([])
        fontprops = fm.FontProperties(size=26)
        scalebar = AnchoredSizeBar(ax.transData,
                           200, '200 pixels', 'lower left',
                           pad=0.3,
                           color='black',
                           frameon=False,
                           size_vertical=15 / 1140 * img.shape[0],
                           sep=5,
                           fontproperties=fontprops,
                           fill_bar=True)
        ax.add_artist(scalebar)
        #plt.show()
        plt.savefig(os.path.join(os.path.splitext(f)[0] + '.pdf'), transparent=True)
        plt.close()

