import json
import numpy as np

import glob
import os


walltime_dict = {}

test_folder_list = glob.glob('outputs/test*')
for test_folder in test_folder_list:
    flist = glob.glob(os.path.join(test_folder, 'walltime*.json'))
    for f in flist:
        walltimes = json.load(open(f, 'r'))
        for func_name, time in walltimes.items():
            if func_name not in walltime_dict.keys():
                walltime_dict[func_name] = []
            walltime_dict[func_name].append((f, time))
            
print('Medians:')
for func_name, file_times in walltime_dict.items():
    file_list = np.sort([x[0] for x in file_times])
    time_list = np.sort([x[1] for x in file_times])
    median_ind = len(time_list) // 2
    print('{}: {} s ({})'.format(func_name, time_list[median_ind], file_list[median_ind]))
    