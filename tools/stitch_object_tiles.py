import numpy as np

pos = np.genfromtxt("positions.csv", delimiter=',')
pos_x = pos[:, 1]
pos_y = pos[:, 0]
pixelsize = 8e-9  # you can change this I think
x = np.arange(pos_x.min() - 0.5e-6, pos_x.max() + 0.5e-6, pixelsize)
y = np.arange(pos_y.min() - 0.5e-6, pos_y.max() + 0.5e-6, pixelsize)

result = np.zeros((y.shape[0], x.shape[0]))
cnt = np.copy(result)
cnt1 = cnt + 1
xx = np.arange(128) * pixelsize
xx -= xx.mean()
yy = np.copy(xx)

for i in range(data.shape[0]):
    xxx = xx + pos_x[i]
    yyy = yy + pos_y[i]
    find_pha = interpolate.interp2d(xxx[40:-40], yyy[40:-40], data[i, 40:-40, 40:-40], kind='linear',
                                    fill_value=0)  # crop out 40 pixels on each end, you can try 32, 16...
    tmp = find_pha(x, y)
    cnt += tmp != 0
    result += tmp

final_result = (result / np.maximum(cnt, cnt1))[50:-50, 50:-50][:, ::-1]