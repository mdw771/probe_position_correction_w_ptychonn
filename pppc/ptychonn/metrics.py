import torch


def total_variation(arr):
    assert len(arr.shape) == 4
    tv = torch.mean(torch.abs(arr[:, :, 1:, :] - arr[:, :, :-1, :]), dim=(-1, -2))
    tv = tv + torch.mean(torch.abs(arr[:, :, :, 1:] - arr[:, :, :, :-1]), dim=(-1, -2))
    tv = torch.mean(tv)
    return tv
