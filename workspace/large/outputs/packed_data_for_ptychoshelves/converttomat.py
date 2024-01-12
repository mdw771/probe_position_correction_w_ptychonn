from unittest import result
import tike.ptycho.io
import tike.constants
import h5py
import numpy as np
import toml
import scipy.io

import typing
import warnings
import argparse
import pathlib
import glob


def read_ptychodus(
    folder: str,
    detector_dist: typing.Union[int, None] = None,
    beam_center_y: typing.Union[int, None] = None,
    beam_center_x: typing.Union[int, None] = None,
    gap_value: typing.Union[int, None] = 0,
    parameters=None,
):
    data = np.load(folder / "diffraction.npy")
    data = np.fft.ifftshift(data, axes=(-2, -1))

    scan = np.genfromtxt(folder / "positions.csv", delimiter=",")
    # convert scan to pixel
    scan = scan / 8e-9

#    scan = tike.ptycho.io.position_units_to_pixels(
#        scan,
#        parameters["detector_dist"],
#        data.shape[-1],
#        parameters["detector_pixel_width"],
#        parameters["beam_energy"],
#    )

    probe = np.load(folder / "probe.npy")[None, None, None, ...]
    eigen_probe = None

    if len(scan) != len(data):
        warnings.warn(
            "The number of scan positions and diffraction patterns are not equal. One will be truncated to match the shorter one."
        )
        num_frames = min(len(scan), len(data))
        scan = scan[:num_frames]
        data = data[:num_frames]

    return scan, data, probe, eigen_probe


def to_mat(
    parameters,
    result_dir,
    roi="0",
):
    print("Creating geometry")

    # scan is supposed to be in pixel
    scan, data, probe, _ = read_ptychodus(
        result_dir,
        detector_dist=parameters["detector_dist"],
        parameters=parameters,
    )
    data = np.fft.ifftshift(data, axes=(-1, -2))

    lam = tike.constants.wavelength(parameters["beam_energy"] / 1000) / 100

    # dx = 1 / tike.ptycho.io.position_units_to_pixels(
    #     1,
    #     parameters["detector_dist"],
    #     data.shape[-1],
    #     parameters["detector_pixel_width"],
    #     parameters["beam_energy"],
    # )
    dx = 8e-9

    # shift positions to center around (0,0)
    scan = scan - (np.max(scan, axis=0) + np.min(scan, axis=0)) / 2

    scan = scan * dx

    probe = np.transpose(probe[0])
    print(probe.shape)

    scipy.io.savemat(
        result_dir / "probe.mat",
        dict(probe=probe),
    )

    for probe_file in glob.glob(str(result_dir / "probe*.npy")):
        mat_file = str(probe_file)[:-4] + ".mat"
        print((str(probe_file) + " -> " + str(mat_file)))
        probe = np.load(probe_file)[None, None, ...]
        probe = np.transpose(probe[0])
        scipy.io.savemat(
            mat_file,
            dict(probe=probe),
        )

    ##################### save data ##########################
    print("saving " + "data_roi" + roi + ".hdf5" + " to " + str(result_dir))
    # save diffraction pattern
    f = h5py.File(result_dir / ("data_roi" + roi + "_dp.hdf5"), "w")
    f.create_dataset(
        "dp", shape=data.shape, dtype="float64", data=data, compression="gzip"
    )
    f.close()
    # save other parameters
    f = h5py.File(result_dir / ("data_roi" + roi + "_para.hdf5"), "w")
    f.create_dataset("angle", shape=(1,), dtype="float64", data=0)
    f.create_dataset("lambda", shape=(1,), dtype="float64", data=lam)
    f.create_dataset("dx", shape=(1,), dtype="float64", data=dx)
    f.create_dataset("ppY", shape=scan[:, 0].shape, dtype="float64", data=scan[:, 0])
    f.create_dataset("ppX", shape=scan[:, 1].shape, dtype="float64", data=scan[:, 1])
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs='+', type=pathlib.Path)
    args = parser.parse_args()

    for d in args.directories:
        with open(d / "parameters.toml", "r") as f:
            parameters = toml.load(f)
        to_mat(
            parameters["data"],
            result_dir=d,
        )
