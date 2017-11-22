from experiment_handler.imu_data_reader import get_imu_data
from experiment_handler.finder import find_all_imu_files
import os
import matplotlib.pyplot as plt


def plot_imu_signals_for_experiment(exp_root, figsize=(10, 10), reference_time="video"):
    imu_files = find_all_imu_files(exp_root)

    for imu_file_path in imu_files:
        source = os.path.basename(imu_file_path).split('.')[0]
        data = get_imu_data(exp_root, source, start=None, end=None, reference_time=reference_time, convert_time=True)

        f, axarr = plt.subplots(4, sharex=True, figsize=figsize)

        index = 1
        axarr[0].plot(data[:, 0], data[:, index])
        axarr[0].plot(data[:, 0], data[:, index + 1])
        axarr[0].plot(data[:, 0], data[:, index + 2])
        axarr[0].grid()
        axarr[0].set_title(source + ' - Acceleration')

        index = 4
        axarr[1].plot(data[:, 0], data[:, index])
        axarr[1].plot(data[:, 0], data[:, index + 1])
        axarr[1].plot(data[:, 0], data[:, index + 2])
        axarr[1].grid()
        axarr[1].set_title(source + ' - Gyro')

        index = 7
        axarr[2].plot(data[:, 0], data[:, index])
        axarr[2].plot(data[:, 0], data[:, index + 1])
        axarr[2].plot(data[:, 0], data[:, index + 2])
        axarr[2].grid()
        axarr[2].set_title(source + ' - Magnetic')

        index = 10
        axarr[3].plot(data[:, 0], data[:, index])
        axarr[3].plot(data[:, 0], data[:, index + 1])
        axarr[3].plot(data[:, 0], data[:, index + 2])
        axarr[3].grid()
        axarr[3].set_title(source + ' - Euler angles')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"

    plot_imu_signals_for_experiment(exp_root)