import os
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from experiment_handler.imu_data_reader import get_imu_data
from experiment_handler.label_data_reader import read_experiment_phases
from experiment_handler.finder import find_all_imu_files
from feature_calculations.window_generator import get_windows
from feature_calculations.common import get_values_in_window


def compute_features(values):
    """
    Compute IMU features for the given array (already windowed)

    Parameters
    ----------
    values: numpy array (n x 17)
        IMU data with columns order:  <timestamp>, <ax>, <ay>, <az>, <gx>, <gy>, <gz>, <mx>, <my>, <mz>, <roll>, <pitch>, <yaw>, <qx>, <qy>, <gz>, <qw>

    Returns
    -------
        feature values (1 x ? )
            Extracted features with in the following order: ?? TODO
    """
    features = np.zeros((1, 10))

    # TODO: compute IMU features

    return features


def generate_imu_features(imu_file, output_dir, window_method, interval_start=None, interval_end=None, save_as_csv=False):
    print("Start processing " + imu_file)
    source_name = os.path.basename(imu_file).split(".")[0]
    exp_root = os.path.dirname(os.path.dirname(imu_file))

    # Access the raw signal:
    raw_signal = get_imu_data(exp_root, source_name, interval_start, interval_end, "video")

    # If no interval defined, used start and end of the signal:
    if interval_start is None and len(raw_signal) > 0:
        interval_start = raw_signal[0, 0]

    if interval_end is None and len(raw_signal) > 0:
        interval_end = raw_signal[-1, 0]

    # Calculate window sizes:
    windows = get_windows(interval_start, interval_end, window_method, source=imu_file)

    features = []

    for window in windows:
        current_values = get_values_in_window(raw_signal, window[0], window[2])
        current_features = compute_features(current_values)

        current_row = np.zeros((1, 4 + current_features.shape[1]))
        current_row[0, 0] = window[0]
        current_row[0, 1] = window[1]
        current_row[0, 2] = window[2]
        current_row[0, 3] = -100 # dummy label
        current_row[0, 4:] = current_features

        features.append(current_row)

    features = np.array(features).reshape(len(features), -1)

    output_file = os.path.join(output_dir, source_name + "_" + window_method + "_empty-labels")
    np.save(output_file + ".npy", features)
    if save_as_csv:
        with open(output_file + ".csv", 'wb') as f:
            np.savetxt(f, features)

    print("Finished processing " + imu_file)


if __name__ == '__main__':
    start = datetime.now()

    parser = argparse.ArgumentParser(
        description='Run imu feature extraction.')
    parser.add_argument(
        '-p', '--path_to_experiment',
        metavar='PATH_TO_EXP',
        type=str,
        help='Path to experiment folder',
        default="/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    )
    args = parser.parse_args()

    experiment_root = args.path_to_experiment
    # TODO: parse additinal arguments (window size, window method)
    start = None
    end = None
    window_method = "SW-5000-1000"

    # Generate output dir:
    output_dir = os.path.join(experiment_root, "processed_data", "imu_features")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imu_files = find_all_imu_files(experiment_root)

    if start is None and end is None:
        # read experiment phases to generate features in that interval
        experiment_phases = read_experiment_phases(experiment_root)
        start = experiment_phases['assembly'][0]
        end = experiment_phases['disassembly'][1]

    arguments = []
    for imu_file in imu_files:
        arguments.append((imu_file, output_dir, window_method, start, end, True))
        #generate_imu_features(imu_file, output_dir, window_method, interval_start=start, interval_end=end, save_as_csv=True)
        #exit()

    p = Pool(4)
    p.starmap(generate_imu_features, arguments)
