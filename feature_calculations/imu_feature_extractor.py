import os, sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

from scipy.signal import butter, lfilter


if __package__ is None:
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from experiment_handler.imu_data_reader import get_imu_data
from experiment_handler.label_data_reader import read_experiment_phases
from experiment_handler.finder import find_all_imu_files
from feature_calculations.window_generator import get_windows
from feature_calculations.common import get_values_in_window


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)

    return y


def zc(d, center=True):
    'calculate the count of d crossing zero.Center=True standardizes the data to mean=0'
    
    if center:
        d = d - np.mean(d)
        
    return np.sum(np.diff(np.signbit(d),axis=0),axis=0)
    

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

    d = pd.DataFrame(values)
    
    # TODO: compensate for yaw/pitch/roll shifts from 0 to 360 degrees ?
    # d.loc[33000:36000,[11]].diff().cumsum().plot()
    
    # combined data vectors
    d['a'] = d.loc[:,[1,2,3]].pow(2).sum(1).pow(0.5) # take the root sum of squares of acceleration
    d['g'] = d.loc[:,[4,5,6]].pow(2).sum(1).pow(0.5) # take the root sum of squares of gyro
    d['m'] = d.loc[:,[7,8,9]].pow(2).sum(1).pow(0.5) # take the root sum of squares of magnetic field
    d['hpy'] = d.loc[:,[10,11,12]].pow(2).sum(1).pow(0.5) # take the root sum of squares of head pitch & yaw
    
    
    # now calculate features
    # zero crossings
    zc = d.sub(d.mean(0)).apply(np.signbit,0).diff(0).sum(0)
    
    
    # TODO: compute IMU features

    return features


def generate_imu_features(imu_file, output_dir, window_method, interval_start=None, interval_end=None, save_as_csv=False):
    print("Start processing " + imu_file)
    source_name = os.path.basename(imu_file).split(".")[0]
    exp_root = os.path.dirname(os.path.dirname(imu_file))

    # Access the raw signal:
    raw_signal = get_imu_data(exp_root, source_name, interval_start, interval_end, "video")

    # TODO: interpolate the data to be evenly sampled   
    # TODO: low-pass filter the data to comply with Nyquist
    

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
