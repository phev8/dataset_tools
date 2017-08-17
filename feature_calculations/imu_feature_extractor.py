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


    

def compute_features(values):
    """
    Compute IMU features for the given array (already windowed)

    Parameters
    ----------
    values: pandas dataframe (n x 20)
        IMU data with row index: <timestamp>, and columns: <ax>, <ay>, <az>, <gx>, <gy>, <gz>, <mx>, <my>, <mz>, <roll>, <pitch>, <yaw>, <qx>, <qy>, <gz>, <qw>, 
             (with combined vectors)                     ... <a>, <g>, <m>, <ypr>

    Returns
    -------
        feature values pandas (1 x ? )
            Extracted features with the column naming scheme <raw data index>_<method name>, e.g. 'ax_mean'
    """
    d = values.copy()
    details = d.describe()    
        
    # calculate features    
    F = pd.DataFrame()    

    for col in d.columns:
    
        for method in details.index: # access mean, std, min, 25%, 50%, 75%, max
            F.loc[0, col + '_' + method] = details.loc[method, col]
        
        F.loc[0, col + '_' + 'skew'] = d[col].skew(0)
        F.loc[0, col + '_' + 'kurt'] = d[col].kurt(0)

       
        d_u = details.loc['mean', col]
        d_with_u0 = d[col].sub( d_u ) # make the mean zero        
        
        # zero crossing rate (with window mean removed from signal beforehand)        
        F.loc[0, col + '_' + 'zc'] = d_with_u0.apply(np.signbit,0).diff(1).sum(0)

        F.loc[0, col + '_' + 'ste'] = d[col].pow(2,fill_value=0).mean(0) # short time energy
                     
        try:
            freq = pd.DataFrame(abs(np.fft.rfft(d_with_u0)))[1:].sort_values(by=0)                     
            F.loc[0, col + '_' + 'freq1'] = freq.iloc[-1].name # dominant frequency
            F.loc[0, col + '_' + 'Pfreq1'] = freq.iloc[-1][0]**2  # dominant frequency power
            F.loc[0, col + '_' + 'freq2'] = freq.iloc[-2].name # 2nd dominant frequency
            F.loc[0, col + '_' + 'Pfreq2'] = freq.iloc[-2][0]**2 # 2nd dominant frequency power
        except:
            F.loc[0, col + '_' + 'freq1'] = None
            F.loc[0, col + '_' + 'Pfreq1'] = None
            F.loc[0, col + '_' + 'freq2'] = None
            F.loc[0, col + '_' + 'Pfreq2'] = None
            
    
    return F


def generate_imu_features(imu_file, output_dir, window_method, interval_start=None, interval_end=None, save_as_csv=False):
    print("Start processing " + imu_file)
    source_name = os.path.basename(imu_file).split(".")[0]
    exp_root = os.path.dirname(os.path.dirname(imu_file))

    # Access the raw signal:
    raw_signal = get_imu_data(exp_root, source_name, interval_start, interval_end, "video")

    raw_df = pd.DataFrame(raw_signal, columns = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'roll', 'pitch', 'yaw', 'qx', 'qy', 'qz', 'qw']) 
    raw_df.set_index('timestamp',inplace=True)
        
    
    # additional data vectors
    raw_df['a'] = raw_df.loc[:,['ax','ay','az']].pow(2).sum(1).pow(0.5) # take the root sum of squares of acceleration
    raw_df['g'] = raw_df.loc[:,['gx','gy','gz']].pow(2).sum(1).pow(0.5) # take the root sum of squares of gyro
    raw_df['m'] = raw_df.loc[:,['mx','my','mz']].pow(2).sum(1).pow(0.5) # take the root sum of squares of magnetic field
    raw_df['hpy'] = raw_df.loc[:,['yaw','pitch','roll']].pow(2).sum(1).pow(0.5) # take the root sum of squares of head pitch & yaw

    # If no interval defined, used start and end of the signal:
    if interval_start is None and len(raw_signal) > 0:
        interval_start = raw_df.index[0]   #  raw_signal[0, 0]

    if interval_end is None and len(raw_signal) > 0:
        interval_end = raw_df.index[-1] # raw_signal[-1, 0]

    # Calculate window sizes:
    windows = get_windows(interval_start, interval_end, window_method, source=imu_file)
    windows = pd.DataFrame( windows, columns = ['t_start','t_mid','t_end'] )
    windows['label'] = -100


    features = pd.DataFrame()

    for wnd in windows.itertuples():
        
        current_values = raw_df[ (raw_df.index >= wnd.t_start) & (raw_df.index <= wnd.t_end) ]          
        #get_values_in_window(raw_signal, window[0], window[2])
        current_features = compute_features(current_values)
        features = features.append( current_features, ignore_index = True)

    # combine with windows data
    features = windows.join(features)

    output_file = os.path.join(output_dir, source_name + "_" + window_method + "_empty-labels")
    #np.save(output_file + ".npy", features)
    pd.to_pickle( features, output_file+".pickle" ) 
    if save_as_csv:
        #with open(output_file + ".csv", 'wb') as f:
            #np.savetxt(f, features)
        features.to_csv( output_file + ".csv", index_label=False )

    print("Finished processing " + output_file)


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
