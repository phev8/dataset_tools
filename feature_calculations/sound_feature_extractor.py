import os, sys
import argparse
import numpy as np
import pandas as pd

from feature_calculations.python_speech_features import mfcc 

from datetime import datetime
from multiprocessing import Pool
if __package__ is None:
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from experiment_handler.sound_reader import get_sound_data
from experiment_handler.label_data_reader import read_experiment_phases
from experiment_handler.finder import find_all_sound_files
from feature_calculations.window_generator import get_windows
from feature_calculations.common import get_values_in_window, mean_crossings

   

def compute_features( d ):
    """
    Compute sound based features for the given audio data 

    Parameters
    ----------
    audio: numpy array (n x 3)
        audio data with <timestamp> and columns <h>, <w>


    Returns
    -------
        feature values (1 x ? )
            Extracted features with in the following order: 
    """
   
    # calculate features    
    F = pd.DataFrame()    

    for col in d.columns:

        d_u = d[col].mean( 0 )
        d_with_u0 = d[col].sub( d_u ) # make the mean zero        
        
        F.loc[0, col + '_' + 'mean'] = d_u        
        F.loc[0, col + '_' + 'std'] = d[col].std(0)    
        F.loc[0, col + '_' + 'max'] = d[col].max(0)
    
        # zero crossing rate (with window mean removed from signal beforehand)        
        F.loc[0, col + '_' + 'zc'] = d_with_u0.apply(np.signbit,0).diff(1).sum(0)
        F.loc[0, col + '_' + 'ste'] = d[col].pow(2,fill_value=0).mean(0) # short time energy
        
      #  F.loc[0,col + '_' + 'env'] =  d_with_u0.abs().rolling(env_wnd).mean()   # envelope detection             
        
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



def generate_sound_features_with_mfcc(participant_path, output_dir, window_method, interval_start=None, interval_end=None, save_as_csv=False):
    """
    TODO: finish the function
        Calculates audio features using the MFCC method borrowed from James Lyon's python_speech_features package (https://github.com/jameslyons/python_speech_features)
        
    """
##
    feat = mfcc(raw_sound_data.values, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=512, lowfreq=0, highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True )

##
    



def generate_sound_features(participant_path, output_dir, window_method, interval_start=None, interval_end=None, save_as_csv=False):
    print("Start processing for " + participant_path)
    exp_root = os.path.dirname(os.path.dirname(participant_path))
    participant = os.path.basename(participant_path).split(".")[0]

    raw_sound_data = get_sound_data(exp_root, participant, interval_start, interval_end, "video")
    
    # If no interval defined, used start and end of the signal:
    if interval_start is None and len(raw_sound_data) > 0:
        interval_start = raw_sound_data.index[0] 

    if interval_end is None and len(raw_sound_data) > 0:
        interval_end = raw_sound_data.index[-1]

    # Calculate window sizes:
    windows = get_windows(interval_start, interval_end, window_method )
    windows = pd.DataFrame( windows, columns = ['t_start','t_mid','t_end'] )
    windows['label'] = 'no_label'

    features = pd.DataFrame()
        
    for wnd in windows.itertuples():        
        current_values = raw_sound_data[ (raw_sound_data.index >= wnd.t_start) & (raw_sound_data.index <= wnd.t_end) ]          
        current_features = compute_features(current_values)
        features = features.append( current_features, ignore_index = True)

    # combine with windows data
    features = windows.join(features)

    output_file = os.path.join(output_dir, participant + "_sound_features_" + window_method + "_empty-labels")

    pd.to_pickle(features, output_file + ".pickle")
    if save_as_csv:
        features.to_csv(output_file + ".csv", index_label=False)

    print("Finished processing " + participant_path)






if __name__ == '__main__':
    start = datetime.now()

    parser = argparse.ArgumentParser(
        description='Run sound feature extraction.')
    parser.add_argument(
        '-p', '--path_to_experiment',
        metavar='PATH_TO_EXP',
        type=str,
        help='Path to experiment folder',
        default="/data/igroups_recordings/experiment_8"
    )
    parser.add_argument(
        '-w', '--window_method',
        metavar='WINDOW',
        type=str,
        help='Name of the window method, e.g. SW-40-40 (40ms, 40ms)',
        default="SW-40-40"
    )
    parser.add_argument(
        '-s', '--interval_start',
        metavar='START',
        type=int,
        help='Start of the evaluation interval in video time reference.',
        default=None
    )
    parser.add_argument(
        '-e', '--interval_end',
        metavar='End',
        type=int,
        help='End of the evaluation interval in video time reference.',
        default=None
    )
    args = parser.parse_args()

    experiment_root = args.path_to_experiment
    window_method = args.window_method
    start = args.interval_start
    end = args.interval_end

    # Generate output dir:
    output_dir = os.path.join(experiment_root, "processed_data", "sound_features")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # find wav files: (ensure directory has only the participant wav files)
    participants = find_all_sound_files(experiment_root)

    if start is None and end is None:
        # read experiment phases to generate features in that interval
        experiment_phases = read_experiment_phases(experiment_root)
        start = experiment_phases['assembly'][0]
        end = experiment_phases['disassembly'][1]

    # run feature computation parallel for multiple participants
    arguments = []
    for participant in participants:
        arguments.append((participant, output_dir, window_method, start, end, False))
        #generate_sound_features(participant, output_dir, window_method, interval_start=start, interval_end=end, save_as_csv=False)
        #exit()

    p = Pool(4)
    p.starmap(generate_sound_features, arguments)
