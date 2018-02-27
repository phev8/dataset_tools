import os, sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
if __package__ is None:
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )


from experiment_handler.object_recognition.object_detection_reader import read_object_detections_for_person
from experiment_handler.label_data_reader import read_experiment_phases
from experiment_handler.finder import get_object_recognition_infos
from feature_calculations.window_generator import get_windows


def compute_features(values):
    # calculate features
    predictions = values["predictions"].mean(axis=0)
    F = pd.DataFrame([predictions], index=[0])
    return F


def generate_obj_rec_features(filename, output_dir, window_method, interval_start=None, interval_end=None, save_as_csv=False):
    print("Start processing for " + filename)
    current_person_id = os.path.basename(filename).split('_')[0]
    current_model = os.path.basename(filename).split('_')[1]
    exp_root = os.path.dirname(os.path.dirname(os.path.dirname(filename)))

    obj_recs = read_object_detections_for_person(exp_root, current_model, current_person_id, interval_start, interval_end, "video")

    # Calculate window sizes:
    windows = get_windows(interval_start, interval_end, window_method)

    # Calculate window sizes:
    windows = get_windows(interval_start, interval_end, window_method)
    windows = pd.DataFrame(windows, columns=['t_start', 't_mid', 't_end'])
    windows['label'] = 'no_label'

    features = pd.DataFrame()

    for wnd in windows.itertuples():
        current_values = obj_recs[(obj_recs.timestamp >= wnd.t_start) & (obj_recs.timestamp <= wnd.t_end)]
        current_features = compute_features(current_values)
        features = features.append(current_features, ignore_index=True)

    # combine with windows data
    features = windows.join(features)



    output_file = os.path.join(output_dir, current_person_id + "_" + current_model + "_" + window_method + "_empty-labels")

    pd.to_pickle(features, output_file + ".pickle")
    if save_as_csv:
        features.to_csv(output_file + ".csv", index_label=False)

    print("Finished processing " + filename)


if __name__ == '__main__':
    start = datetime.now()

    parser = argparse.ArgumentParser(
        description='Run object recognition feature extraction.')
    parser.add_argument(
        '-p', '--path_to_experiment',
        metavar='PATH_TO_EXP',
        type=str,
        help='Path to experiment folder',
        default="/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    )
    parser.add_argument(
        '-w', '--window_method',
        metavar='WINDOW',
        type=str,
        help='Name of the window method, e.g. SW-5000-1000',
        default="SW-5000-1000"
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
    output_dir = os.path.join(experiment_root, "processed_data", "object_recognition_results", "feature_files")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    object_rec_files, persons, model_names, training_sets = get_object_recognition_infos(experiment_root)

    if start is None and end is None:
        # read experiment phases to generate features in that interval
        experiment_phases = read_experiment_phases(experiment_root)
        start = experiment_phases['assembly'][0]
        end = experiment_phases['disassembly'][1]

    # run feature computation parallel for multiple participants
    arguments = []
    for obj_rec_file in object_rec_files:
        arguments.append((obj_rec_file, output_dir, window_method, start, end, True))
        #generate_obj_rec_features(obj_rec_file, output_dir, window_method, interval_start=start, interval_end=end, save_as_csv=True)
        #exit()

    p = Pool(4)
    p.starmap(generate_obj_rec_features, arguments)
