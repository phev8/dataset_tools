# -*- coding: utf-8 -*-
"""
Spyder Editor

Jamie Ward testing script


"""

import os, sys
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool

from feature_calculations.window_generator import get_windows
from feature_calculations.imu_feature_extractor import generate_imu_features
from feature_calculations.common import get_values_in_window

from experiment_handler.imu_data_reader import get_imu_data
from experiment_handler.label_data_reader import read_experiment_phases
from experiment_handler.finder import find_all_imu_files



if 1:
    start = datetime.now()

    parser = argparse.ArgumentParser(
        description='Run imu feature extraction.')
    parser.add_argument(
        '-p', '--path_to_experiment',
        metavar='PATH_TO_EXP',
        type=str,
        help='Path to experiment folder',
        default="/Data/igroups_recordings/igroups_experiment_8"
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

    p = Pool(1)
    p.starmap(generate_imu_features, arguments)
    
    
    
    