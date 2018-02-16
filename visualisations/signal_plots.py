import matplotlib.pyplot as plt

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
from experiment_handler.sound_reader import get_sound_data
from experiment_handler.pupil_data_reader import get_eye_data, get_fixation_events
from feature_calculations.eye_feature_extractor import get_fixation_length_and_gap_curves
from experiment_handler.label_data_reader import read_experiment_phases, read_localtion_labels
from experiment_handler.finder import find_all_imu_files
from feature_calculations.window_generator import get_windows
from feature_calculations.common import get_values_in_window



def plot_sound(exp_root, person_id, start, end, reference_time, channel=None):
    raw_signal = get_sound_data(exp_root, person_id, start, end, reference_time)

    plt.figure(figsize=(12, 3))
    plt.title(person_id + " microphones")
    if channel is not None and channel == "w":
        plt.plot(raw_signal.index, raw_signal.w)
    elif channel is not None and channel == "h":
        plt.plot(raw_signal.index, raw_signal.h)
    else:
        plt.plot(raw_signal.index, raw_signal.w, label="wrist")
        plt.plot(raw_signal.index, raw_signal.h, label="head")

    plt.grid()
    plt.tight_layout()
    plt.draw()


    import pylab
    plt.figure(figsize=(12, 3))
    window_size = 256
    pylab.plot()
    results = pylab.specgram(raw_signal.h, NFFT=256, Fs=16000)
    pylab.show()


def plot_fixation_length(person_id, start, end, reference_time ):
    fixation_events = get_fixation_events(exp_root, person_id, start, end, reference_time)

    fixation_curves = get_fixation_length_and_gap_curves(fixation_events, start, end)
    plt.figure(figsize=(12, 3))
    plt.title(person_id + " fixation length")
    plt.plot(fixation_curves[:, 0], fixation_curves[:, 1])
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_location_ground_truth(exp_path):
    gt_labels = read_localtion_labels(exp_path)
    exp_phase = read_experiment_phases(exp_path)

    width = 0.03
    location_lookup = {"L1": 0.17, "L2": 0.34, "L3": 0.5, "L4": 0.67, "L5": 0.84}
    person_offset = {"P1": -2*width, "P2": -width, "P3": 0, "P4": width}
    person_color = {"P1": "blue", "P2": "grey", "P3": "green", "P4": "orange"}


    plt.figure(figsize=(8, 4))
    for p in gt_labels.keys():
        locs = gt_labels[p]

        for loc in locs:
            if loc['start'] > exp_phase['assembly'][0] and loc['end'] < exp_phase['disassembly'][1]:
                plt.axvspan(loc['start'], loc['end'], location_lookup[loc['location']] + person_offset[p],  location_lookup[loc['location']] + person_offset[p] + width/1.1, color=person_color[p])


    plt.yticks(list(location_lookup.values()), list(location_lookup.keys()))
    plt.grid()
    plt.show()



if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"

    plot_location_ground_truth(exp_root)
    exit()


    start = 31*60 + 30
    end = start + 60
    person_id = "P2"
    imu_file = person_id + "_imu_right"
    raw_signal = get_imu_data(exp_root, imu_file, start, end, "video")

    #plot_fixation_length(person_id, start, end, "video")
    #exit()
    plot_sound(exp_root, "P4", start, end, "P4_gopro", "w")
    #plot_sound(exp_root, "P2", start, end, "P3_gopro", "h")
    #plot_sound(exp_root, "P3", start, end, "P3_gopro", "h")
    #plot_sound(exp_root, "P4", start, end, "P3_gopro", "h")
    #exit()
    print(raw_signal)
    plt.figure(figsize=(12,3))
    plt.title(person_id + " right arm gyroscope")
    plt.plot(raw_signal[:, 0], raw_signal[:, 4])
    plt.plot(raw_signal[:, 0], raw_signal[:, 5])
    plt.plot(raw_signal[:, 0], raw_signal[:, 6])
    plt.grid()
    plt.tight_layout()
    plt.show()
