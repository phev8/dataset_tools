import argparse
from datetime import datetime
from experiment_handler.finder import get_eyetracker_participant_list, get_eyetracker_recording_list_in_folder
from experiment_handler.pupil_data_reader import read_eyetracker_info_file

import os
import pandas as pd
import numpy as np


def find_eye_timestamp_file(et_recording_path):
    file = os.path.join(et_recording_path, "eye0_timestamps.npy")
    if os.path.exists(file):
        return file
    return None


def find_fixation_file(et_recording_path):
    # Returns the first fixation file:
    export_dir = os.path.join(et_recording_path, "exports")
    if os.path.exists(export_dir):
        export_list = [os.path.join(export_dir, dir) for dir in os.listdir(export_dir) if
                            os.path.isdir(os.path.join(export_dir, dir))]
        for e in export_list:
            fixation_file = os.path.join(e, "fixations.csv")
            if os.path.exists(fixation_file):
                return fixation_file
    return None


class MergeFixationFiles:
    def __init__(self, recordings_path):
        self.path = recordings_path
        self.recordings = get_eyetracker_recording_list_in_folder(recordings_path)
        self.merged_fixations = None

    def merge_fixations(self):
        for r in self.recordings:
            current_fixations = self._read_and_convert_fixations_for_single_recording(r)

            if self.merged_fixations is None:
                self.merged_fixations = current_fixations
            else:
                self.merged_fixations = self.merged_fixations.append(current_fixations)
        print("Number of total fixations: ", len(self.merged_fixations))
        return self.merged_fixations

    def save_merged_fixations(self, output_dir):
        if self.merged_fixations is None:
            self.merge_fixations()

        filename = os.path.join(output_dir, os.path.basename(self.path) + "_fixations.csv")
        self.merged_fixations.to_csv(filename)

    def _read_and_convert_fixations_for_single_recording(self, current_recording):
        fixation_file = find_fixation_file(current_recording)

        if fixation_file is None:
            print("Error: fixations file found in ", current_recording)
            return None
        fixations_raw = pd.DataFrame.from_csv(fixation_file)
        eye_timestamps = self._read_and_convert_eye_timestamps(current_recording)
        if eye_timestamps is None:
            print("Error: no eye timestamps found in ", current_recording)
            return None
        fixations = self._convert_raw_fixations(fixations_raw, eye_timestamps)
        print(datetime.fromtimestamp(fixations.start_timestamp.iloc[0]), datetime.fromtimestamp(fixations.start_timestamp.iloc[-1]))
        return fixations

    def _read_and_convert_eye_timestamps(self, current_recording):
        eye_timestamp_file = find_eye_timestamp_file(current_recording)

        basetime = read_eyetracker_info_file(current_recording)["basetime"]
        eye_timestamps = np.load(eye_timestamp_file)

        #eye_timestamps = eye_timestamps - eye_timestamps[0]  # norm timestamps
        #eye_timestamps += basetime.timestamp()  # norm timestamps
        offset = basetime.timestamp() - eye_timestamps[0]
        return offset

    def _convert_raw_fixations(self, fixations, offset):
        for i in range(len(fixations)):
            fixations.set_value(i, "start_timestamp",fixations.start_timestamp.iloc[i] + offset)
        return fixations


if __name__ == '__main__':
    start = datetime.now()

    parser = argparse.ArgumentParser(
        description='Merge fixation files for all participant of an experiment')
    parser.add_argument(
        '-p', '--path_to_experiment',
        metavar='PATH_TO_EXP',
        type=str,
        help='Path to experiment folder',
        default="/Volumes/DataDrive/igroups_recordings/igroups_experiment_9"
    )
    args = parser.parse_args()

    experiment_root = args.path_to_experiment

    participants = get_eyetracker_participant_list(experiment_root)

    output_dir = os.path.join(experiment_root, "fixations")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for p in participants:
        mff = MergeFixationFiles(p)
        mff.save_merged_fixations(output_dir)

    print("Runtime: ", (datetime.now() - start).seconds, ' seconds')
