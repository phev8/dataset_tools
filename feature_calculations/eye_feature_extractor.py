import os, sys
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from experiment_handler.pupil_data_reader import get_fixation_events, get_eye_data
from experiment_handler.finder import get_eyetracker_participant_list
from feature_calculations.window_generator import get_windows
from feature_calculations.common import get_values_in_window


def get_fixation_length_and_gap_curves(fixation_events, start=None, end=None, step_size=1.0):
    """
    Generating time series curves out of fixation events. Two features (fixation length and gap between two fixations is calculated)

    Parameters
    ----------
    fixation_events: numpy array

    start: float or None
        start of the curve, if None the first event's starting timestamp is used
    end: float or None
        end of the curve, if None the last event's ending timestamp is used
    step_size: float
        generate a samples with this distance

    Returns
    -------
        fixation_curves: numpy array
            fixation curves with columns order: <timestamp>, <fixation_length>, <fixation_gap>

    """
    fixation_curves = np.zeros((0, 3))

    if start is None:
        start = fixation_events[0, 0]
    if end is None:
        end = fixation_events[-1, 1]

    current_time = start
    while current_time <= end:
        current_row = np.zeros((1, 3))
        current_row[0, 0] = current_time

        # get accumulated fixation length for fixations active around current sample
        window_start = current_time - step_size/2
        window_end = current_time + step_size/2
        fixations_in_window = fixation_events[fixation_events[:, 1] >= window_start, :]
        fixations_in_window = fixations_in_window[fixations_in_window[:, 0] <= window_end, :]
        current_row[0, 1] = np.sum(fixations_in_window[:, 2])

        # get gap between the next fixations at the current sample
        index_of_closest = np.argmin(np.abs(fixation_events[:, 0] - current_time))
        index_of_previous = index_of_closest - 1
        index_of_next = index_of_closest + 1

        if index_of_previous < 0:
            current_row[0, 2] = -1
        elif index_of_next >= fixation_events.shape[0]:
            current_row[0, 2] = -1
        else:
            if fixation_events[index_of_closest, 0] <= current_time <= fixation_events[index_of_closest, 1] or \
                                    fixation_events[index_of_previous, 0] <= current_time <= fixation_events[index_of_previous, 1] or \
                                    fixation_events[index_of_next, 0] <= current_time <= fixation_events[index_of_next, 1]:
                current_row[0, 2] = 0
            else:
                if fixation_events[index_of_closest, 0] > current_time:
                    current_row[0, 2] = fixation_events[index_of_closest, 0] - fixation_events[index_of_previous, 1]
                else:
                    current_row[0, 2] = fixation_events[index_of_next, 0] - fixation_events[index_of_closest, 1]

        fixation_curves = np.append(fixation_curves, current_row, axis=0)
        current_time += step_size

    return fixation_curves


def compute_features(gaze_values, fixation_curves):
    """
    Compute eye based features for the given arrays (already windowed)

    Parameters
    ----------
    gaze_values: numpy array (n x 5)
        eye gaze data with columns order:  <timestamp>, <theta>, <phi>, <diameter>, <confidence>
    fixation_curves: numpy array (n x ?)
        fixation curves with columns order: <timestamp>, <fixation_length>, <fixation_gap>


    Returns
    -------
        feature values (1 x ? )
            Extracted features with in the following order: ?? TODO
    """
    features = np.zeros((1, 10))

    # TODO: compute eye features

    return features


def generate_eye_features(participant_path, output_dir, window_method, interval_start=None, interval_end=None, save_as_csv=False):
    print("Start processing for " + participant_path)
    exp_root = os.path.dirname(os.path.dirname(participant_path))
    participant = os.path.basename(participant_path)

    raw_eye_data = get_eye_data(exp_root, participant, interval_start, interval_end, "video")
    fixation_events = get_fixation_events(exp_root, participant, interval_start, interval_end, "video")

    fixation_curves = get_fixation_length_and_gap_curves(fixation_events, interval_start, interval_end)
    import matplotlib.pyplot as plt
    plt.plot(fixation_curves[:, 0], fixation_curves[:, 1], label="len")
    plt.plot(fixation_curves[:, 0], fixation_curves[:, 2], label="gap")
    plt.legend()
    plt.show()

    # If no interval defined, used start and end of the signal:
    if interval_start is None and len(raw_eye_data) > 0:
        interval_start = raw_eye_data[0, 0]

    if interval_end is None and len(raw_eye_data) > 0:
        interval_end = raw_eye_data[-1, 0]

    # Calculate window sizes:
    windows = get_windows(interval_start, interval_end, window_method)

    features = []

    for window in windows:
        current_gaze_values = get_values_in_window(raw_eye_data, window[0], window[2])
        current_fixation_curve_values = get_values_in_window(fixation_curves, window[0], window[2])
        current_features = compute_features(current_gaze_values, current_fixation_curve_values)

        current_row = np.zeros((1, 4 + current_features.shape[1]))
        current_row[0, 0] = window[0]
        current_row[0, 1] = window[1]
        current_row[0, 2] = window[2]
        current_row[0, 3] = -100 # dummy label
        current_row[0, 4:] = current_features

        features.append(current_row)

    features = np.array(features).reshape(len(features), -1)

    output_file = os.path.join(output_dir, participant + "_eye_features_" + window_method + "_empty-labels")
    np.save(output_file + ".npy", features)
    if save_as_csv:
        with open(output_file + ".csv", 'wb') as f:
            np.savetxt(f, features)

    print("Finished processing " + participant_path)


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

    # Generate output dir:
    output_dir = os.path.join(experiment_root, "processed_data", "eye_features")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # find participants:
    eyetracker_participants = get_eyetracker_participant_list(experiment_root)

    # TODO: read experiment phases to generate features in that interval
    start = 50
    end = 650
    window_method = "SW-5000-1000"

    # run feature computation parallel for multiple participants
    arguments = []
    for participant in eyetracker_participants:
        arguments.append((participant, output_dir, window_method, start, end, True))
        generate_eye_features(participant, output_dir, window_method, interval_start=start, interval_end=end, save_as_csv=True)
        exit()

    p = Pool(4)
    p.starmap(generate_eye_features, arguments)
