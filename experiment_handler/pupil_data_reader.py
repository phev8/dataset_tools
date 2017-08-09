import os, sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from common_methods.time_format_conversions import convert_to_timestamp_p2
from experiment_handler.finder import get_eyetracker_recording_list_in_folder
from experiment_handler.time_synchronisation import convert_timestamps


def get_fixation_events(experiment_path, participant, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read fixation event data for a given participant (e.g. P3) in a time interval

    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    participant: str
        Name of the participant e.g. P3
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)
    convert_time: bool
        If set the returned array will contain timestamp in reference_time's values

    Returns
    -------
        parsed_data: numpy array
            fixation events with columns order:  <start timestamp>, <end timestamp>, <duration>, <norm_x>, <norm_y>, <dispersion>, <avg pupil size>, <confidence>

    """
    parsed_data = np.zeros((0, 8))

    fixation_event_file = os.path.join(experiment_path, "fixations", participant + "_fixations.csv")

    data = pd.read_csv(fixation_event_file)

    et_reference_time = participant + "_eyetracker"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, et_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, et_reference_time)

    for i, current_row in data.iterrows():
        if start is not None and current_row.start_timestamp + current_row.duration < start_timestamp:
            continue
        if end is not None and current_row.start_timestamp > end_timestamp:
            break

        current_data = np.zeros((1, 8))
        current_data[0, 0] = current_row.start_timestamp
        current_data[0, 1] = current_row.start_timestamp + current_row.duration
        current_data[0, 2] = current_row.duration
        current_data[0, 3] = current_row.norm_pos_x
        current_data[0, 4] = current_row.norm_pos_y
        current_data[0, 5] = current_row.dispersion
        current_data[0, 6] = current_row.avg_pupil_size
        current_data[0, 7] = current_row.confidence

        parsed_data = np.append(parsed_data, current_data, axis=0)

    if convert_time:
        parsed_data[:, 0] = convert_timestamps(experiment_path, parsed_data[:, 0], et_reference_time, reference_time)
        parsed_data[:, 1] = convert_timestamps(experiment_path, parsed_data[:, 1], et_reference_time, reference_time)

    return parsed_data


def get_eye_data(experiment_path, participant, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read eye data for a given participant (e.g. P3) in a time interval

    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    participant: str
        Name of the participant e.g. P3
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)
    convert_time: bool
        If set the returned array will contain timestamp in reference_time's values

    Returns
    -------
        parsed_data: numpy array
            eyetracker data with columns order:  <timestamp>, <theta>, <phi>, <diameter>, <confidence>
    """
    parsed_data = np.zeros((0, 5))

    eyetracker_rec_folder = os.path.join(experiment_path, "eyetracker", participant)

    eyetracker_recordings = get_eyetracker_recording_list_in_folder(eyetracker_rec_folder)

    et_reference_time = participant + "_eyetracker"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, et_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, et_reference_time)

    for rec_path in eyetracker_recordings:
        current_data = _read_eye_data(rec_path)
        if start is not None and current_data[-1, 0] < start_timestamp:
            continue
        if end is not None and current_data[0, 0] > end_timestamp:
            break
        parsed_data = np.append(parsed_data, current_data, axis=0)

    # filter numpy array for interval:
    if start is not None:
        parsed_data = parsed_data[parsed_data[:, 0] >= start_timestamp, :]
    if end is not None:
        parsed_data = parsed_data[parsed_data[:, 0] <= end_timestamp, :]

    if convert_time:
        parsed_data[:, 0] = convert_timestamps(experiment_path, parsed_data[:, 0], et_reference_time, reference_time)

    return parsed_data


def get_gaze_data(experiment_path, participant, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read gaze data for a given participant (e.g. P3) in a time interval

    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    participant: str
        Name of the participant e.g. P3
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)
    convert_time: bool
        If set the returned array will contain timestamp in reference_time's values

    Returns
    -------
        parsed_data: numpy array
            eyetracker data with columns order:  <timestamp>, <norm pos_x>, <norm pos_y>, <confidence>
    """
    parsed_data = np.zeros((0, 4))

    eyetracker_rec_folder = os.path.join(experiment_path, "eyetracker", participant)

    eyetracker_recordings = get_eyetracker_recording_list_in_folder(eyetracker_rec_folder)

    et_reference_time = participant + "_eyetracker"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, et_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, et_reference_time)

    for rec_path in eyetracker_recordings:
        current_data = _read_gaze_data(rec_path)
        if start is not None and current_data[-1, 0] < start_timestamp:
            continue
        if end is not None and current_data[0, 0] > end_timestamp:
            break
        parsed_data = np.append(parsed_data, current_data, axis=0)

    # filter numpy array for interval:
    if start is not None:
        parsed_data = parsed_data[parsed_data[:, 0] >= start_timestamp, :]
    if end is not None:
        parsed_data = parsed_data[parsed_data[:, 0] <= end_timestamp, :]

    if convert_time:
        parsed_data[:, 0] = convert_timestamps(experiment_path, parsed_data[:, 0], et_reference_time, reference_time)

    return parsed_data


def get_basetime_from_info_file_text(info_file_text_lines):
    date = None
    time = None
    for line in info_file_text_lines:
        items = line.strip("\r\n").split(",")
        if items[0] == "Start Date":
            date = items[1]
        elif items[0] == "Start Time":
            time = items[1]

    basetime = datetime.strptime(date + "-" + time, "%d.%m.%Y-%H:%M:%S")
    return basetime


def get_duration_from_info_file_text(info_file_text_lines):
    delta = None
    for line in info_file_text_lines:
        items = line.strip("\r\n").split(",")
        if items[0] == "Duration Time":
            delta = items[1]
    try:
        basetime = datetime.strptime("11.11.1977-" + delta, "%d.%m.%Y-%H:%M:%S")
    except TypeError:
        print(info_file_text_lines)
    return timedelta(hours=basetime.hour, minutes=basetime.minute, seconds=basetime.second)


def read_eyetracker_info_file(eyetracker_recording_path):
    """
    Find and parse info file as a dictionary
    """
    filename = os.path.join(eyetracker_recording_path, "info.csv")
    if not os.path.exists(filename):
        return None

    content = open(filename, "r")
    lines = content.readlines()

    infos = {
        "basetime": get_basetime_from_info_file_text(lines),
        "duration": get_duration_from_info_file_text(lines)
    }
    # TODO: get other infos from the file
    return infos


def read_world_timestamps(recording_path):
    """
    Read and return timestamps (posix) of the world camera video frames
    """
    world_timestamps = np.load(os.path.join(recording_path, "world_timestamps.npy"))
    world_timestamps = world_timestamps - world_timestamps[0]  # norm timestamps
    infos = read_eyetracker_info_file(recording_path)

    if sys.version_info > (3, 0):
        # Python 3 code in this block
        world_timestamps += infos["basetime"].timestamp()
    else:
        # Python 2 code in this block
        world_timestamps += convert_to_timestamp_p2(infos["basetime"])

    return world_timestamps


def _read_gaze_data(recording_path):
    """
    Read and parse pupil eyetracking data
    """
    with open(os.path.join(recording_path, "pupil_data"), 'rb') as handle:
        et_data = handle.read()
        if sys.version_info > (3, 0):
            pupil_data = pickle.loads(et_data, encoding='ISO-8859-1')
        else:
            pupil_data = pickle.loads(et_data)

        current_pupil_data = []
        for p_d in pupil_data["gaze_positions"]:
            current_pupil_data.append([p_d["timestamp"], p_d["norm_pos"][0], p_d["norm_pos"][1], p_d["confidence"]])

        current_pupil_data = np.array(current_pupil_data)
        current_pupil_data[:, 0] = current_pupil_data[:, 0] - current_pupil_data[0, 0]  # norm timestamps

        infos = read_eyetracker_info_file(recording_path)

        if sys.version_info > (3, 0):
            # Python 3 code in this block
            current_pupil_data[:, 0] += infos["basetime"].timestamp()
        else:
            # Python 2 code in this block
            current_pupil_data[:, 0] += convert_to_timestamp_p2(infos["basetime"])

        return current_pupil_data

    return None


def _read_eye_data(recording_path, reload=True, save_binary=True):
    """
    Read and parse pupil eyetracking data
    """
    binary_name = os.path.join(recording_path, "eye_data.npy")
    if reload and os.path.exists(binary_name):
        return np.load(binary_name)
    else:
        with open(os.path.join(recording_path, "pupil_data"), 'rb') as handle:
            et_data = handle.read()
            pupil_data = pickle.loads(et_data, encoding='ISO-8859-1')

            current_pupil_data = []
            for p in pupil_data["gaze_positions"]:
                current_pupil_data.append([p["base_data"][0]["timestamp"],
                                           p["base_data"][0]["theta"], p["base_data"][0]["phi"],
                                           p["base_data"][0]["diameter_3d"], p["confidence"]])

            current_pupil_data = np.array(current_pupil_data)
            current_pupil_data[:, 0] = current_pupil_data[:, 0] - current_pupil_data[0, 0]  # norm timestamps

            infos = read_eyetracker_info_file(recording_path)

            if sys.version_info > (3, 0):
                # Python 3 code in this block
                current_pupil_data[:, 0] += infos["basetime"].timestamp()
            else:
                # Python 2 code in this block
                current_pupil_data[:, 0] += convert_to_timestamp_p2(infos["basetime"])

            if save_binary:
                np.save(binary_name, current_pupil_data)
            return current_pupil_data
    return None


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    data = get_fixation_events(exp_root, "P3", start=653, end=1350, reference_time="video", convert_time=True)
    # data = get_ble_data(exp_root, "P3_imu_left", start=652, end=668, reference_time="video", convert_time=True)
    # data = get_eye_data(exp_root, "P3", start=653, end=1350, reference_time="video", convert_time=True)
    print(data)
    print(data.shape)
