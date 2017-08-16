import os
import json
import numpy as np
from experiment_handler.time_synchronisation import convert_timestamps


def load_imu_file(filepath):
    lines = []
    with open(filepath, 'r') as file:
        try:
            lines = file.read().split("\n")
        except UnicodeDecodeError as e:
            print(e)

    return lines


def get_ble_data(experiment_path, source, start=None, end=None, reference_time=None, convert_time=True):
    # TODO: add description how the bluetooth beacon data is accessed

    parsed_data = np.zeros((0, 17))

    filepath = os.path.join(experiment_path, "imu", source + ".log")
    imu_lines = load_imu_file(filepath)

    # Convert start and end time:
    imu_reference_time = source.split("_")[0] + "_IMU"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, imu_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, imu_reference_time)

        # Parse lines:
        for imu_line in imu_lines:
            try:
                data = json.loads(imu_line)
            except json.decoder.JSONDecodeError:
                continue

            if data['type'] != 'ble':
                continue

            if start is not None and start_timestamp > data['time']:
                continue
            if end is not None and end_timestamp < data['time']:
                break

            # TODO: parse BLE data
            current_data = np.zeros((1, 17))
            print(data)
            continue

            parsed_data = np.append(parsed_data, current_data, axis=0)

        if convert_time:
            parsed_data[:, 0] = convert_timestamps(experiment_path, parsed_data[:, 0], imu_reference_time,
                                                   reference_time)

        return parsed_data


def get_imu_data(experiment_path, source, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read imu data for a given source (e.g. P3 left hand) in a time interval
    
    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    source: str
        Name of the IMU file without extension e.g. P3_imu_right
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)
    convert_time: bool
        If set the returned array will contain timestamp in reference_time's values
    use_pkl: bool
        If set the pickle serialized binary file will be read instead of the .log text file. (If the file doesn't exists yet, the .log file is loaded and then the .pkl file is created.)

    Returns
    -------
        parsed_data: numpy array
            IMU data with columns order:  <timestamp>, <ax>, <ay>, <az>, <gx>, <gy>, <gz>, <mx>, <my>, <mz>, <roll>, <pitch>, <yaw>, <qx>, <qy>, <gz>, <qw>
    """
    npy_filepath = os.path.join(experiment_path, "imu", source + "_movement-data.npy")

    if not os.path.exists(npy_filepath):
        log_filepath = os.path.join(experiment_path, "imu", source + ".log")
        parsed_data = create_imu_log_file_movement_data(log_filepath, npy_filepath)
    else:
        parsed_data = np.load(npy_filepath)

    # Convert start and end time:
    imu_reference_time = source.split("_")[0] + "_IMU"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, imu_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, imu_reference_time)

    if start is not None:
        parsed_data = parsed_data[parsed_data[:, 0] >= start_timestamp, :]
    if end is not None:
        parsed_data = parsed_data[parsed_data[:, 0] <= end_timestamp, :]

    if convert_time:
        parsed_data[:, 0] = convert_timestamps(experiment_path, parsed_data[:, 0], imu_reference_time, reference_time)

    return parsed_data


def create_imu_log_file_movement_data(log_file_path, npy_file_path):
    imu_lines = load_imu_file(log_file_path)

    parsed_data = np.zeros((len(imu_lines), 17))
    parsed_data.fill(np.nan)

    # Parse lines:
    for i in range(len(imu_lines)):
        imu_line = imu_lines[i]
        try:
            data = json.loads(imu_line)
        except json.decoder.JSONDecodeError:
            continue

        if data['type'] != 'imu-bosch':
            continue

        parsed_data[i, 0] = data['time']
        parsed_data[i, 1] = data['measurement']['ax']
        parsed_data[i, 2] = data['measurement']['ay']
        parsed_data[i, 3] = data['measurement']['az']

        parsed_data[i, 4] = data['measurement']['gx']
        parsed_data[i, 5] = data['measurement']['gy']
        parsed_data[i, 6] = data['measurement']['gz']

        parsed_data[i, 7] = data['measurement']['mx']
        parsed_data[i, 8] = data['measurement']['my']
        parsed_data[i, 9] = data['measurement']['mz']

        parsed_data[i, 10] = data['measurement']['roll']
        parsed_data[i, 11] = data['measurement']['pitch']
        parsed_data[i, 12] = data['measurement']['yaw']

        parsed_data[i, 13] = data['measurement']['qx']
        parsed_data[i, 14] = data['measurement']['qy']
        parsed_data[i, 15] = data['measurement']['qz']
        parsed_data[i, 16] = data['measurement']['qw']

    parsed_data = parsed_data[~np.isnan(parsed_data).any(axis=1)]
    np.save(npy_file_path, parsed_data)
    return parsed_data


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"

    #data = get_ble_data(exp_root, "P3_imu_left", start=652, end=668, reference_time="video", convert_time=True)
    data = get_imu_data(exp_root, "P3_imu_left", start=652, end=668, reference_time="video", convert_time=True)
