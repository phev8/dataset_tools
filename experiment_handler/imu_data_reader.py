import os
import json
import numpy as np
import pandas as pd
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
    # TODO: get bluetooth beacon data

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

    Returns
    -------
        parsed_data: numpy array
            IMU data with columns order:  <timestamp>, <ax>, <ay>, <az>, <gx>, <gy>, <gz>, <mx>, <my>, <mz>, <roll>, <pitch>, <yaw>, <qx>, <qy>, <gz>, <qw>
    """
    parsed_data = np.zeros((0, 17))

    filepath = os.path.join(experiment_path, "imu", source + ".log")
    imu_lines = load_imu_file(filepath)


    # Convert start and end time:
    imu_reference_time = source.split("_")[0] + "_IMU"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, imu_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, imu_reference_time)
            
    def rss(d):
        return np.sqrt( np.square(d).sum(axis=1) ) 


    # Parse lines:
    for imu_line in imu_lines:
        try:
            data = json.loads(imu_line)
        except json.decoder.JSONDecodeError:
            continue

        if data['type'] != 'imu-bosch':
            continue

        if start is not None and start_timestamp > data['time']:
            continue
        if end is not None and end_timestamp < data['time']:
            break

        current_data = np.zeros((1, 17))

        current_data[0, 0] = data['time']
        current_data[0, 1] = data['measurement']['ax']
        current_data[0, 2] = data['measurement']['ay']
        current_data[0, 3] = data['measurement']['az']

        current_data[0, 4] = data['measurement']['gx']
        current_data[0, 5] = data['measurement']['gy']
        current_data[0, 6] = data['measurement']['gz']

        current_data[0, 7] = data['measurement']['mx']
        current_data[0, 8] = data['measurement']['my']
        current_data[0, 9] = data['measurement']['mz']

        current_data[0, 10] = data['measurement']['roll']
        current_data[0, 11] = data['measurement']['pitch']
        current_data[0, 12] = data['measurement']['yaw']

        current_data[0, 13] = data['measurement']['qx']
        current_data[0, 14] = data['measurement']['qy']
        current_data[0, 15] = data['measurement']['qz']
        current_data[0, 16] = data['measurement']['qw']
 
        # RSS combination of accelerometer & gyro signals       
        #current_data[0, 17] = np.sqrt( np.square( data['measurement']['ax','ay','az'] ).sum() )
        #current_data[0, 18] = np.sqrt( np.square( data['measurement']['gx','gy','gz'] ).sum() )
        

        parsed_data = np.append(parsed_data, current_data, axis=0)

    if convert_time:
        parsed_data[:, 0] = convert_timestamps(experiment_path, parsed_data[:, 0], imu_reference_time, reference_time)

    return parsed_data


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"

    data = get_ble_data(exp_root, "P3_imu_left", start=652, end=668, reference_time="video", convert_time=True)
    data = get_imu_data(exp_root, "P3_imu_left", start=652, end=668, reference_time="video", convert_time=True)
    print(data)
