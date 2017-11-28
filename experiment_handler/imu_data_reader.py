import os
import json
import numpy as np
from experiment_handler.time_synchronisation import convert_timestamps
from experiment_handler.finder import find_all_imu_files


def load_imu_file(filepath):
    lines = []
    with open(filepath, 'r') as file:
        try:
            lines = file.read().split("\n")
        except UnicodeDecodeError as e:
            print(e)

    return lines


def _get_beacon_id(ble_data, use_uuid=True):
    if use_uuid:
        return ble_data['uuid'] + "-" + str(ble_data['major']) + "-" + str(ble_data['minor'])
    else:
        return ble_data['macAdd']


def find_and_categorize_beacon_ids(experiments, threshold=45, save_ids=True):
    """
    Find bluetooth beacon ids and count how many times they occur during the given experiments
    Observation showed that some beacon ids randomly appear only a few times during the experiments
    This function sort the ids into two categories (static id: appearing again and again, changing ids: occuring only a few times)


    Parameters
    ----------
    experiments: list of str
        List of pathes to the experiment roots
    threshold: int
        Defines separation threshold. Beacon ids with equal or less then this many detection are defined as changing id,
    save_ids: boolean
        If True result dictionary is saved to 'beacon_ids.json'

    Returns
    -------
        static_ids: list of str
            IDs with more detection than the threshold
        changing_ids: list of str
            IDs with less or equal detection than the threshold
        ids_detection_counts: dict
            Dictionary with id as key and count of detection as value

    """
    # Recompute set of ids from bluetooth beacons which are apparently static
    ids_detection_counts = {}

    for exp_root in experiments:
        imu_files = find_all_imu_files(exp_root)
        for filepath in imu_files:
            imu_lines = load_imu_file(filepath)

            for imu_line in imu_lines:
                try:
                    data = json.loads(imu_line)
                except json.decoder.JSONDecodeError:
                    continue

                if data['type'] != 'ble':
                    continue

                for beacon in data['beacons']:
                    current_id = _get_beacon_id(beacon)

                    if current_id in ids_detection_counts.keys():
                        ids_detection_counts[current_id] += 1
                    else:
                        ids_detection_counts[current_id] = 1

    static_ids = []
    changing_ids = []

    for key in ids_detection_counts.keys():
        if ids_detection_counts[key] > threshold:
            static_ids.append(key)
        else:
            changing_ids.append(key)

    # save ids into file
    if save_ids:
        sorted_ids = {
            "static_ids": static_ids,
            "changing_ids": changing_ids
        }
        with open('beacon_ids.json', 'w') as fp:
            json.dump(sorted_ids, fp)

    return static_ids, changing_ids, ids_detection_counts


def load_beacon_ids():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'beacon_ids.json'), 'r') as fp:
        sorted_ids = json.load(fp)
        return sorted_ids


def get_ble_data(experiment_path, source, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read beacon data for a given source (e.g. P3 left hand) in a time interval

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
        parsed_data_rssi: numpy array
            beacon's rssi data with columns order:  <timestamp>, <rssi value of unique id 0>, ...
        parsed_data_tx: numpy array
            beacon's tx data with columns order:  <timestamp>, <tx value of unique id 0>, ...
        unique_ids: list of strings
            Containing unique beacon ids in the same order as the columns above (offseted by timestamp)
    """

    filepath = os.path.join(experiment_path, "imu", source + ".log")
    imu_lines = load_imu_file(filepath)

    # Convert start and end time:
    imu_reference_time = source.split("_")[0] + "_IMU"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, imu_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, imu_reference_time)

    ids = load_beacon_ids()
    unique_ids = ids['static_ids']
    parsed_data_tx = np.zeros((0, len(unique_ids) + 1))
    parsed_data_rssi = np.zeros((0, len(unique_ids) + 1))

    # Parse lines:
    for imu_line in imu_lines:
        try:
            data = json.loads(imu_line)
        except json.decoder.JSONDecodeError:
            continue

        if data['type'] != 'ble':
            continue

        new_data_entry_tx = np.zeros((1, len(unique_ids) + 1))
        new_data_entry_rssi = np.zeros((1, len(unique_ids) + 1))

        new_data_entry_rssi[0, 0] = data['time']
        new_data_entry_tx[0, 0] = data['time']

        for beacon in data['beacons']:
            current_id = _get_beacon_id(beacon)
            if current_id in unique_ids:
                column_index = unique_ids.index(current_id)
                new_data_entry_rssi[0, column_index + 1] = beacon['rssi']
                new_data_entry_tx[0, column_index + 1] = beacon['tx']
        parsed_data_tx = np.append(parsed_data_tx, new_data_entry_tx, axis=0)
        parsed_data_rssi = np.append(parsed_data_rssi, new_data_entry_rssi, axis=0)

    if start is not None:
        parsed_data_tx = parsed_data_tx[parsed_data_tx[:, 0] >= start_timestamp, :]
        parsed_data_rssi = parsed_data_rssi[parsed_data_rssi[:, 0] >= start_timestamp, :]
    if end is not None:
        parsed_data_tx = parsed_data_tx[parsed_data_tx[:, 0] <= end_timestamp, :]
        parsed_data_rssi = parsed_data_rssi[parsed_data_rssi[:, 0] <= end_timestamp, :]

    if convert_time:
        parsed_data_rssi[:, 0] = convert_timestamps(experiment_path, parsed_data_rssi[:, 0], imu_reference_time,
                                               reference_time)
        parsed_data_tx[:, 0] = convert_timestamps(experiment_path, parsed_data_tx[:, 0], imu_reference_time,
                                               reference_time)

    return parsed_data_rssi, parsed_data_tx, unique_ids



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
    if False:
        experiments = [
            "/Users/hevesi/ownCloud/Datasets/igroups_experiment_8",
            "/Users/hevesi/ownCloud/Datasets/igroups_experiment_9"
        ]
        good, bad, id_det_counts = find_and_categorize_beacon_ids(experiments)
        print(good)
        print("Bad:")
        print(bad)

    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    exp_root = "/Users/hevesi/ownCloud/Datasets/igroups_experiment_8"

    data = get_ble_data(exp_root, "P2_imu_head", start=1800, end=2100, reference_time="video", convert_time=True)
    #data = get_imu_data(exp_root, "P3_imu_left", start=1800, end=2368, reference_time="video", convert_time=True)
    print(data[2])

    import matplotlib.pyplot as plt

    plt.imshow(data[0][:, 1:].T)
    plt.show()

