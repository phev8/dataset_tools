import numpy as np


def generate_sample_times(start, end, step):
    return np.arange(start, end, step)


def get_last_beacon_info(time, beacon_data):
    """
    Find latest beacon data before the given time (last seen beacons)

    Parameters
    ----------
    time: float
        Current time to look up beacon data for
    beacon_data: numpy array
        Matrix of the beacon data as got from get_ble_data(...) method (e.g. first return value)

    Returns
    -------
        last_seen_beacons: numpy array
            One row of the beacon_data matrix, right before the selected time (or full or zeros if none found)
    """
    last_seen_beacons = np.zeros((1, beacon_data.shape[1]))
    last_seen_beacons[0] = time

    time_diff = beacon_data[:, 0] - time

    tmp = beacon_data[time_diff <= 0, :]
    time_diff = time_diff[time_diff <= 0]
    if len(tmp) > 0:
        index = np.argmax(time_diff)
        last_seen_beacons[0, 1:] = tmp[index, 1:]
    return last_seen_beacons