import numpy as np
from experiment_handler.time_synchronisation import convert_timestamps


location_number_lookup = {
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5
}


def generate_sample_times(start, end, step):
    return np.arange(start, end, step)


def get_last_known_location_label_for_person(ground_truth, person, time, reference, experiment_root):
    p_locs = ground_truth[person]

    loc = location_number_lookup[p_locs[0]['location']]

    ct = convert_timestamps(experiment_root, time, reference, p_locs[0]['reference'])
    for p_loc in p_locs:
        if p_loc['start'] <= ct < p_loc['end']:
            loc = location_number_lookup[p_loc["location"]]
            break
        if ct > p_loc['end']:
            loc = location_number_lookup[p_loc["location"]]

    return loc


def get_location_of_persons_at_samples(location_labels, sample_times, experiment_root):
    locations = []

    for t in sample_times:
        p1_loc = get_last_known_location_label_for_person(location_labels, "P1", t, "video", experiment_root)
        p2_loc = get_last_known_location_label_for_person(location_labels, "P2", t, "video", experiment_root)
        p3_loc = get_last_known_location_label_for_person(location_labels, "P3", t, "video", experiment_root)
        p4_loc = get_last_known_location_label_for_person(location_labels, "P4", t, "video", experiment_root)
        current = [t, p1_loc, p2_loc, p3_loc, p4_loc]
        locations.append(current)

    return np.array(locations)


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