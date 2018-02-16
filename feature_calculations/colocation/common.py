import numpy as np
from experiment_handler.time_synchronisation import convert_timestamps

location_number_lookup = {
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5
}


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