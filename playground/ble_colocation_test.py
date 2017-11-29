import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from experiment_handler.label_data_reader import read_experiment_phases, read_location_labels
from experiment_handler.imu_data_reader import get_ble_data

from experiment_handler.time_synchronisation import convert_timestamps

location_number_lookup = {
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5
}

# TODO: generate sample times
def generate_sample_times(start, end, step):
    return np.arange(start, end, step)


def get_last_known_location_label_for_person(ground_truth, person, time, reference):
    p_locs = ground_truth[person]

    loc = location_number_lookup[p_locs[0]['location']]

    ct = convert_timestamps(experiment, time, reference, p_locs[0]['reference'])
    for p_loc in p_locs:
        if p_loc['start'] <= ct < p_loc['end']:
            loc = location_number_lookup[p_loc["location"]]
            break
        if ct > p_loc['end']:
            loc = location_number_lookup[p_loc["location"]]

    return loc


def get_location_of_persons_at_samples(location_labels, sample_times):
    locations = []

    for t in sample_times:
        p1_loc = get_last_known_location_label_for_person(location_labels, "P1", t, "video")
        p2_loc = get_last_known_location_label_for_person(location_labels, "P2", t, "video")
        p3_loc = get_last_known_location_label_for_person(location_labels, "P3", t, "video")
        p4_loc = get_last_known_location_label_for_person(location_labels, "P4", t, "video")
        current = [t, p1_loc, p2_loc, p3_loc, p4_loc]
        locations.append(current)

    return np.array(locations)


def get_colocation_labels(locations_at_samples):
    colocation = {}
    times = locations_at_samples[:, 0]

    colocation["P1"] = {
        "P2": np.ones((len(locations_at_samples), 2)),
        "P3": np.ones((len(locations_at_samples), 2)),
        "P4": np.ones((len(locations_at_samples), 2))
    }
    colocation["P2"] = {
        "P1": np.ones((len(locations_at_samples), 2)),
        "P3": np.ones((len(locations_at_samples), 2)),
        "P4": np.ones((len(locations_at_samples), 2))
    }
    colocation["P3"] = {
        "P1": np.ones((len(locations_at_samples), 2)),
        "P2": np.ones((len(locations_at_samples), 2)),
        "P4": np.ones((len(locations_at_samples), 2))
    }
    colocation["P4"] = {
        "P1": np.ones((len(locations_at_samples), 2)),
        "P3": np.ones((len(locations_at_samples), 2)),
        "P2": np.ones((len(locations_at_samples), 2))
    }

    colocation["P1"]["P2"][:, 0] = times
    colocation["P1"]["P2"][(locations_at_samples[:, 1] - locations_at_samples[:, 2]) != 0, 1] = 0
    colocation["P1"]["P3"][:, 0] = times
    colocation["P1"]["P3"][(locations_at_samples[:, 1] - locations_at_samples[:, 3]) != 0, 1] = 0
    colocation["P1"]["P4"][:, 0] = times
    colocation["P1"]["P4"][(locations_at_samples[:, 1] - locations_at_samples[:, 4]) != 0, 1] = 0

    colocation["P2"]["P1"][:, 0] = times
    colocation["P2"]["P1"][(locations_at_samples[:, 1] - locations_at_samples[:, 2]) != 0, 1] = 0
    colocation["P2"]["P3"][:, 0] = times
    colocation["P2"]["P3"][(locations_at_samples[:, 2] - locations_at_samples[:, 3]) != 0, 1] = 0
    colocation["P2"]["P4"][:, 0] = times
    colocation["P2"]["P4"][(locations_at_samples[:, 2] - locations_at_samples[:, 4]) != 0, 1] = 0

    colocation["P3"]["P1"][:, 0] = times
    colocation["P3"]["P1"][(locations_at_samples[:, 1] - locations_at_samples[:, 3]) != 0, 1] = 0
    colocation["P3"]["P2"][:, 0] = times
    colocation["P3"]["P2"][(locations_at_samples[:, 2] - locations_at_samples[:, 3]) != 0, 1] = 0
    colocation["P3"]["P4"][:, 0] = times
    colocation["P3"]["P4"][(locations_at_samples[:, 3] - locations_at_samples[:, 4]) != 0, 1] = 0

    colocation["P4"]["P1"][:, 0] = times
    colocation["P4"]["P1"][(locations_at_samples[:, 1] - locations_at_samples[:, 4]) != 0, 1] = 0
    colocation["P4"]["P3"][:, 0] = times
    colocation["P4"]["P3"][(locations_at_samples[:, 3] - locations_at_samples[:, 4]) != 0, 1] = 0
    colocation["P4"]["P2"][:, 0] = times
    colocation["P4"]["P2"][(locations_at_samples[:, 2] - locations_at_samples[:, 4]) != 0, 1] = 0

    return colocation


def get_last_beacon_info(time, beacon_data):
    last_seen_beacons = np.zeros((1, beacon_data.shape[1]))
    last_seen_beacons[0] = time

    time_diff = beacon_data[:, 0] - time

    tmp = beacon_data[time_diff <= 0, :]
    time_diff = time_diff[time_diff <= 0]
    if len(tmp) > 0:
        index = np.argmax(time_diff)
        last_seen_beacons[0, 1:] = tmp[index, 1:]
    return last_seen_beacons


def detect_colocation(times, p1_beacons, p2_beacons, p3_beacons, p4_beacons, threshold):
    colocation = {}

    colocation["P1"] = {
        "P2": np.ones((len(times), 2)),
        "P3": np.ones((len(times), 2)),
        "P4": np.ones((len(times), 2))
    }
    colocation["P2"] = {
        "P1": np.ones((len(times), 2)),
        "P3": np.ones((len(times), 2)),
        "P4": np.ones((len(times), 2))
    }
    colocation["P3"] = {
        "P1": np.ones((len(times), 2)),
        "P2": np.ones((len(times), 2)),
        "P4": np.ones((len(times), 2))
    }
    colocation["P4"] = {
        "P1": np.ones((len(times), 2)),
        "P3": np.ones((len(times), 2)),
        "P2": np.ones((len(times), 2))
    }


    for idx, t in enumerate(times):
        p1 = get_last_beacon_info(t, p1_beacons)
        p2 = get_last_beacon_info(t, p2_beacons)
        p3 = get_last_beacon_info(t, p3_beacons)
        p4 = get_last_beacon_info(t, p4_beacons)

        colocation["P1"]["P2"][idx, 0] = t
        colocation["P1"]["P2"][idx, 1] = 0 if np.linalg.norm(p1[0, 1:] - p2[0, 1:]) > threshold else 1

        colocation["P1"]["P3"][idx, 0] = t
        colocation["P1"]["P3"][idx, 1] = 0 if np.linalg.norm(p1[0, 1:] - p3[0, 1:]) > threshold else 1

        colocation["P1"]["P4"][idx, 0] = t
        colocation["P1"]["P4"][idx, 1] = 0 if np.linalg.norm(p1[0, 1:] - p4[0, 1:]) > threshold else 1

        colocation["P2"]["P1"][idx, 0] = t
        colocation["P2"]["P1"][idx, 1] = 0 if np.linalg.norm(p1[0, 1:] - p2[0, 1:]) > threshold else 1

        colocation["P2"]["P4"][idx, 0] = t
        colocation["P2"]["P4"][idx, 1] = 0 if np.linalg.norm(p2[0, 1:] - p4[0, 1:]) > threshold else 1

        colocation["P2"]["P3"][idx, 0] = t
        colocation["P2"]["P3"][idx, 1] = 0 if np.linalg.norm(p2[0, 1:] - p3[0, 1:]) > threshold else 1

        colocation["P3"]["P1"][idx, 0] = t
        colocation["P3"]["P1"][idx, 1] = 0 if np.linalg.norm(p1[0, 1:] - p3[0, 1:]) > threshold else 1

        colocation["P3"]["P4"][idx, 0] = t
        colocation["P3"]["P4"][idx, 1] = 0 if np.linalg.norm(p3[0, 1:] - p4[0, 1:]) > threshold else 1

        colocation["P3"]["P2"][idx, 0] = t
        colocation["P3"]["P2"][idx, 1] = 0 if np.linalg.norm(p2[0, 1:] - p3[0, 1:]) > threshold else 1

        colocation["P4"]["P1"][idx, 0] = t
        colocation["P4"]["P1"][idx, 1] = 0 if np.linalg.norm(p1[0, 1:] - p4[0, 1:]) > threshold else 1

        colocation["P4"]["P3"][idx, 0] = t
        colocation["P4"]["P3"][idx, 1] = 0 if np.linalg.norm(p3[0, 1:] - p4[0, 1:]) > threshold else 1

        colocation["P4"]["P2"][idx, 0] = t
        colocation["P4"]["P2"][idx, 1] = 0 if np.linalg.norm(p2[0, 1:] - p4[0, 1:]) > threshold else 1

    return colocation






def create_colocation_plots(colocation_labels, colocation_detections):
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca() # fig.add_axes([0, 0, 1, 1])
    plt.title("Co-location of persons", loc="left")

    yticks = [
        "P1 - P2",
        "P1 - P3",
        "P1 - P4",
        "P2 - P3",
        "P2 - P4",
        "P3 - P4",
    ]

    height = 0.05

    gt_color = "#1f77b4"
    for values in colocation_labels["P1"]["P2"]:
        if values[1] > 0:
            y_pos = (yticks.index("P1 - P2") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos, y_pos + height)
    for values in colocation_labels["P1"]["P3"]:
        if values[1] > 0:
            y_pos = (yticks.index("P1 - P3") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos, y_pos + height)

    for values in colocation_labels["P1"]["P4"]:
        if values[1] > 0:
            y_pos = (yticks.index("P1 - P4") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos, y_pos + height)

    for values in colocation_labels["P2"]["P3"]:
        if values[1] > 0:
            y_pos = (yticks.index("P2 - P3") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos, y_pos + height)

    for values in colocation_labels["P2"]["P4"]:
        if values[1] > 0:
            y_pos = (yticks.index("P2 - P4") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos, y_pos + height)

    for values in colocation_labels["P3"]["P4"]:
        if values[1] > 0:
            y_pos = (yticks.index("P3 - P4") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos, y_pos + height)

    det_color = "orange"
    for values in colocation_detections["P1"]["P2"]:
        if values[1] > 0:
            y_pos = (yticks.index("P1 - P2") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos - height, y_pos, color=det_color)
    for values in colocation_detections["P1"]["P3"]:
        if values[1] > 0:
            y_pos = (yticks.index("P1 - P3") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos - height, y_pos, color=det_color)

    for values in colocation_detections["P1"]["P4"]:
        if values[1] > 0:
            y_pos = (yticks.index("P1 - P4") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos - height, y_pos, color=det_color)

    for values in colocation_detections["P2"]["P3"]:
        if values[1] > 0:
            y_pos = (yticks.index("P2 - P3") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos - height, y_pos, color=det_color)

    for values in colocation_detections["P2"]["P4"]:
        if values[1] > 0:
            y_pos = (yticks.index("P2 - P4") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos - height, y_pos, color=det_color)

    for values in colocation_detections["P3"]["P4"]:
        if values[1] > 0:
            y_pos = (yticks.index("P3 - P4") + 1) * (1/7)
            plt.axvspan(values[0] - stepsize / 2, values[0] + stepsize / 2, y_pos - height, y_pos, color=det_color)


    plt.grid()
    plt.ylim([0, len(yticks)+1])
    plt.yticks(range(1, len(yticks)+1), yticks)
    plt.tight_layout()
    plt.subplots_adjust(top=0.83)

    box_height = 0.07
    box_y = 1.02

    left_box_x = 0.5
    right_box_x = 0.75
    det_legend = Rectangle((right_box_x, box_y), 0.2, box_height, color=det_color, transform=ax.transAxes, clip_on=False)
    gt_legend = Rectangle((left_box_x, box_y), 0.2, box_height, color=gt_color, transform=ax.transAxes, clip_on=False)

    ax.text(right_box_x + 0.1, box_y + 0.5*box_height, 'Detections',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10, color='#444433',
            transform=ax.transAxes)

    ax.text(left_box_x + 0.1, box_y + 0.5*box_height, 'Ground Truth',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10, color='white',
            transform=ax.transAxes)

    ax.add_patch(det_legend)
    ax.add_patch(gt_legend)
    plt.show()


def _get_score_for_single_pair(labels, detections):
    """

    Parameters
    ----------
    labels: numpy array (Nx2)
        Co-location ground truth (first column = sample time, second column = 0 if not colocated 1 if colocated)
    detections: numpy array (Nx2)
        Detected Co-locations (first column = sample time, second column = 0 if not colocated 1 if colocated)

    Returns
    -------
        scores: list
            List of four items. 1: number of true positives 2: number false positives
            3: number of false negatives 4: number of true negatives

    """
    tmp = labels[:, 1] - detections[:, 1]

    fn = np.count_nonzero(tmp == 1)
    fp = np.count_nonzero(tmp == -1)
    tp = np.count_nonzero(labels[tmp == 0, 1])
    tn = len(tmp) - fn - fp - tp

    return [tp, fp, fn, tn]


def calculate_score(colocation_labels, colocation_detections):
    """
    Calculating some standard frame-by-frame scores for colocation detections

    Parameters
    ----------
    colocation_labels
    colocation_detections

    Returns
    -------
        p: float
            Precision
        r: float
            Recall
        a: float
            Accuracy
        f1: float
            F1-score
        fpr: float
            False Positive Rate

    """
    p1p2_scores = _get_score_for_single_pair(colocation_labels["P1"]["P2"], colocation_detections["P1"]["P2"])
    p1p3_scores = _get_score_for_single_pair(colocation_labels["P1"]["P3"], colocation_detections["P1"]["P3"])
    p1p4_scores = _get_score_for_single_pair(colocation_labels["P1"]["P4"], colocation_detections["P1"]["P4"])
    p2p3_scores = _get_score_for_single_pair(colocation_labels["P2"]["P3"], colocation_detections["P2"]["P3"])
    p2p4_scores = _get_score_for_single_pair(colocation_labels["P2"]["P4"], colocation_detections["P2"]["P4"])
    p3p4_scores = _get_score_for_single_pair(colocation_labels["P3"]["P4"], colocation_detections["P3"]["P4"])

    scores = np.array([
        p1p2_scores, p1p3_scores, p1p4_scores, p2p3_scores, p2p4_scores, p3p4_scores
    ])
    sums = np.sum(scores, axis=0)
    p = sums[0] / (sums[0] + sums[1])
    r = sums[0] / (sums[0] + sums[2])
    a = (sums[0] + sums[3]) / (sums[0] + sums[3] + sums[1] + sums[2])
    f1 = 2*sums[0] / (2*sums[0] + sums[1] + sums[2])
    fpr = sums[1] / (sums[1] + sums[3])

    return p, r, a, f1, fpr


def find_best_threshold_workflow(experiment, sample_step_size, threshold_min, threshold_max, th_stepsize):
    """
    Testing a range of threshold and printing evaluations scores (Precision, recall and f1 score)

    Parameters
    ----------
    experiment: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    sample_step_size: int
        Generating samples between start and end of the experiment with this step size (seconds)
    threshold_min: int
        Minimum of the range where thresholds are tested
    threshold_max: int
        Maximum of the range where thresholds are tested
    th_stepsize: int
        Thresholds are incremented with this value in each iteration

    Returns
    -------
        None
    """
    phases = read_experiment_phases(experiment)
    times = generate_sample_times(phases['assembly'][0], phases['disassembly'][1], sample_step_size)

    # Read & convert Labels:
    location_labels = read_location_labels(experiment)
    locations = get_location_of_persons_at_samples(location_labels, times)

    colocation_labels = get_colocation_labels(locations)

    # Read & convert Beacon data:
    p1_beacons = \
    get_ble_data(experiment, "P1_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
    p2_beacons = \
    get_ble_data(experiment, "P2_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
    p3_beacons = \
    get_ble_data(experiment, "P3_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
    p4_beacons = \
    get_ble_data(experiment, "P4_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]

    roc = []
    for threshold in range(threshold_min, threshold_max+stepsize, th_stepsize):
        detected_colocations = detect_colocation(times, p1_beacons, p2_beacons, p3_beacons, p4_beacons, threshold)
        scores = calculate_score(colocation_labels, detected_colocations)

        print("\n----------- Score for threshold = " + str(threshold) + " ------------")
        print("Precision: " + str(scores[0]) + ", Recall: " + str(scores[1]) + ", Accuracy: " + str(scores[2]) + ", F1-score: " + str(scores[3]))
        print("----------- ------------------------- ------------")
        roc.append([scores[4], scores[1]])

    roc = np.array(roc)

    plt.plot(roc[:, 0], roc[:, 1])
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.show()


def save_detected_colocations(colocations):
    with open("tmp_colocations.pkl", "wb") as f:
        pickle.dump(colocations, f)


def load_detected_colocations():
    with open("tmp_colocations.pkl", "rb") as f:
        return pickle.load(f)

if __name__ == '__main__':
    experiment = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    stepsize = 3

    # Use this part to find optimal threshold value:
    if False:
        find_best_threshold_workflow(experiment, stepsize, 150, 550, 10)
        exit()


    threshold = 367
    recalc = False

    phases = read_experiment_phases(experiment)
    times = generate_sample_times(phases['assembly'][0], phases['disassembly'][1], 3)

    # Labels:
    location_labels = read_location_labels(experiment)
    locations = get_location_of_persons_at_samples(location_labels, times)

    colocation_labels = get_colocation_labels(locations)

    # Beacon data:
    if recalc:
        p1_beacons = get_ble_data(experiment, "P1_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
        p2_beacons = get_ble_data(experiment, "P2_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
        p3_beacons = get_ble_data(experiment, "P3_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
        p4_beacons = get_ble_data(experiment, "P4_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]


        detected_colocations = detect_colocation(times, p1_beacons, p2_beacons, p3_beacons, p4_beacons, threshold)
        save_detected_colocations(detected_colocations)
    else:
        detected_colocations = load_detected_colocations()

    if False:
        plt.figure(figsize=(8, 4))
        plt.title("P1 - P2")
        plt.plot(detected_colocations["P1"]["P2"][:, 0], detected_colocations["P1"]["P2"][:, 1])
        plt.grid()
        plt.draw()
        plt.figure(figsize=(8, 4))
        plt.title("P1 - P3")
        plt.plot(detected_colocations["P1"]["P3"][:, 0], detected_colocations["P1"]["P3"][:, 1])
        plt.grid()
        plt.draw()
        plt.figure(figsize=(8, 4))
        plt.title("P1 - P4")
        plt.plot(detected_colocations["P1"]["P4"][:, 0], detected_colocations["P1"]["P4"][:, 1])
        plt.grid()
        plt.show()

        exit()


    print("Start plot")
    create_colocation_plots(colocation_labels, detected_colocations)
    exit()


    for p in colocation_labels["P1"].keys():
        plt.plot(colocation_labels["P1"][p][:, 0], colocation_labels["P1"][p][:, 1], label=p)
    plt.legend()
    plt.show()

    print(locations)