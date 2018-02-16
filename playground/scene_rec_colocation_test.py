import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from experiment_handler.time_synchronisation import convert_timestamps
from experiment_handler.label_data_reader import read_experiment_phases, read_location_labels
from feature_calculations.colocation.common import get_location_of_persons_at_samples


def read_roi_scene_recognitions_for_person(exp_root, person_id):
    filepath = os.path.join(exp_root, "processed_data", "scene_recognition_results", person_id + "_et_scene_classes.pkl")
    data = pd.read_pickle(filepath)
    return data


def generate_sample_times(start, end, step):
    return np.arange(start, end, step)


def calc_feature_distances(exp_root, times, reference_time="video", filename="tmp.pkl"):
    p1_features = read_roi_scene_recognitions_for_person(exp_root, "P1")
    p2_features = read_roi_scene_recognitions_for_person(exp_root, "P2")
    p3_features = read_roi_scene_recognitions_for_person(exp_root, "P3")
    p4_features = read_roi_scene_recognitions_for_person(exp_root, "P4")

    differences = []

    for t in times:
        t_et = convert_timestamps(exp_root, t, reference_time, "P1_eyetracker")
        ts_d = p1_features['timestamp'] - t_et
        idx = ts_d.abs().argmin()
        p1_f = p1_features.loc[idx]['predictions']

        t_et = convert_timestamps(exp_root, t, reference_time, "P2_eyetracker")
        ts_d = p2_features['timestamp'] - t_et
        idx = ts_d.abs().argmin()
        p2_f = p2_features.loc[idx]['predictions']

        t_et = convert_timestamps(exp_root, t, reference_time, "P3_eyetracker")
        ts_d = p3_features['timestamp'] - t_et
        idx = ts_d.abs().argmin()
        p3_f = p3_features.loc[idx]['predictions']

        t_et = convert_timestamps(exp_root, t, reference_time, "P4_eyetracker")
        ts_d = p4_features['timestamp'] - t_et
        idx = ts_d.abs().argmin()
        p4_f = p4_features.loc[idx]['predictions']

        diff = {}
        diff["timestamp"] = t
        diff["P1-P2"] = np.linalg.norm(p1_f - p2_f)
        diff["P1-P3"] = np.linalg.norm(p1_f - p3_f)
        diff["P1-P4"] = np.linalg.norm(p1_f - p4_f)
        diff["P2-P3"] = np.linalg.norm(p2_f - p3_f)
        diff["P2-P4"] = np.linalg.norm(p2_f - p4_f)
        diff["P3-P4"] = np.linalg.norm(p3_f - p4_f)

        differences.append(diff)

    diff_df = pd.DataFrame(differences)
    diff_df.to_pickle(filename)
    return diff_df


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


def test_colocation(coloc_features, threshold):
    colocation = {}

    colocation["P1"] = {
        "P2": np.ones((len(coloc_features), 2)),
        "P3": np.ones((len(coloc_features), 2)),
        "P4": np.ones((len(coloc_features), 2))
    }
    colocation["P2"] = {
        "P1": np.ones((len(coloc_features), 2)),
        "P3": np.ones((len(coloc_features), 2)),
        "P4": np.ones((len(coloc_features), 2))
    }
    colocation["P3"] = {
        "P1": np.ones((len(coloc_features), 2)),
        "P2": np.ones((len(coloc_features), 2)),
        "P4": np.ones((len(coloc_features), 2))
    }
    colocation["P4"] = {
        "P1": np.ones((len(coloc_features), 2)),
        "P3": np.ones((len(coloc_features), 2)),
        "P2": np.ones((len(coloc_features), 2))
    }

    value = 1
    #value = np.random.randint(0, 2)
    for idx, row in coloc_features.iterrows():
        colocation["P1"]["P2"][idx, 0] = row['timestamp']
        colocation["P1"]["P2"][idx, 1] = value

        colocation["P1"]["P3"][idx, 0] = row['timestamp']
        colocation["P1"]["P3"][idx, 1] = value

        colocation["P1"]["P4"][idx, 0] = row['timestamp']
        colocation["P1"]["P4"][idx, 1] = value

        colocation["P2"]["P1"][idx, 0] = row['timestamp']
        colocation["P2"]["P1"][idx, 1] = value

        colocation["P2"]["P4"][idx, 0] = row['timestamp']
        colocation["P2"]["P4"][idx, 1] = value

        colocation["P2"]["P3"][idx, 0] = row['timestamp']
        colocation["P2"]["P3"][idx, 1] = value

        colocation["P3"]["P1"][idx, 0] = row['timestamp']
        colocation["P3"]["P1"][idx, 1] = value

        colocation["P3"]["P4"][idx, 0] = row['timestamp']
        colocation["P3"]["P4"][idx, 1] = value

        colocation["P3"]["P2"][idx, 0] = row['timestamp']
        colocation["P3"]["P2"][idx, 1] = value

        colocation["P4"]["P1"][idx, 0] = row['timestamp']
        colocation["P4"]["P1"][idx, 1] = value

        colocation["P4"]["P3"][idx, 0] = row['timestamp']
        colocation["P4"]["P3"][idx, 1] = value

        colocation["P4"]["P2"][idx, 0] = row['timestamp']
        colocation["P4"]["P2"][idx, 1] = value

    return colocation


def detect_colocation(coloc_features, threshold):
    colocation = {}

    colocation["P1"] = {
        "P2": np.ones((len(coloc_features), 2)),
        "P3": np.ones((len(coloc_features), 2)),
        "P4": np.ones((len(coloc_features), 2))
    }
    colocation["P2"] = {
        "P1": np.ones((len(coloc_features), 2)),
        "P3": np.ones((len(coloc_features), 2)),
        "P4": np.ones((len(coloc_features), 2))
    }
    colocation["P3"] = {
        "P1": np.ones((len(coloc_features), 2)),
        "P2": np.ones((len(coloc_features), 2)),
        "P4": np.ones((len(coloc_features), 2))
    }
    colocation["P4"] = {
        "P1": np.ones((len(coloc_features), 2)),
        "P3": np.ones((len(coloc_features), 2)),
        "P2": np.ones((len(coloc_features), 2))
    }

    for idx, row in coloc_features.iterrows():
        colocation["P1"]["P2"][idx, 0] = row['timestamp']
        colocation["P1"]["P2"][idx, 1] = 0 if row["P1-P2"] > threshold else 1

        colocation["P1"]["P3"][idx, 0] = row['timestamp']
        colocation["P1"]["P3"][idx, 1] = 0 if row["P1-P3"] > threshold else 1

        colocation["P1"]["P4"][idx, 0] = row['timestamp']
        colocation["P1"]["P4"][idx, 1] = 0 if row["P1-P4"] > threshold else 1

        colocation["P2"]["P1"][idx, 0] = row['timestamp']
        colocation["P2"]["P1"][idx, 1] = 0 if row["P1-P2"] > threshold else 1

        colocation["P2"]["P4"][idx, 0] = row['timestamp']
        colocation["P2"]["P4"][idx, 1] = 0 if row["P2-P4"] > threshold else 1

        colocation["P2"]["P3"][idx, 0] = row['timestamp']
        colocation["P2"]["P3"][idx, 1] = 0 if row["P2-P3"] > threshold else 1

        colocation["P3"]["P1"][idx, 0] = row['timestamp']
        colocation["P3"]["P1"][idx, 1] = 0 if row["P1-P3"] > threshold else 1

        colocation["P3"]["P4"][idx, 0] = row['timestamp']
        colocation["P3"]["P4"][idx, 1] = 0 if row["P3-P4"] > threshold else 1

        colocation["P3"]["P2"][idx, 0] = row['timestamp']
        colocation["P3"]["P2"][idx, 1] = 0 if row["P2-P3"] > threshold else 1

        colocation["P4"]["P1"][idx, 0] = row['timestamp']
        colocation["P4"]["P1"][idx, 1] = 0 if row["P1-P4"] > threshold else 1

        colocation["P4"]["P3"][idx, 0] = row['timestamp']
        colocation["P4"]["P3"][idx, 1] = 0 if row["P3-P4"] > threshold else 1

        colocation["P4"]["P2"][idx, 0] = row['timestamp']
        colocation["P4"]["P2"][idx, 1] = 0 if row["P2-P4"] > threshold else 1

    return colocation


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
    if sums[0] > 0:
        p = sums[0] / (sums[0] + sums[1])
    else:
        p = 0

    r = sums[0] / (sums[0] + sums[2])
    a = (sums[0] + sums[3]) / (sums[0] + sums[3] + sums[1] + sums[2])
    f1 = 2*sums[0] / (2*sums[0] + sums[1] + sums[2])
    fpr = sums[1] / (sums[1] + sums[3])

    return p, r, a, f1, fpr


def create_colocation_plots(colocation_labels, colocation_detections):
    # get step size:
    stepsize = colocation_detection["P1"]["P2"][1][0] - colocation_detection["P1"]["P2"][0][0]

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


def find_best_threshold_workflow(coloc_features, coloc_labels, threshold_min, threshold_max, th_stepsize):
    """
    Testing a range of threshold and printing evaluations scores (Precision, recall and f1 score)
    """

    roc = []
    for threshold in range(threshold_min, threshold_max+th_stepsize, th_stepsize):
        threshold = float(threshold)/1000.0
        detected_colocations = detect_colocation(coloc_features, threshold)
        scores = calculate_score(coloc_labels, detected_colocations)

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


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    force_recalc = False
    sample_step = 0.5

    phases = read_experiment_phases(exp_root)
    times = generate_sample_times(phases['assembly'][0], phases['disassembly'][1], sample_step)
    #times = generate_sample_times(sample_start, sample_end, sample_step)

    start = datetime.now()

    if force_recalc or not os.path.exists("tmp.pkl"):
        print("Warning: (re)generating features could take several minutes to compute.")
        coloc_features = calc_feature_distances(exp_root, times, filename="tmp.pkl")
    else:
        coloc_features = pd.read_pickle("tmp.pkl")

    # Read labels:
    location_labels = read_location_labels(exp_root)
    locations = get_location_of_persons_at_samples(location_labels, times, exp_root)
    colocation_labels = get_colocation_labels(locations)

    if False:
        colocation_detection = detect_colocation(coloc_features, 1000)
        plt.figure(figsize=(8, 4))
        plt.title("P1 - P2")
        plt.plot(colocation_detection["P1"]["P2"][:, 0], colocation_detection["P1"]["P2"][:, 1])
        plt.grid()
        plt.draw()
        plt.figure(figsize=(8, 4))
        plt.title("P1 - P3")
        plt.plot(colocation_detection["P1"]["P3"][:, 0], colocation_detection["P1"]["P3"][:, 1])
        plt.grid()
        plt.draw()
        plt.figure(figsize=(8, 4))
        plt.title("P1 - P4")
        plt.plot(colocation_detection["P1"]["P4"][:, 0], colocation_detection["P1"]["P4"][:, 1])
        plt.grid()
        plt.show()

    #find_best_threshold_workflow(coloc_features, colocation_labels, 1, 30, 1)


    #print(coloc_features)
    colocation_detection = detect_colocation(coloc_features, 0.0115)
    #print(colocation_detection)


    print("Runtime:", datetime.now() - start)
    score = calculate_score(colocation_labels, colocation_detection)
    print(score)
    #exit()

    create_colocation_plots(colocation_labels, colocation_detection)
