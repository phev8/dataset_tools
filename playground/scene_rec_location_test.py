import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from experiment_handler.time_synchronisation import convert_timestamps
from experiment_handler.label_data_reader import read_experiment_phases, read_location_labels
from feature_calculations.colocation.common import get_location_of_persons_at_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def read_roi_scene_recognitions_for_person(exp_root, person_id):
    filepath = os.path.join(exp_root, "processed_data", "scene_recognition_results", person_id + "_et_scene_classes.pkl")
    data = pd.read_pickle(filepath)
    return data


def generate_sample_times(start, end, step):
    return np.arange(start, end, step)


def create_training_matrices(experiment, sample_times):
    """
    Create a training matrix (including sample time, feature vector and label) for each person

    Parameters
    ----------
    experiment: str
        Path to the root of the selected experiment
    sample_times: array like
        List of times where the samples should be generated.

    Returns
    -------
        data: dict
            with keys: person IDs and values: numpy array with columns [t, beacon_detection_vector, location_label]
    """
    # Labels:
    location_labels = read_location_labels(experiment)
    locations = get_location_of_persons_at_samples(location_labels, sample_times, experiment)

    p1_scene_features = read_roi_scene_recognitions_for_person(experiment, "P1")
    p2_scene_features = read_roi_scene_recognitions_for_person(experiment, "P2")
    p3_scene_features = read_roi_scene_recognitions_for_person(experiment, "P3")
    p4_scene_features = read_roi_scene_recognitions_for_person(experiment, "P4")

    print(p1_scene_features.loc[0]['predictions'].shape[0])
    print(len(p1_scene_features['predictions'][0]))
    data = {
        "P1": np.zeros((len(sample_times), 2 + p1_scene_features.loc[0]['predictions'].shape[0])),
        "P2": np.zeros((len(sample_times), 2 + p1_scene_features.loc[0]['predictions'].shape[0])),
        "P3": np.zeros((len(sample_times), 2 + p1_scene_features.loc[0]['predictions'].shape[0])),
        "P4": np.zeros((len(sample_times), 2 + p1_scene_features.loc[0]['predictions'].shape[0]))
    }

    for index, loc_label in enumerate(locations):
        t = loc_label[0]

        t_et = convert_timestamps(exp_root, t, "video", "P1_eyetracker")
        p1 = np.mean(p1_scene_features[p1_scene_features['timestamp'].between(t_et - sample_step/2, t_et + sample_step/2)]['predictions'].as_matrix(), axis=0)


        t_et = convert_timestamps(exp_root, t, "video", "P2_eyetracker")
        p2 = np.mean(
            p2_scene_features[p2_scene_features['timestamp'].between(t_et - sample_step / 2, t_et + sample_step / 2)][
                'predictions'].as_matrix(), axis=0)

        t_et = convert_timestamps(exp_root, t, "video", "P3_eyetracker")
        p3 = np.mean(
            p3_scene_features[p3_scene_features['timestamp'].between(t_et - sample_step / 2, t_et + sample_step / 2)][
                'predictions'].as_matrix(), axis=0)

        t_et = convert_timestamps(exp_root, t, "video", "P4_eyetracker")
        p4 = np.mean(
            p4_scene_features[p4_scene_features['timestamp'].between(t_et - sample_step / 2, t_et + sample_step / 2)][
                'predictions'].as_matrix(), axis=0)

        data["P1"][index, 0] = t
        data["P1"][index, 1:-1] = p1
        data["P1"][index, -1] = loc_label[1]

        data["P2"][index, 0] = t
        data["P2"][index, 1:-1] = p2
        data["P2"][index, -1] = loc_label[2]

        data["P3"][index, 0] = t
        data["P3"][index, 1:-1] = p3
        data["P3"][index, -1] = loc_label[3]

        data["P4"][index, 0] = t
        data["P4"][index, 1:-1] = p4
        data["P4"][index, -1] = loc_label[4]

    return data


# TODO: test classifiers for location detections using leave one person out for one experiment
def test_location_detection_with_one_experiment(experiment_root, sample_distance):
    phases = read_experiment_phases(exp_root)
    times = generate_sample_times(phases['assembly'][0], phases['disassembly'][1], sample_distance)

    data_matrices = create_training_matrices(experiment_root, times)

    person_list = ["P1", "P2", "P3", "P4"]
    scores = []

    for for_training in person_list:
        X_train = data_matrices[for_training][:, 1:-1]
        y_train = data_matrices[for_training][:, -1]

        y_train[y_train == 5] = 4

        X_test = np.zeros((0, X_train.shape[1]))
        y_test = np.zeros((0))

        for p in person_list:
            if p == for_training:
                continue
            X_test = np.append(X_test, data_matrices[p][:, 1:-1], axis=0)
            y_test = np.append(y_test, data_matrices[p][:, -1], axis=0)
            y_test[y_test == 5] = 4

        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        score = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print(for_training, score)
        scores.append(score)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print(cnf_matrix)
        print("----------")

    scores = np.array(scores)
    print(np.mean(scores[:, :-1], axis=0))

    # TODO: confusion matrix

    # event plot if trained for one person
    X_train = data_matrices["P1"][:, 1:-1]
    y_train = data_matrices["P1"][:, -1]

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)

    f, axarr = plt.subplots(4, sharex=True, figsize=(16, 10))
    for idx, test_for in enumerate(person_list):
        t_test = data_matrices[test_for][:, 0]
        y_test = data_matrices[test_for][:, -1]
        y_pred = clf.predict(data_matrices[test_for][:, 1:-1])

        axarr[idx].plot(t_test, y_test, 'o', label="Ground truth")
        axarr[idx].plot(t_test, y_pred, 'x', label="Detection")
        axarr[idx].grid()
        axarr[idx].legend()
        axarr[idx].set_title(test_for + " locations")
        axarr[idx].set_ylabel("Location id")

    plt.xlabel("Time [s]")
    plt.show()


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    sample_step = 3.0

    test_location_detection_with_one_experiment(exp_root, sample_step)