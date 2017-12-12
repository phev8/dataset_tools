import numpy as np
import matplotlib.pyplot as plt
from experiment_handler.label_data_reader import read_experiment_phases, read_location_labels
from experiment_handler.imu_data_reader import get_ble_data

from playground.ble_common_methods import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# TODO: train model for detecting locations using data from one experiment
def train_model():
    # TODO: create one big training matrix
    #create_training_matrices()
    # TODO: train classifier
    # TODO: save model
    return


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


    p1_beacons = \
        get_ble_data(experiment, "P1_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
    p2_beacons = \
        get_ble_data(experiment, "P2_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
    p3_beacons = \
        get_ble_data(experiment, "P3_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]
    p4_beacons = \
        get_ble_data(experiment, "P4_imu_head", start=None, end=None, reference_time="video", convert_time=True)[0]

    data = {
        "P1": np.zeros((len(sample_times), 1 + p1_beacons.shape[1])),
        "P2": np.zeros((len(sample_times), 1 + p1_beacons.shape[1])),
        "P3": np.zeros((len(sample_times), 1 + p1_beacons.shape[1])),
        "P4": np.zeros((len(sample_times), 1 + p1_beacons.shape[1])),
    }

    for idx, loc_label in enumerate(locations):
        t = loc_label[0]
        p1 = get_last_beacon_info(t, p1_beacons)
        p2 = get_last_beacon_info(t, p2_beacons)
        p3 = get_last_beacon_info(t, p3_beacons)
        p4 = get_last_beacon_info(t, p4_beacons)

        data["P1"][idx, :-1] = p1
        data["P1"][idx, -1] = loc_label[1]

        data["P2"][idx, :-1] = p2
        data["P2"][idx, -1] = loc_label[2]

        data["P3"][idx, :-1] = p3
        data["P3"][idx, -1] = loc_label[3]

        data["P4"][idx, :-1] = p4
        data["P4"][idx, -1] = loc_label[4]

    return data


# TODO: test classifiers for location detections using leave one person out for one experiment
def test_location_detection_with_one_experiment(experiment_root, sample_distance):
    phases = read_experiment_phases(experiment)
    times = generate_sample_times(phases['assembly'][0], phases['disassembly'][1], sample_distance)

    data_matrices = create_training_matrices(experiment_root, times)

    person_list = ["P1", "P2", "P3", "P4"]
    scores = []

    for for_training in person_list:
        X_train = data_matrices[for_training][:, 1:-1]
        y_train = data_matrices[for_training][:, -1]

        X_test = np.zeros((0, X_train.shape[1]))
        y_test = np.zeros((0))

        for p in person_list:
            if p == for_training:
                continue
            X_test = np.append(X_test, data_matrices[p][:, 1:-1], axis=0)
            y_test = np.append(y_test, data_matrices[p][:, -1], axis=0)

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
    X_train = data_matrices["P2"][:, 1:-1]
    y_train = data_matrices["P2"][:, -1]

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
    experiment = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    stepsize = 3

    test_location_detection_with_one_experiment(experiment, stepsize)
