import os
import numpy as np
import pandas as pd
from experiment_handler.label_data_reader import read_activity_labels_from_eyetracker_labelling
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier


def read_eyetracker_feature_file(exp_root, person_id, window_size, labelset="empty", step_size=1000):
    fname = person_id + "_eye_features_SW-" + str(window_size) + "-" + str(step_size) + "_" + labelset + "-labels.pickle"
    filepath = os.path.join(exp_root, "processed_data", "eye_features", fname)
    data = pd.read_pickle(filepath)
    return data


def read_IMU_feature_file(exp_root, person_id, window_size, labelset="empty", step_size=1000, position="right"):
    fname = person_id + "_imu_" + position + "_SW-" + str(window_size) + "-" + str(step_size) + "_" + labelset + "-labels.pickle"
    filepath = os.path.join(exp_root, "processed_data", "imu_features", fname)
    data = pd.read_pickle(filepath)
    return data


def read_sound_feature_file(exp_root, person_id, window_size, labelset="empty", step_size=40):
    fname = person_id + "_sound_features_SW-" + str(window_size) + "-" + str(step_size) + "_" + labelset + "-labels.pickle"
    filepath = os.path.join(exp_root, "processed_data", "sound_features", fname)
    data = pd.read_pickle(filepath)
    return data


def get_class_label_attr(n_class=6):
    if n_class == 1:
        activity_label_replacements = [
            ("walking on the floor", "Large"),
            ("carry tv spacer", "Large"),
            ("carry tools", "Large"),
            ("carry toolbox", "Large"),
            ("TV lifting: taking out of the box", "Large"),
            ("TV lifting: putting on the wall", "Large"),
            ("TV lifting: taking off the wall", "Large"),
            ("TV lifting: putting in the box", "Large"),
            ("carry screen", "Large"),
            ("screw: by screw driver", "Precise"),
            ("screw: by electric drill", "Precise"),
            ("screw: by hand", "Precise"),
            ("placing items", "Precise"),
            ("unpack tools", "Precise"),
            ("close TV box", "Precise"),
        ]

        activity_labels_to_remove = [
            "synchronisation",
            "Large",
        ]
    elif n_class == 2:
        activity_label_replacements = [
            ("walking on the floor", "Large"),
            ("carry tv spacer", "Large"),
            ("carry tools", "Large"),
            ("carry toolbox", "Large"),
            ("TV lifting: taking out of the box", "Large"),
            ("TV lifting: putting on the wall", "Large"),
            ("TV lifting: taking off the wall", "Large"),
            ("TV lifting: putting in the box", "Large"),
            ("carry screen", "Large"),
            ("screw: by screw driver", "Precise"),
            ("screw: by electric drill", "Precise"),
            ("screw: by hand", "Precise"),
            ("placing items", "Precise"),
            ("unpack tools", "Precise"),
            ("close TV box", "Precise"),
        ]

        activity_labels_to_remove = [
            "synchronisation",
        ]
    else:
        activity_label_replacements = [
            ("walking on the floor", "Walk"),
            ("carry tv spacer", "Walk"),
            ("carry tools", "Walk"),
            ("carry toolbox", "Walk"),
            ("TV lifting: taking out of the box", "Screen placement"),
            ("TV lifting: putting on the wall", "Screen placement"),
            ("TV lifting: taking off the wall", "Screen placement"),
            ("TV lifting: putting in the box", "Screen placement"),
            ("close TV box", "Screen placement"),
            ("carry screen", "Carry"),
            ("screw: by screw driver", "Screwdriver"),
            ("screw: by electric drill", "Drill"),
            ("screw: by hand", "Adjust"),
            ("placing items", "Adjust"),
            ("unpack tools", "Adjust"),
        ]

        activity_labels_to_remove = [
            "synchronisation",
        ]
    return activity_label_replacements, activity_labels_to_remove


def get_label_events_for_person(all_labels, person_id):
    return all_labels.loc[all_labels["subject"] == person_id]


def get_label_for_person_at_time(all_labels, person_id, time):
    persons_labels = get_label_events_for_person(all_labels, person_id)
    tmp1 = persons_labels.loc[persons_labels['start'] <= time]
    tmp1 = tmp1.loc[tmp1["end"] >= time]
    if tmp1["label"].size > 0:
        return tmp1.iloc[0]["label"]
    else:
        return "no activity"


def generate_temporary_feature_files_with_labels(exp_root, persons, window_sizes, number_of_classes):
    exp_name = os.path.basename(exp_root)
    tmp_root = os.path.join("tmp_eyetracker_test", exp_name)
    os.makedirs(tmp_root, exist_ok=True)

    activity_label_replacements, activity_labels_to_remove = get_class_label_attr(number_of_classes)
    for p in persons:
        # Eyetracker feature files:
        for w in window_sizes:
            data = read_eyetracker_feature_file(exp_root, p, w)
            labels = read_activity_labels_from_eyetracker_labelling(exp_root, "video", activity_label_replacements,
                                                                    activity_labels_to_remove)
            labels.drop_duplicates(inplace=True)

            for index, row in data.iterrows():
                current_label = get_label_for_person_at_time(labels, p, row["t_mid"])
                data.loc[index, "label"] = current_label

            new_filename = p + "_eye_features_SW-" + str(w) + "-1000_" + str(number_of_classes) + "-labels.pickle"
            data.to_pickle(os.path.join(tmp_root, new_filename))

        # IMU feature file:
        position = "right"
        data = read_IMU_feature_file(exp_root, p, 5000, position=position)
        labels = read_activity_labels_from_eyetracker_labelling(exp_root, "video", activity_label_replacements,
                                                                activity_labels_to_remove)
        labels.drop_duplicates(inplace=True)

        for index, row in data.iterrows():
            current_label = get_label_for_person_at_time(labels, p, row["t_mid"])
            data.loc[index, "label"] = current_label

        new_filename = p + "_imu_" + position + "_SW-5000-1000_" + str(number_of_classes) + "-labels.pickle"
        data.to_pickle(os.path.join(tmp_root, new_filename))

        # sound feature file:
        data = read_sound_feature_file(exp_root, p, 40)
        labels = read_activity_labels_from_eyetracker_labelling(exp_root, "video", activity_label_replacements,
                                                                activity_labels_to_remove)
        labels.drop_duplicates(inplace=True)

        for index, row in data.iterrows():
            current_label = get_label_for_person_at_time(labels, p, row["t_mid"])
            data.loc[index, "label"] = current_label

        new_filename = p + "_sound_features_SW-40-40_" + str(number_of_classes) + "-labels.pickle"
        data.to_pickle(os.path.join(tmp_root, new_filename))


def convert_frame_by_frame_to_events(times, labels, person):
    events = []
    current_event = None
    for index, label in enumerate(labels):
        if current_event is None:
            # Start new event:
            current_event = {
                "start": times[index],
                "label": label,
                "person_id": person,
                "end": times[index], # to close last event as well
            }
        else:
            if current_event["label"] == label:
                continue
            else:
                current_event["end"] = times[index]
                events.append(current_event)
                current_event = {
                    "start": times[index],
                    "label": label,
                    "person_id": person,
                    "end": times[index],  # to close last event as well
                }
    return pd.DataFrame(events)


def get_features_and_labels_eye(dataframe):
    columms_to_features = ['fixation_gap_mean', 'fixation_gap_std',
                                   'fixation_length_mean', 'fixation_length_std', 'theta_mean', 'theta_std',
                                   'theta_mean_crossings', 'phi_mean', 'phi_std', 'phi_mean_crossings',
                                   'pupil_size_mean', 'pupil_size_std', 'confidence_mean', 'confidence_mean_crossings']

    features = np.vstack((
        [dataframe[col].values for col in columms_to_features]
    )).transpose()

    return dataframe["t_mid"].values, dataframe["label"].values.tolist(), features


def get_features_and_labels_imu(dataframe):
    print(list(dataframe.columns.values))
    columms_to_features = ['ax_count', 'ax_mean', 'ax_std', 'ax_min', 'ax_25%', 'ax_50%', 'ax_75%', 'ax_max', 'ax_skew', 'ax_kurt', 'ax_zc', 'ax_ste', 'ax_freq1', 'ax_Pfreq1', 'ax_freq2', 'ax_Pfreq2', 'ay_count', 'ay_mean', 'ay_std', 'ay_min', 'ay_25%', 'ay_50%', 'ay_75%', 'ay_max', 'ay_skew', 'ay_kurt', 'ay_zc', 'ay_ste', 'ay_freq1', 'ay_Pfreq1', 'ay_freq2', 'ay_Pfreq2', 'az_count', 'az_mean', 'az_std', 'az_min', 'az_25%', 'az_50%', 'az_75%', 'az_max', 'az_skew', 'az_kurt', 'az_zc', 'az_ste', 'az_freq1', 'az_Pfreq1', 'az_freq2', 'az_Pfreq2', 'gx_count', 'gx_mean', 'gx_std', 'gx_min', 'gx_25%', 'gx_50%', 'gx_75%', 'gx_max', 'gx_skew', 'gx_kurt', 'gx_zc', 'gx_ste', 'gx_freq1', 'gx_Pfreq1', 'gx_freq2', 'gx_Pfreq2', 'gy_count', 'gy_mean', 'gy_std', 'gy_min', 'gy_25%', 'gy_50%', 'gy_75%', 'gy_max', 'gy_skew', 'gy_kurt', 'gy_zc', 'gy_ste', 'gy_freq1', 'gy_Pfreq1', 'gy_freq2', 'gy_Pfreq2', 'gz_count', 'gz_mean', 'gz_std', 'gz_min', 'gz_25%', 'gz_50%', 'gz_75%', 'gz_max', 'gz_skew', 'gz_kurt', 'gz_zc', 'gz_ste', 'gz_freq1', 'gz_Pfreq1', 'gz_freq2', 'gz_Pfreq2', 'mx_count', 'mx_mean', 'mx_std', 'mx_min', 'mx_25%', 'mx_50%', 'mx_75%', 'mx_max', 'mx_skew', 'mx_kurt', 'mx_zc', 'mx_ste', 'mx_freq1', 'mx_Pfreq1', 'mx_freq2', 'mx_Pfreq2', 'my_count', 'my_mean', 'my_std', 'my_min', 'my_25%', 'my_50%', 'my_75%', 'my_max', 'my_skew', 'my_kurt', 'my_zc', 'my_ste', 'my_freq1', 'my_Pfreq1', 'my_freq2', 'my_Pfreq2', 'mz_count', 'mz_mean', 'mz_std', 'mz_min', 'mz_25%', 'mz_50%', 'mz_75%', 'mz_max', 'mz_skew', 'mz_kurt', 'mz_zc', 'mz_ste', 'mz_freq1', 'mz_Pfreq1', 'mz_freq2', 'mz_Pfreq2', 'roll_count', 'roll_mean', 'roll_std', 'roll_min', 'roll_25%', 'roll_50%', 'roll_75%', 'roll_max', 'roll_skew', 'roll_kurt', 'roll_zc', 'roll_ste', 'roll_freq1', 'roll_Pfreq1', 'roll_freq2', 'roll_Pfreq2', 'pitch_count', 'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_25%', 'pitch_50%', 'pitch_75%', 'pitch_max', 'pitch_skew', 'pitch_kurt', 'pitch_zc', 'pitch_ste', 'pitch_freq1', 'pitch_Pfreq1', 'pitch_freq2', 'pitch_Pfreq2', 'yaw_count', 'yaw_mean', 'yaw_std', 'yaw_min', 'yaw_25%', 'yaw_50%', 'yaw_75%', 'yaw_max', 'yaw_skew', 'yaw_kurt', 'yaw_zc', 'yaw_ste', 'yaw_freq1', 'yaw_Pfreq1', 'yaw_freq2', 'yaw_Pfreq2', 'qx_count', 'qx_mean', 'qx_std', 'qx_min', 'qx_25%', 'qx_50%', 'qx_75%', 'qx_max', 'qx_skew', 'qx_kurt', 'qx_zc', 'qx_ste', 'qx_freq1', 'qx_Pfreq1', 'qx_freq2', 'qx_Pfreq2', 'qy_count', 'qy_mean', 'qy_std', 'qy_min', 'qy_25%', 'qy_50%', 'qy_75%', 'qy_max', 'qy_skew', 'qy_kurt', 'qy_zc', 'qy_ste', 'qy_freq1', 'qy_Pfreq1', 'qy_freq2', 'qy_Pfreq2', 'qz_count', 'qz_mean', 'qz_std', 'qz_min', 'qz_25%', 'qz_50%', 'qz_75%', 'qz_max', 'qz_skew', 'qz_kurt', 'qz_zc', 'qz_ste', 'qz_freq1', 'qz_Pfreq1', 'qz_freq2', 'qz_Pfreq2', 'qw_count', 'qw_mean', 'qw_std', 'qw_min', 'qw_25%', 'qw_50%', 'qw_75%', 'qw_max', 'qw_skew', 'qw_kurt', 'qw_zc', 'qw_ste', 'qw_freq1', 'qw_Pfreq1', 'qw_freq2', 'qw_Pfreq2', 'a_count', 'a_mean', 'a_std', 'a_min', 'a_25%', 'a_50%', 'a_75%', 'a_max', 'a_skew', 'a_kurt', 'a_zc', 'a_ste', 'a_freq1', 'a_Pfreq1', 'a_freq2', 'a_Pfreq2', 'g_count', 'g_mean', 'g_std', 'g_min', 'g_25%', 'g_50%', 'g_75%', 'g_max', 'g_skew', 'g_kurt', 'g_zc', 'g_ste', 'g_freq1', 'g_Pfreq1', 'g_freq2', 'g_Pfreq2', 'm_count', 'm_mean', 'm_std', 'm_min', 'm_25%', 'm_50%', 'm_75%', 'm_max', 'm_skew', 'm_kurt', 'm_zc', 'm_ste', 'm_freq1', 'm_Pfreq1', 'm_freq2', 'm_Pfreq2', 'hpy_count', 'hpy_mean', 'hpy_std', 'hpy_min', 'hpy_25%', 'hpy_50%', 'hpy_75%', 'hpy_max', 'hpy_skew', 'hpy_kurt', 'hpy_zc', 'hpy_ste', 'hpy_freq1', 'hpy_Pfreq1', 'hpy_freq2', 'hpy_Pfreq2']

    features = np.vstack((
        [dataframe[col].values for col in columms_to_features]
    )).transpose()

    return dataframe["t_mid"].values, dataframe["label"].values.tolist(), features


def get_features_and_labels_sound(dataframe):
    print(list(dataframe.columns.values))
    columms_to_features = ['h_mean', 'h_std', 'h_max', 'h_zc', 'h_ste', 'h_freq1', 'h_Pfreq1', 'h_freq2', 'h_Pfreq2', 'w_mean', 'w_std', 'w_max', 'w_zc', 'w_ste', 'w_freq1', 'w_Pfreq1', 'w_freq2', 'w_Pfreq2']

    features = np.vstack((
        [dataframe[col].values for col in columms_to_features]
    )).transpose()

    return dataframe["t_mid"].values, dataframe["label"].values.tolist(), features


def train_multiclass_classifier(exp_root_1,  exp_root_2, window_size, number_of_classes):
    p1_features_1 = read_eyetracker_feature_file(exp_root_1, "P1", window_size, str(number_of_classes)).dropna()
    p1_features_2 = read_eyetracker_feature_file(exp_root_2, "P1", window_size, str(number_of_classes)).dropna()

    p2_features_1 = read_eyetracker_feature_file(exp_root_1, "P2", window_size, str(number_of_classes)).dropna()
    p2_features_2 = read_eyetracker_feature_file(exp_root_2, "P2", window_size, str(number_of_classes)).dropna()

    p3_features_1 = read_eyetracker_feature_file(exp_root_1, "P3", window_size, str(number_of_classes)).dropna()
    p3_features_2 = read_eyetracker_feature_file(exp_root_2, "P3", window_size, str(number_of_classes)).dropna()

    p4_features_1 = read_eyetracker_feature_file(exp_root_1, "P4", window_size, str(number_of_classes)).dropna()
    p4_features_2 = read_eyetracker_feature_file(exp_root_2, "P4", window_size, str(number_of_classes)).dropna()

    #p1_features_1.drop(p1_features_1.loc[p1_features_1["label"] == "no activity"].index, inplace=True)
    #p2_features_1.drop(p2_features_1.loc[p2_features_1["label"] == "no activity"].index, inplace=True)
    #p3_features_1.drop(p3_features_1.loc[p3_features_1["label"] == "no activity"].index, inplace=True)
    #p4_features_1.drop(p4_features_1.loc[p4_features_1["label"] == "no activity"].index, inplace=True)

    print("P1", p1_features_1.count(), p1_features_2.count())
    print("P2", p2_features_1.count(), p2_features_2.count())
    print("P3", p3_features_1.count(), p3_features_2.count())
    print("P4", p4_features_1.count(), p4_features_2.count())

    print(p1_features_1['label'].unique())
    print(p2_features_1['label'].unique())
    print(p3_features_1['label'].unique())
    print(p4_features_1['label'].unique())
    print(p1_features_2['label'].unique())
    print(p2_features_2['label'].unique())
    print(p3_features_2['label'].unique())
    print(p4_features_2['label'].unique())

    training_dfs = pd.concat([p1_features_2, p2_features_2, p3_features_2, p4_features_2])
    test_dfs = pd.concat([p1_features_1, p2_features_1, p3_features_1, p4_features_1])
    t_train, y_train, X_train = get_features_and_labels_eye(training_dfs)
    t_test, y_test, X_test = get_features_and_labels_eye(test_dfs)

    print(t_train.shape, len(y_train), X_train.shape)
    print(t_test.shape, len(y_test), X_test.shape)
    # classifier = SVC(kernel='linear')
    # classifier = KNeighborsClassifier(5)
    # classifier = DecisionTreeClassifier(random_state=0)
    classifier = GaussianNB()
    classifier = OneVsRestClassifier(GaussianNB())
    classifier = OneVsOneClassifier(GaussianNB())
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    print(y_pred)

    t_test_1, y_test_1, X_test_1 = get_features_and_labels_eye(p1_features_1)
    t_test_2, y_test_2, X_test_2 = get_features_and_labels_eye(p2_features_1)
    t_test_3, y_test_3, X_test_3 = get_features_and_labels_eye(p3_features_1)
    t_test_4, y_test_4, X_test_4 = get_features_and_labels_eye(p4_features_1)
    y_pred_1 = classifier.predict(X_test_1)
    y_pred_2 = classifier.predict(X_test_2)
    y_pred_3 = classifier.predict(X_test_3)
    y_pred_4 = classifier.predict(X_test_4)

    p1_dets = convert_frame_by_frame_to_events(t_test_1, y_pred_1, "P1")
    p2_dets = convert_frame_by_frame_to_events(t_test_2, y_pred_2, "P2")
    p3_dets = convert_frame_by_frame_to_events(t_test_3, y_pred_3, "P3")
    p4_dets = convert_frame_by_frame_to_events(t_test_4, y_pred_4, "P4")

    det_results = pd.concat([p1_dets, p2_dets, p3_dets, p4_dets])

    p1_gt = convert_frame_by_frame_to_events(p1_features_1["t_mid"], p1_features_1["label"], "P1")
    p2_gt = convert_frame_by_frame_to_events(p2_features_1["t_mid"], p2_features_1["label"], "P2")
    p3_gt = convert_frame_by_frame_to_events(p3_features_1["t_mid"], p3_features_1["label"], "P3")
    p4_gt = convert_frame_by_frame_to_events(p4_features_1["t_mid"], p4_features_1["label"], "P4")

    gt_results = pd.concat([p1_gt, p2_gt, p3_gt, p4_gt])

    plot_det_vs_label_events(det_results, gt_results)
    # for i, l in enumerate(y_pred):
    #    print(y_test[i], l)

    # Compute confusion matrix
    activities = [
        "Walk",
        "Carry",
        "Screen placement",
        "Adjust",
        "Screwdriver",
        "Drill"
    ]
    cnf_matrix = confusion_matrix(y_test, y_pred, activities)
    #np.set_printoptions(precision=2)
    print(cnf_matrix)

    acc = accuracy_score(y_test, y_pred)
    print(acc)
    classification_report(y_test, y_pred, activities)


def plot_det_vs_label_events(detections, labels):
    if len(labels["label"].unique()) > 2 or len(detections["label"].unique()) > 2:
        activities = [
            "Walk",
            "Carry",
            "Screen placement",
            "Adjust",
            "Screwdriver",
            "Drill"
        ]
    else:
        activities = ["Precise"]

    persons = labels["person_id"].unique()

    for p in persons:
        plt.figure(figsize=(16, 6))
        ax = plt.gca()
        gt_color = "#1f77b4"
        det_color = "orange"
        height = 0.05
        yticks = activities

        person_labels = labels[labels["person_id"] == p]
        person_detections = detections[detections["person_id"] == p]

        for a_i, a in enumerate(activities):
            label_events = person_labels[person_labels["label"] == a]
            det_events = person_detections[person_detections["label"] == a]

            for i, event in label_events.iterrows():
                y_pos = (yticks.index(event["label"]) + 1) * (1 / (len(activities) + 1))
                plt.axvspan(event["start"], max(event["end"], event["start"] + 1), y_pos - height, y_pos, color=gt_color)

            for i, event in det_events.iterrows():
                y_pos = (yticks.index(event["label"]) + 1) * (1 / (len(activities) + 1))
                plt.axvspan(event["start"], max(event["end"], event["start"] + 1), y_pos, y_pos + height, color=det_color)


        plt.title("Activity recognitions for " + str(p), loc="left")
        plt.grid()
        plt.ylim([0, len(yticks) + 1])
        plt.yticks(range(1, len(yticks) + 1), yticks)
        plt.tight_layout()
        plt.subplots_adjust(top=0.83)

        box_height = 0.07
        box_y = 1.02

        left_box_x = 0.5
        right_box_x = 0.75
        det_legend = Rectangle((right_box_x, box_y), 0.2, box_height, color=det_color, transform=ax.transAxes,
                               clip_on=False)
        gt_legend = Rectangle((left_box_x, box_y), 0.2, box_height, color=gt_color, transform=ax.transAxes,
                              clip_on=False)

        ax.text(right_box_x + 0.1, box_y + 0.5 * box_height, 'Detections',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10, color='#444433',
                transform=ax.transAxes)

        ax.text(left_box_x + 0.1, box_y + 0.5 * box_height, 'Ground Truth',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10, color='white',
                transform=ax.transAxes)

        ax.add_patch(det_legend)
        ax.add_patch(gt_legend)
        plt.draw()

    plt.show()


def find_best_window_size_for_classes(exp_root_1, exp_root_2, windows, number_of_classes):
    activities = [
        "no activity",
        "Walk",
        "Carry",
        "Screen placement",
        "Adjust",
        "Screwdriver",
        "Drill"
    ]

    if number_of_classes == 1:
        activities = [
            "no activity",
            "Precise",
        ]

    if number_of_classes == 2:
        activities = [
            "no activity",
            "Precise",
            "Large",
        ]

    for window_size in windows:
        p1_features_1 = read_eyetracker_feature_file(exp_root_1, "P1", window_size, str(number_of_classes)).dropna()
        p1_features_2 = read_eyetracker_feature_file(exp_root_2, "P1", window_size, str(number_of_classes)).dropna()

        p2_features_1 = read_eyetracker_feature_file(exp_root_1, "P2", window_size, str(number_of_classes)).dropna()
        p2_features_2 = read_eyetracker_feature_file(exp_root_2, "P2", window_size, str(number_of_classes)).dropna()

        p3_features_1 = read_eyetracker_feature_file(exp_root_1, "P3", window_size, str(number_of_classes)).dropna()
        p3_features_2 = read_eyetracker_feature_file(exp_root_2, "P3", window_size, str(number_of_classes)).dropna()

        p4_features_1 = read_eyetracker_feature_file(exp_root_1, "P4", window_size, str(number_of_classes)).dropna()
        p4_features_2 = read_eyetracker_feature_file(exp_root_2, "P4", window_size, str(number_of_classes)).dropna()

        if False:
            p1_features_1.drop(p1_features_1.loc[p1_features_1["label"] == "no activity"].index, inplace=True)
            p2_features_1.drop(p2_features_1.loc[p2_features_1["label"] == "no activity"].index, inplace=True)
            p3_features_1.drop(p3_features_1.loc[p3_features_1["label"] == "no activity"].index, inplace=True)
            p4_features_1.drop(p4_features_1.loc[p4_features_1["label"] == "no activity"].index, inplace=True)

            p1_features_2.drop(p1_features_2.loc[p1_features_2["label"] == "no activity"].index, inplace=True)
            p2_features_2.drop(p2_features_2.loc[p2_features_2["label"] == "no activity"].index, inplace=True)
            p3_features_2.drop(p3_features_2.loc[p3_features_2["label"] == "no activity"].index, inplace=True)
            p4_features_2.drop(p4_features_2.loc[p4_features_2["label"] == "no activity"].index, inplace=True)


        training_dfs = pd.concat([p1_features_2, p2_features_2, p3_features_2, p4_features_2])
        test_dfs = pd.concat([p1_features_1, p2_features_1, p3_features_1, p4_features_1])
        t_train, y_train, X_train = get_features_and_labels_eye(training_dfs)
        t_test, y_test, X_test = get_features_and_labels_eye(test_dfs)

        classifier = SVC(kernel='linear')
        #classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
        # classifier = KNeighborsClassifier(5)
        # classifier = DecisionTreeClassifier(random_state=0)
        classifier = GaussianNB()
        classifier = OneVsRestClassifier(classifier)
        classifier = OneVsOneClassifier(classifier)
        y_pred = classifier.fit(X_train, y_train).predict(X_test)

        cnf_matrix = confusion_matrix(y_test, y_pred, activities)

        print("\n\n")
        print(window_size)
        # np.set_printoptions(precision=2)
        print(cnf_matrix)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, activities)
        print("accuracy score:", acc)
        print(report)


def test_imu_only(exp_root_1, exp_root_2, number_of_classes):
    activities = [
        "no activity",
        "Walk",
        "Carry",
        "Screen placement",
        "Adjust",
        "Screwdriver",
        "Drill"
    ]

    if number_of_classes == 1:
        activities = [
            "no activity",
            "Precise",
        ]

    if number_of_classes == 2:
        activities = [
            "no activity",
            "Precise",
            "Large",
        ]

    position = "right"
    window_size = 5000
    p1_features_1 = read_IMU_feature_file(exp_root_1, "P1", window_size, labelset=str(number_of_classes), position=position).dropna()
    p1_features_2 = read_IMU_feature_file(exp_root_2, "P1", window_size, labelset=str(number_of_classes), position=position).dropna()

    p2_features_1 = read_IMU_feature_file(exp_root_1, "P2", window_size, labelset=str(number_of_classes), position=position).dropna()
    p2_features_2 = read_IMU_feature_file(exp_root_2, "P2", window_size, labelset=str(number_of_classes), position=position).dropna()

    p3_features_1 = read_IMU_feature_file(exp_root_1, "P3", window_size, labelset=str(number_of_classes), position=position).dropna()
    p3_features_2 = read_IMU_feature_file(exp_root_2, "P3", window_size, labelset=str(number_of_classes), position=position).dropna()

    p4_features_1 = read_IMU_feature_file(exp_root_1, "P4", window_size, labelset=str(number_of_classes), position=position).dropna()
    p4_features_2 = read_IMU_feature_file(exp_root_2, "P4", window_size, labelset=str(number_of_classes), position=position).dropna()

    if False:
        p1_features_1.drop(p1_features_1.loc[p1_features_1["label"] == "no activity"].index, inplace=True)
        p2_features_1.drop(p2_features_1.loc[p2_features_1["label"] == "no activity"].index, inplace=True)
        p3_features_1.drop(p3_features_1.loc[p3_features_1["label"] == "no activity"].index, inplace=True)
        p4_features_1.drop(p4_features_1.loc[p4_features_1["label"] == "no activity"].index, inplace=True)

        p1_features_2.drop(p1_features_2.loc[p1_features_2["label"] == "no activity"].index, inplace=True)
        p2_features_2.drop(p2_features_2.loc[p2_features_2["label"] == "no activity"].index, inplace=True)
        p3_features_2.drop(p3_features_2.loc[p3_features_2["label"] == "no activity"].index, inplace=True)
        p4_features_2.drop(p4_features_2.loc[p4_features_2["label"] == "no activity"].index, inplace=True)

    training_dfs = pd.concat([p1_features_2, p2_features_2, p3_features_2, p4_features_2])
    test_dfs = pd.concat([p1_features_1, p2_features_1, p3_features_1, p4_features_1])
    t_train, y_train, X_train = get_features_and_labels_imu(training_dfs)
    t_test, y_test, X_test = get_features_and_labels_imu(test_dfs)

    classifier = SVC(kernel='linear')
    # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
    # classifier = KNeighborsClassifier(5)
    # classifier = DecisionTreeClassifier(random_state=0)
    classifier = GaussianNB()
    classifier = OneVsRestClassifier(classifier)
    classifier = OneVsOneClassifier(classifier)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred, activities)

    print("\n\n")
    print(window_size)
    # np.set_printoptions(precision=2)
    print(cnf_matrix)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, activities)
    print("accuracy score:", acc)
    print(report)


def test_sound_only(exp_root_1, exp_root_2, number_of_classes):
    activities = [
        "no activity",
        "Walk",
        "Carry",
        "Screen placement",
        "Adjust",
        "Screwdriver",
        "Drill"
    ]

    if number_of_classes == 1:
        activities = [
            "no activity",
            "Precise",
        ]

    if number_of_classes == 2:
        activities = [
            "no activity",
            "Precise",
            "Large",
        ]

    step_size = 40
    window_size = 40
    p1_features_1 = read_sound_feature_file(exp_root_1, "P1", window_size, labelset=str(number_of_classes),
                                          step_size=step_size).dropna()
    p1_features_2 = read_sound_feature_file(exp_root_2, "P1", window_size, labelset=str(number_of_classes),
                                          step_size=step_size).dropna()

    p2_features_1 = read_sound_feature_file(exp_root_1, "P2", window_size, labelset=str(number_of_classes),
                                          step_size=step_size).dropna()
    p2_features_2 = read_sound_feature_file(exp_root_2, "P2", window_size, labelset=str(number_of_classes),
                                          step_size=step_size).dropna()

    p3_features_1 = read_sound_feature_file(exp_root_1, "P3", window_size, labelset=str(number_of_classes),
                                          step_size=step_size).dropna()
    p3_features_2 = read_sound_feature_file(exp_root_2, "P3", window_size, labelset=str(number_of_classes),
                                          step_size=step_size).dropna()

    p4_features_1 = read_sound_feature_file(exp_root_1, "P4", window_size, labelset=str(number_of_classes),
                                          step_size=step_size).dropna()
    p4_features_2 = read_sound_feature_file(exp_root_2, "P4", window_size, labelset=str(number_of_classes),
                                          step_size=step_size).dropna()

    if False:
        p1_features_1.drop(p1_features_1.loc[p1_features_1["label"] == "no activity"].index, inplace=True)
        p2_features_1.drop(p2_features_1.loc[p2_features_1["label"] == "no activity"].index, inplace=True)
        p3_features_1.drop(p3_features_1.loc[p3_features_1["label"] == "no activity"].index, inplace=True)
        p4_features_1.drop(p4_features_1.loc[p4_features_1["label"] == "no activity"].index, inplace=True)

        p1_features_2.drop(p1_features_2.loc[p1_features_2["label"] == "no activity"].index, inplace=True)
        p2_features_2.drop(p2_features_2.loc[p2_features_2["label"] == "no activity"].index, inplace=True)
        p3_features_2.drop(p3_features_2.loc[p3_features_2["label"] == "no activity"].index, inplace=True)
        p4_features_2.drop(p4_features_2.loc[p4_features_2["label"] == "no activity"].index, inplace=True)

    training_dfs = pd.concat([p1_features_2, p2_features_2, p3_features_2, p4_features_2])
    test_dfs = pd.concat([p1_features_1, p2_features_1, p3_features_1, p4_features_1])
    t_train, y_train, X_train = get_features_and_labels_sound(training_dfs)
    t_test, y_test, X_test = get_features_and_labels_sound(test_dfs)

    classifier = SVC(kernel='linear')
    # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
    # classifier = KNeighborsClassifier(5)
    # classifier = DecisionTreeClassifier(random_state=0)
    classifier = GaussianNB()
    classifier = OneVsRestClassifier(classifier)
    classifier = OneVsOneClassifier(classifier)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred, activities)

    print("\n\n")
    print(window_size)
    # np.set_printoptions(precision=2)
    print(cnf_matrix)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, activities)
    print("accuracy score:", acc)
    print(report)


def run_labelled_feature_file_gen():
    experiment_1 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    experiment_2 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_9"
    persons = ["P1", "P2", "P3", "P4"]
    windows = [3000, 5000, 10000, 15000, 20000, 30000, 45000, 60000]

    generate_temporary_feature_files_with_labels(experiment_1, persons, windows, 6)
    generate_temporary_feature_files_with_labels(experiment_1, persons, windows, 2)
    generate_temporary_feature_files_with_labels(experiment_1, persons, windows, 1)
    generate_temporary_feature_files_with_labels(experiment_2, persons, windows, 6)
    generate_temporary_feature_files_with_labels(experiment_2, persons, windows, 2)
    generate_temporary_feature_files_with_labels(experiment_2, persons, windows, 1)


if __name__ == '__main__':
    # Regenerate features files with label sets:
    if False:
        run_labelled_feature_file_gen()


    experiment_1 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    experiment_2 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_9"


    windows = [3000, 5000, 10000, 15000, 20000, 30000, 45000, 60000]
    number_of_classes = 6 # or 1

    #train_multiclass_classifier(experiment_1, experiment_2, 15000, number_of_classes)
    # find_best_window_size_for_classes(experiment_1, experiment_2, windows, number_of_classes)

    # test_imu_only(experiment_1, experiment_2, number_of_classes)
    test_sound_only(experiment_1, experiment_2, number_of_classes)

