import os
import numpy as np
import pandas as pd
import itertools
from experiment_handler.label_data_reader import read_activity_labels_from_eyetracker_labelling, read_experiment_phases
from experiment_handler.object_recognition.object_detection_reader import read_object_detections
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

import wardmetrics
from wardmetrics.core_methods import eval_events, eval_segments
from wardmetrics.utils import *
from wardmetrics.visualisations import *

print("Using ward metrics' version",  wardmetrics.__version__)


default_time_point = 't_mid'
imu_features_to_use = ['a_mean', 'a_std', 'a_zc', 'a_ste', 'a_skew'] # ['ax_count', 'ax_mean', 'ax_std', 'ax_min', 'ax_25%', 'ax_50%', 'ax_75%', 'ax_max', 'ax_skew', 'ax_kurt', 'ax_zc', 'ax_ste', 'ax_freq1', 'ax_Pfreq1', 'ax_freq2', 'ax_Pfreq2', 'ay_count', 'ay_mean', 'ay_std', 'ay_min', 'ay_25%', 'ay_50%', 'ay_75%', 'ay_max', 'ay_skew', 'ay_kurt', 'ay_zc', 'ay_ste', 'ay_freq1', 'ay_Pfreq1', 'ay_freq2', 'ay_Pfreq2', 'az_count', 'az_mean', 'az_std', 'az_min', 'az_25%', 'az_50%', 'az_75%', 'az_max', 'az_skew', 'az_kurt', 'az_zc', 'az_ste', 'az_freq1', 'az_Pfreq1', 'az_freq2', 'az_Pfreq2', 'a_count', 'a_mean', 'a_std', 'a_min', 'a_25%', 'a_50%', 'a_75%', 'a_max', 'a_skew', 'a_kurt', 'a_zc', 'a_ste', 'a_freq1', 'a_Pfreq1', 'a_freq2', 'a_Pfreq2'] # ['ax_count', 'ax_mean', 'ax_std', 'ax_min', 'ax_25%', 'ax_50%', 'ax_75%', 'ax_max', 'ax_skew', 'ax_kurt', 'ax_zc', 'ax_ste', 'ax_freq1', 'ax_Pfreq1', 'ax_freq2', 'ax_Pfreq2', 'ay_count', 'ay_mean', 'ay_std', 'ay_min', 'ay_25%', 'ay_50%', 'ay_75%', 'ay_max', 'ay_skew', 'ay_kurt', 'ay_zc', 'ay_ste', 'ay_freq1', 'ay_Pfreq1', 'ay_freq2', 'ay_Pfreq2', 'az_count', 'az_mean', 'az_std', 'az_min', 'az_25%', 'az_50%', 'az_75%', 'az_max', 'az_skew', 'az_kurt', 'az_zc', 'az_ste', 'az_freq1', 'az_Pfreq1', 'az_freq2', 'az_Pfreq2', 'gx_count', 'gx_mean', 'gx_std', 'gx_min', 'gx_25%', 'gx_50%', 'gx_75%', 'gx_max', 'gx_skew', 'gx_kurt', 'gx_zc', 'gx_ste', 'gx_freq1', 'gx_Pfreq1', 'gx_freq2', 'gx_Pfreq2', 'gy_count', 'gy_mean', 'gy_std', 'gy_min', 'gy_25%', 'gy_50%', 'gy_75%', 'gy_max', 'gy_skew', 'gy_kurt', 'gy_zc', 'gy_ste', 'gy_freq1', 'gy_Pfreq1', 'gy_freq2', 'gy_Pfreq2', 'gz_count', 'gz_mean', 'gz_std', 'gz_min', 'gz_25%', 'gz_50%', 'gz_75%', 'gz_max', 'gz_skew', 'gz_kurt', 'gz_zc', 'gz_ste', 'gz_freq1', 'gz_Pfreq1', 'gz_freq2', 'gz_Pfreq2', 'mx_count', 'mx_mean', 'mx_std', 'mx_min', 'mx_25%', 'mx_50%', 'mx_75%', 'mx_max', 'mx_skew', 'mx_kurt', 'mx_zc', 'mx_ste', 'mx_freq1', 'mx_Pfreq1', 'mx_freq2', 'mx_Pfreq2', 'my_count', 'my_mean', 'my_std', 'my_min', 'my_25%', 'my_50%', 'my_75%', 'my_max', 'my_skew', 'my_kurt', 'my_zc', 'my_ste', 'my_freq1', 'my_Pfreq1', 'my_freq2', 'my_Pfreq2', 'mz_count', 'mz_mean', 'mz_std', 'mz_min', 'mz_25%', 'mz_50%', 'mz_75%', 'mz_max', 'mz_skew', 'mz_kurt', 'mz_zc', 'mz_ste', 'mz_freq1', 'mz_Pfreq1', 'mz_freq2', 'mz_Pfreq2', 'roll_count', 'roll_mean', 'roll_std', 'roll_min', 'roll_25%', 'roll_50%', 'roll_75%', 'roll_max', 'roll_skew', 'roll_kurt', 'roll_zc', 'roll_ste', 'roll_freq1', 'roll_Pfreq1', 'roll_freq2', 'roll_Pfreq2', 'pitch_count', 'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_25%', 'pitch_50%', 'pitch_75%', 'pitch_max', 'pitch_skew', 'pitch_kurt', 'pitch_zc', 'pitch_ste', 'pitch_freq1', 'pitch_Pfreq1', 'pitch_freq2', 'pitch_Pfreq2', 'yaw_count', 'yaw_mean', 'yaw_std', 'yaw_min', 'yaw_25%', 'yaw_50%', 'yaw_75%', 'yaw_max', 'yaw_skew', 'yaw_kurt', 'yaw_zc', 'yaw_ste', 'yaw_freq1', 'yaw_Pfreq1', 'yaw_freq2', 'yaw_Pfreq2', 'qx_count', 'qx_mean', 'qx_std', 'qx_min', 'qx_25%', 'qx_50%', 'qx_75%', 'qx_max', 'qx_skew', 'qx_kurt', 'qx_zc', 'qx_ste', 'qx_freq1', 'qx_Pfreq1', 'qx_freq2', 'qx_Pfreq2', 'qy_count', 'qy_mean', 'qy_std', 'qy_min', 'qy_25%', 'qy_50%', 'qy_75%', 'qy_max', 'qy_skew', 'qy_kurt', 'qy_zc', 'qy_ste', 'qy_freq1', 'qy_Pfreq1', 'qy_freq2', 'qy_Pfreq2', 'qz_count', 'qz_mean', 'qz_std', 'qz_min', 'qz_25%', 'qz_50%', 'qz_75%', 'qz_max', 'qz_skew', 'qz_kurt', 'qz_zc', 'qz_ste', 'qz_freq1', 'qz_Pfreq1', 'qz_freq2', 'qz_Pfreq2', 'qw_count', 'qw_mean', 'qw_std', 'qw_min', 'qw_25%', 'qw_50%', 'qw_75%', 'qw_max', 'qw_skew', 'qw_kurt', 'qw_zc', 'qw_ste', 'qw_freq1', 'qw_Pfreq1', 'qw_freq2', 'qw_Pfreq2', 'a_count', 'a_mean', 'a_std', 'a_min', 'a_25%', 'a_50%', 'a_75%', 'a_max', 'a_skew', 'a_kurt', 'a_zc', 'a_ste', 'a_freq1', 'a_Pfreq1', 'a_freq2', 'a_Pfreq2', 'g_count', 'g_mean', 'g_std', 'g_min', 'g_25%', 'g_50%', 'g_75%', 'g_max', 'g_skew', 'g_kurt', 'g_zc', 'g_ste', 'g_freq1', 'g_Pfreq1', 'g_freq2', 'g_Pfreq2', 'm_count', 'm_mean', 'm_std', 'm_min', 'm_25%', 'm_50%', 'm_75%', 'm_max', 'm_skew', 'm_kurt', 'm_zc', 'm_ste', 'm_freq1', 'm_Pfreq1', 'm_freq2', 'm_Pfreq2', 'hpy_count', 'hpy_mean', 'hpy_std', 'hpy_min', 'hpy_25%', 'hpy_50%', 'hpy_75%', 'hpy_max', 'hpy_skew', 'hpy_kurt', 'hpy_zc', 'hpy_ste', 'hpy_freq1', 'hpy_Pfreq1', 'hpy_freq2', 'hpy_Pfreq2']
sound_features_to_use = ['h_zc', 'h_ste', 'w_ste', 'w_zc']#['h_mean', 'h_std', 'h_max', 'h_zc', 'h_ste', 'h_freq1', 'h_Pfreq1', 'h_freq2', 'h_Pfreq2', 'w_mean', 'w_std', 'w_max', 'w_zc', 'w_ste', 'w_freq1', 'w_Pfreq1', 'w_freq2', 'w_Pfreq2']

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


def read_sound_feature_file(exp_root, person_id, window_size, labelset="empty", step_size=1000):
    fname = person_id + "_sound_features_SW-" + str(window_size) + "-" + str(step_size) + "_" + labelset + "-labels.pickle"
    filepath = os.path.join(exp_root, "processed_data", "sound_features", fname)
    data = pd.read_pickle(filepath)
    return data


def read_merged_feature_file(exp_root, person_id, eye_window, imu_window, sound_window, labelset="6"):
    fname = person_id + "_merged_features_SW-" + str(eye_window) + "-" + str(imu_window) + "-" + str(sound_window) + "_" + labelset + "-labels.pickle"
    filepath = os.path.join(exp_root, "processed_data", "merged_features", fname)
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


def merge_events(events, merge_threshold):
    persons = events["person_id"].unique()
    activities = events["label"].unique()

    new_events = pd.DataFrame()

    for p in persons:
        persons_events = events.loc[events["person_id"] == p]

        for a in activities:
            activitys_events = persons_events.loc[persons_events["label"] == a]
            activitys_events = activitys_events.sort_values(by="start")

            current_event = None
            for index, row in activitys_events.iterrows():
                if current_event is None:
                    current_event = row
                else:
                    if current_event["end"] + merge_threshold > row["start"]:
                        current_event["end"] = row["end"]
                    else:
                        new_events = new_events.append(current_event)
                        current_event = row

    return new_events


def filter_events(events, duration_threshold):
    events["duration"] = events["end"] - events["start"]
    events.drop(events.loc[events["duration"] < duration_threshold].index, inplace=True)
    return events


def get_features_and_labels(dataframe, feature_set):
    if feature_set == "eye":
        return get_features_and_labels_eye(dataframe)
    elif feature_set == "imu":
        return get_features_and_labels_imu(dataframe)
    elif feature_set == "snd":
        return get_features_and_labels_sound(dataframe)
    elif feature_set == "imu+snd":
        return get_features_and_labels_merged(dataframe, use_eye=False)
    elif feature_set == "all":
        return get_features_and_labels_merged(dataframe)


def get_features_and_labels_eye(dataframe):
    columms_to_features = ['fixation_gap_mean', 'fixation_gap_std',
                                   'fixation_length_mean', 'fixation_length_std', 'theta_mean', 'theta_std',
                                   'theta_mean_crossings', 'phi_mean', 'phi_std', 'phi_mean_crossings',
                                   'pupil_size_mean', 'pupil_size_std', 'confidence_mean', 'confidence_mean_crossings']

    features = np.vstack((
        [dataframe[col].values for col in columms_to_features]
    )).transpose()

    return dataframe[default_time_point].values, dataframe["label"].values.tolist(), features


def get_features_and_labels_imu(dataframe):
    columms_to_features = imu_features_to_use # ['ax_count', 'ax_mean', 'ax_std', 'ax_min', 'ax_25%', 'ax_50%', 'ax_75%', 'ax_max', 'ax_skew', 'ax_kurt', 'ax_zc', 'ax_ste', 'ax_freq1', 'ax_Pfreq1', 'ax_freq2', 'ax_Pfreq2', 'ay_count', 'ay_mean', 'ay_std', 'ay_min', 'ay_25%', 'ay_50%', 'ay_75%', 'ay_max', 'ay_skew', 'ay_kurt', 'ay_zc', 'ay_ste', 'ay_freq1', 'ay_Pfreq1', 'ay_freq2', 'ay_Pfreq2', 'az_count', 'az_mean', 'az_std', 'az_min', 'az_25%', 'az_50%', 'az_75%', 'az_max', 'az_skew', 'az_kurt', 'az_zc', 'az_ste', 'az_freq1', 'az_Pfreq1', 'az_freq2', 'az_Pfreq2', 'gx_count', 'gx_mean', 'gx_std', 'gx_min', 'gx_25%', 'gx_50%', 'gx_75%', 'gx_max', 'gx_skew', 'gx_kurt', 'gx_zc', 'gx_ste', 'gx_freq1', 'gx_Pfreq1', 'gx_freq2', 'gx_Pfreq2', 'gy_count', 'gy_mean', 'gy_std', 'gy_min', 'gy_25%', 'gy_50%', 'gy_75%', 'gy_max', 'gy_skew', 'gy_kurt', 'gy_zc', 'gy_ste', 'gy_freq1', 'gy_Pfreq1', 'gy_freq2', 'gy_Pfreq2', 'gz_count', 'gz_mean', 'gz_std', 'gz_min', 'gz_25%', 'gz_50%', 'gz_75%', 'gz_max', 'gz_skew', 'gz_kurt', 'gz_zc', 'gz_ste', 'gz_freq1', 'gz_Pfreq1', 'gz_freq2', 'gz_Pfreq2', 'mx_count', 'mx_mean', 'mx_std', 'mx_min', 'mx_25%', 'mx_50%', 'mx_75%', 'mx_max', 'mx_skew', 'mx_kurt', 'mx_zc', 'mx_ste', 'mx_freq1', 'mx_Pfreq1', 'mx_freq2', 'mx_Pfreq2', 'my_count', 'my_mean', 'my_std', 'my_min', 'my_25%', 'my_50%', 'my_75%', 'my_max', 'my_skew', 'my_kurt', 'my_zc', 'my_ste', 'my_freq1', 'my_Pfreq1', 'my_freq2', 'my_Pfreq2', 'mz_count', 'mz_mean', 'mz_std', 'mz_min', 'mz_25%', 'mz_50%', 'mz_75%', 'mz_max', 'mz_skew', 'mz_kurt', 'mz_zc', 'mz_ste', 'mz_freq1', 'mz_Pfreq1', 'mz_freq2', 'mz_Pfreq2', 'roll_count', 'roll_mean', 'roll_std', 'roll_min', 'roll_25%', 'roll_50%', 'roll_75%', 'roll_max', 'roll_skew', 'roll_kurt', 'roll_zc', 'roll_ste', 'roll_freq1', 'roll_Pfreq1', 'roll_freq2', 'roll_Pfreq2', 'pitch_count', 'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_25%', 'pitch_50%', 'pitch_75%', 'pitch_max', 'pitch_skew', 'pitch_kurt', 'pitch_zc', 'pitch_ste', 'pitch_freq1', 'pitch_Pfreq1', 'pitch_freq2', 'pitch_Pfreq2', 'yaw_count', 'yaw_mean', 'yaw_std', 'yaw_min', 'yaw_25%', 'yaw_50%', 'yaw_75%', 'yaw_max', 'yaw_skew', 'yaw_kurt', 'yaw_zc', 'yaw_ste', 'yaw_freq1', 'yaw_Pfreq1', 'yaw_freq2', 'yaw_Pfreq2', 'qx_count', 'qx_mean', 'qx_std', 'qx_min', 'qx_25%', 'qx_50%', 'qx_75%', 'qx_max', 'qx_skew', 'qx_kurt', 'qx_zc', 'qx_ste', 'qx_freq1', 'qx_Pfreq1', 'qx_freq2', 'qx_Pfreq2', 'qy_count', 'qy_mean', 'qy_std', 'qy_min', 'qy_25%', 'qy_50%', 'qy_75%', 'qy_max', 'qy_skew', 'qy_kurt', 'qy_zc', 'qy_ste', 'qy_freq1', 'qy_Pfreq1', 'qy_freq2', 'qy_Pfreq2', 'qz_count', 'qz_mean', 'qz_std', 'qz_min', 'qz_25%', 'qz_50%', 'qz_75%', 'qz_max', 'qz_skew', 'qz_kurt', 'qz_zc', 'qz_ste', 'qz_freq1', 'qz_Pfreq1', 'qz_freq2', 'qz_Pfreq2', 'qw_count', 'qw_mean', 'qw_std', 'qw_min', 'qw_25%', 'qw_50%', 'qw_75%', 'qw_max', 'qw_skew', 'qw_kurt', 'qw_zc', 'qw_ste', 'qw_freq1', 'qw_Pfreq1', 'qw_freq2', 'qw_Pfreq2', 'a_count', 'a_mean', 'a_std', 'a_min', 'a_25%', 'a_50%', 'a_75%', 'a_max', 'a_skew', 'a_kurt', 'a_zc', 'a_ste', 'a_freq1', 'a_Pfreq1', 'a_freq2', 'a_Pfreq2', 'g_count', 'g_mean', 'g_std', 'g_min', 'g_25%', 'g_50%', 'g_75%', 'g_max', 'g_skew', 'g_kurt', 'g_zc', 'g_ste', 'g_freq1', 'g_Pfreq1', 'g_freq2', 'g_Pfreq2', 'm_count', 'm_mean', 'm_std', 'm_min', 'm_25%', 'm_50%', 'm_75%', 'm_max', 'm_skew', 'm_kurt', 'm_zc', 'm_ste', 'm_freq1', 'm_Pfreq1', 'm_freq2', 'm_Pfreq2', 'hpy_count', 'hpy_mean', 'hpy_std', 'hpy_min', 'hpy_25%', 'hpy_50%', 'hpy_75%', 'hpy_max', 'hpy_skew', 'hpy_kurt', 'hpy_zc', 'hpy_ste', 'hpy_freq1', 'hpy_Pfreq1', 'hpy_freq2', 'hpy_Pfreq2']

    features = np.vstack((
        [dataframe[col].values for col in columms_to_features]
    )).transpose()

    return dataframe[default_time_point].values, dataframe["label"].values.tolist(), features


def get_features_and_labels_sound(dataframe):
    #print(list(dataframe.columns.values))
    columms_to_features = sound_features_to_use # ['h_mean', 'h_std', 'h_max', 'h_zc', 'h_ste', 'h_freq1', 'h_Pfreq1', 'h_freq2', 'h_Pfreq2', 'w_mean', 'w_std', 'w_max', 'w_zc', 'w_ste', 'w_freq1', 'w_Pfreq1', 'w_freq2', 'w_Pfreq2']

    features = np.vstack((
        [dataframe[col].values for col in columms_to_features]
    )).transpose()

    return dataframe[default_time_point].values, dataframe["label"].values.tolist(), features


def get_features_and_labels_merged(dataframe, use_eye=True, use_imu=True, use_sound=True):
    #print(list(dataframe.columns.values))
    eye_features = ['fixation_gap_mean', 'fixation_gap_std', 'fixation_length_mean', 'fixation_length_std', 'theta_mean', 'theta_std', 'theta_mean_crossings', 'phi_mean', 'phi_std', 'phi_mean_crossings', 'pupil_size_mean', 'pupil_size_std', 'confidence_mean', 'confidence_mean_crossings']
    imu_features = imu_features_to_use
    sound_features = sound_features_to_use

    columns_to_features = []
    if use_eye:
        columns_to_features.extend(eye_features)
    if use_imu:
        columns_to_features.extend(imu_features)
    if use_sound:
        columns_to_features.extend(sound_features)

    features = np.vstack((
        [dataframe[col].values for col in columns_to_features]
    )).transpose()

    return dataframe[default_time_point].values, dataframe["label"].values.tolist(), features


def get_features_and_labels_object_rec(dataframe):
    #print(list(dataframe.columns.values))
    columms_to_features = sound_features_to_use # ['h_mean', 'h_std', 'h_max', 'h_zc', 'h_ste', 'h_freq1', 'h_Pfreq1', 'h_freq2', 'h_Pfreq2', 'w_mean', 'w_std', 'w_max', 'w_zc', 'w_ste', 'w_freq1', 'w_Pfreq1', 'w_freq2', 'w_Pfreq2']

    features = np.vstack((
        [dataframe[col].values for col in columms_to_features]
    )).transpose()

    return dataframe[default_time_point].values, dataframe["label"].values.tolist(), features


def test_object_detections(exp_root_1, exp_root_2, window, number_of_classes):
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
            "Large",
            "Precise",
        ]

    phases = read_experiment_phases(exp_root_1)
    start_1 = phases['assembly'][0]
    end_1 = phases['disassembly'][1]

    phases = read_experiment_phases(exp_root_2)
    start_2 = phases['assembly'][0]
    end_2 = phases['disassembly'][1]

    models = ["InceptionV3", "ResNet50", "VGG16", "VGG19", "Xception"]

    results = []
    for model in models:
        object_recognitions_1 = read_object_detections(exp_root_1, model, start_1, end_1, "video")
        object_recognitions_2 = read_object_detections(exp_root_2, model, start_2, end_2, "video")



    # TODO: read object rec results for each category
    # TODO: do inter and intra experiment evaluation
    # TODO: smooth feature matrixes with window

    persons = ["P1", "P2", "P3", "P4"]

    features = {}
    for p in persons:
        feature_matrix = object_recognitions[p]
        # print(feature_matrix[:, 0])
        # print(feature_matrix[:, 1])
        for row in feature_matrix:
            print(row)



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

        experiment_1_features = [p1_features_1, p2_features_1, p3_features_1, p4_features_1]
        experiment_2_features = [p1_features_2, p2_features_2, p3_features_2, p4_features_2]

        cnf_1, acc_1 = in_experiment_eval(experiment_1_features, exp_root_1, activities, "eye")
        cnf_2, acc_2 = in_experiment_eval(experiment_2_features, exp_root_2, activities, "eye")
        cnf_3, acc_3 = inter_experiment_eval(experiment_1_features, experiment_2_features, exp_root_1, activities, "eye", use_ward_metrics_flag=False)
        cnf_4, acc_4 = inter_experiment_eval(experiment_2_features, experiment_1_features, exp_root_2, activities, "eye", use_ward_metrics_flag=False)
        results.append({
            "window_size": window_size,
            "intra_1": acc_1,
            "intra_2": acc_2,
            "inter_1": acc_3,
            "inter_2": acc_4,
        })

    results = pd.DataFrame(results)
    print(results)

    width = 0.2
    x = np.array(list(range(0, len(windows))))

    plt.figure(figsize=(10, 3))
    plt.bar(x - 1 * width, results["inter_1"], width=width, label='independent 1', align='center')
    plt.bar(x + 0 * width, results["inter_2"], width=width, label='independent 2', align='center')
    plt.bar(x + 1 * width, results["intra_1"], width=width, label='dependent 1', align='center')
    plt.bar(x + 2 * width, results["intra_2"], width=width, label='dependent 2', align='center')
    plt.plot(x, results[["inter_1", "inter_2", "inter_1", "inter_2"]].mean(axis=1, numeric_only=True), linestyle=':', marker='o', color="black")
    plt.ylabel("Accuracy score")
    plt.xlabel("Window size in seconds")
    plt.xticks(x, results["window_size"]/1000)
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.show()



def use_ward_metrics(ground_truth, detections, wm_plots=False):
    persons = ground_truth["person_id"].unique()
    activities = ground_truth["label"].unique()

    results = []
    results_sd = []
    for p in persons:
        for activity in activities:
            if activity == "no activity":
                continue

            activity_det_events = detections.loc[(detections["person_id"] == p) & (detections["label"] == activity)]
            activity_gt_events = ground_truth.loc[(ground_truth["person_id"] == p) & (ground_truth["label"] == activity)]

            print("\n---------------------------")
            print("\tResults for " + p + " - " + activity)
            ground_truth_test = [
                (row["start"], row["end"]) for index, row in activity_gt_events.iterrows()
            ]

            detection_test = [
                (row["start"], row["end"]) for index, row in activity_det_events.iterrows()
            ]


            # Run event-based evaluation:
            try:
                gt_event_scores, det_event_scores, detailed_scores, standard_scores = eval_events(ground_truth_test, detection_test)
            except AttributeError:
                print("no enough data to evaluate", activity_gt_events )
                continue

            detailed_scores["person_id"] = p
            detailed_scores["activity"] = activity
            results.append(detailed_scores)
            standard_scores["person_id"] = p
            standard_scores["activity"] = activity
            results_sd.append(standard_scores)
            # print(p, activity)
            #plot_events_with_event_scores(gt_event_scores, det_event_scores, ground_truth_test, detection_test)

            # Print results:
            print_standard_event_metrics(standard_scores)
            print_detailed_event_metrics(detailed_scores)

    results = pd.DataFrame(results)
    results_sd = pd.DataFrame(results_sd)
    #print(results_sd)

    print("Event length weighted values:")
    print("\t\tprecision & recall")
    for a in activities:
        current_results = results_sd.loc[results["activity"] == a]
        avg_results_for_activity = current_results.mean(axis=0)
        print(a + " & " + str(avg_results_for_activity["precision (weighted)"]) + " & " + str(avg_results_for_activity["recall (weighted)"]))

    if wm_plots:
        for p in persons:
            current_results = results.loc[results["person_id"] == p]
            totals = {}
            for col in results.columns.values:
                if col != "activity" and col != "person_id":
                    totals[col] = current_results[col].sum()
            #print(results.columns.values)
            fig = plot_event_analysis_diagram(totals, use_percentage=False, show=False)
            plt.title(p)
            plt.tight_layout()
            plt.draw()

        for a in activities:
            current_results = results.loc[results["activity"] == a]
            totals = {}
            for col in results.columns.values:
                if col != "activity" and col != "person_id":
                    totals[col] = current_results[col].sum()
            #print(results.columns.values)
            fig = plot_event_analysis_diagram(totals, use_percentage=False, show=False)
            plt.title(a)
            plt.tight_layout()
            plt.draw()
        plt.show()


def test_eye_only(exp_root_1,  exp_root_2, window_size, number_of_classes,
                    use_ward_metrics_flag=False, wm_plots=False):
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
            "Large",
            "Precise",
        ]

    p1_features_1 = read_eyetracker_feature_file(exp_root_1, "P1", window_size, str(number_of_classes)).dropna()
    p1_features_2 = read_eyetracker_feature_file(exp_root_2, "P1", window_size, str(number_of_classes)).dropna()

    p2_features_2 = read_eyetracker_feature_file(exp_root_2, "P2", window_size, str(number_of_classes)).dropna()
    p2_features_1 = read_eyetracker_feature_file(exp_root_1, "P2", window_size, str(number_of_classes)).dropna()

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

    experiment_1_features = [p1_features_1, p2_features_1, p3_features_1, p4_features_1]
    experiment_2_features = [p1_features_2, p2_features_2, p3_features_2, p4_features_2]

    cnf_1, acc_1 = in_experiment_eval(experiment_1_features, exp_root_1, activities, "eye")
    cnf_2, acc_2 = in_experiment_eval(experiment_2_features, exp_root_2, activities, "eye")
    cnf_3, acc_3 = inter_experiment_eval(experiment_1_features, experiment_2_features, exp_root_1, activities, "eye", use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)
    cnf_4, acc_4 = inter_experiment_eval(experiment_2_features, experiment_1_features, exp_root_2, activities, "eye", use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)
    print({
        "intra_1": acc_1,
        "intra_2": acc_2,
        "inter_1": acc_3,
        "inter_2": acc_4,
    })

    plot_confusion_matrix(cnf_4, activities, normalize=False)

    # Event analysis
    return
    t_test_1, y_test_1, X_test_1 = get_features_and_labels(p1_features_1, "eye")
    t_test_2, y_test_2, X_test_2 = get_features_and_labels(p2_features_1, "eye")
    t_test_3, y_test_3, X_test_3 = get_features_and_labels(p3_features_1, "eye")
    t_test_4, y_test_4, X_test_4 = get_features_and_labels(p4_features_1, "eye")
    y_pred_1 = classifier.predict(X_test_1)
    y_pred_2 = classifier.predict(X_test_2)
    y_pred_3 = classifier.predict(X_test_3)
    y_pred_4 = classifier.predict(X_test_4)
    print(y_pred_4)

    p1_dets = convert_frame_by_frame_to_events(t_test_1, y_pred_1, "P1")
    p2_dets = convert_frame_by_frame_to_events(t_test_2, y_pred_2, "P2")
    p3_dets = convert_frame_by_frame_to_events(t_test_3, y_pred_3, "P3")
    p4_dets = convert_frame_by_frame_to_events(t_test_4, y_pred_4, "P4")

    det_results = pd.concat([p1_dets, p2_dets, p3_dets, p4_dets])
    print(det_results)

    det_results = merge_events(det_results, 1)
    det_results = filter_events(det_results, 1)

    p1_gt = convert_frame_by_frame_to_events(p1_features_1[default_time_point], p1_features_1["label"], "P1")
    p2_gt = convert_frame_by_frame_to_events(p2_features_1[default_time_point], p2_features_1["label"], "P2")
    p3_gt = convert_frame_by_frame_to_events(p3_features_1[default_time_point], p3_features_1["label"], "P3")
    p4_gt = convert_frame_by_frame_to_events(p4_features_1[default_time_point], p4_features_1["label"], "P4")

    gt_results = pd.concat([p1_gt, p2_gt, p3_gt, p4_gt])

    # use_ward_metrics(gt_results, det_results)
    # plot_det_vs_label_events(det_results, gt_results)
    # for i, l in enumerate(y_pred):
    #    print(y_test[i], l)



    acc = accuracy_score(y_test, y_pred)
    print(acc)
    classification_report(y_test, y_pred, activities)


def plot_confusion_matrix(cm, classes, cm_std=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, show=True, save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(14, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)

    if normalize:
        if cm_std is not None:
            cm_std = cm_std.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm_std is None:
            plt.text(j, i, int(np.round(cm[i, j]*100)), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, str(int(np.round(cm[i, j]*100))) + "$\pm$" + str(int(np.round(cm_std[i, j]*100))), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.draw()


def plot_det_vs_label_events(detections, labels):
    if len(labels["label"].unique()) > 3 or len(detections["label"].unique()) > 3:
        activities = [
            "Walk",
            "Carry",
            "Screen placement",
            "Adjust",
            "Screwdriver",
            "Drill"
        ]
    elif len(labels["label"].unique()) > 2 or len(detections["label"].unique()) > 2:
        activities = [
            "Precise",
            "Large",
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


def in_experiment_eval(experiment_1_features, experiment_name, activities, feature_set):
    preds = []
    labels = []
    for i in range(0, len(experiment_1_features)):
        training_dfs = pd.concat(
            [features for fi, features in enumerate(experiment_1_features) if fi != i]
        )
        test_dfs = experiment_1_features[i]

        t_train, y_train, X_train = get_features_and_labels(training_dfs, feature_set)
        t_test, y_test, X_test = get_features_and_labels(test_dfs, feature_set)


        # classifier = SVC(kernel='linear')
        # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
        # classifier = KNeighborsClassifier(5)
        # classifier = DecisionTreeClassifier(random_state=0)
        classifier = GaussianNB()
        classifier = OneVsRestClassifier(classifier)
        # classifier = OneVsOneClassifier(classifier)
        y_pred = classifier.fit(X_train, y_train).predict(X_test)

        preds.extend(y_pred)
        labels.extend(y_test)

    cnf_matrix = confusion_matrix(labels, preds, activities)

    print("\n\n")
    print("In experiment resuts for " + experiment_name + " with feature set " + feature_set)
    # np.set_printoptions(precision=2)
    print(cnf_matrix)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, activities)
    print("accuracy score:", acc)
    print(report)
    return cnf_matrix, acc


def inter_experiment_eval(experiment_1_features, experiment_2_features, exp_1_name,
                          activities, feature_set, event_merging_threshold=10, event_filter_threshold=1.5,
                          #activities, feature_set, event_merging_threshold=2, event_filter_threshold=0,
                          use_ward_metrics_flag=True,
                          wm_plots=False,
                          to_plot_events=True
                          ):
    # Direction 1:
    training_dfs = pd.concat(experiment_1_features)
    test_dfs = pd.concat(experiment_2_features)
    t_train, y_train, X_train = get_features_and_labels(training_dfs, feature_set)
    t_test, y_test, X_test = get_features_and_labels(test_dfs, feature_set)

    classifier = SVC(kernel='linear')
    # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
    # classifier = KNeighborsClassifier(5)
    # classifier = DecisionTreeClassifier(random_state=0)
    classifier = GaussianNB()
    classifier = OneVsRestClassifier(classifier)
    # classifier = OneVsOneClassifier(classifier)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred, activities)

    print("\n\n")
    print("Inter experiment results trained with " + exp_1_name + " with feature set " + feature_set)
    # np.set_printoptions(precision=2)
    print(cnf_matrix)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, activities)
    print("accuracy score:", acc)
    print(report)

    # event based:
    t_test_1, y_test_1, X_test_1 = get_features_and_labels(experiment_2_features[0], feature_set)
    t_test_2, y_test_2, X_test_2 = get_features_and_labels(experiment_2_features[1], feature_set)
    t_test_3, y_test_3, X_test_3 = get_features_and_labels(experiment_2_features[2], feature_set)
    t_test_4, y_test_4, X_test_4 = get_features_and_labels(experiment_2_features[3], feature_set)
    y_pred_1 = classifier.predict(X_test_1)
    y_pred_2 = classifier.predict(X_test_2)
    y_pred_3 = classifier.predict(X_test_3)
    y_pred_4 = classifier.predict(X_test_4)

    p1_dets = convert_frame_by_frame_to_events(t_test_1, y_pred_1, "P1")
    p2_dets = convert_frame_by_frame_to_events(t_test_2, y_pred_2, "P2")
    p3_dets = convert_frame_by_frame_to_events(t_test_3, y_pred_3, "P3")
    p4_dets = convert_frame_by_frame_to_events(t_test_4, y_pred_4, "P4")

    det_results = pd.concat([p1_dets, p2_dets, p3_dets, p4_dets])

    det_results = merge_events(det_results, event_merging_threshold)
    det_results = filter_events(det_results, event_filter_threshold)

    p1_gt = convert_frame_by_frame_to_events(experiment_2_features[0][default_time_point].values, experiment_2_features[0]["label"], "P1")
    p2_gt = convert_frame_by_frame_to_events(experiment_2_features[1][default_time_point].values, experiment_2_features[1]["label"], "P2")
    p3_gt = convert_frame_by_frame_to_events(experiment_2_features[2][default_time_point].values, experiment_2_features[2]["label"], "P3")
    p4_gt = convert_frame_by_frame_to_events(experiment_2_features[3][default_time_point].values, experiment_2_features[3]["label"], "P4")

    gt_results = pd.concat([p1_gt, p2_gt, p3_gt, p4_gt])

    if use_ward_metrics_flag:
        use_ward_metrics(gt_results, det_results, wm_plots=wm_plots)

    if to_plot_events:
        plot_det_vs_label_events(det_results, gt_results)


    return cnf_matrix, acc


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
            "Large",
            "Precise",
        ]

    results = []
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

        experiment_1_features = [p1_features_1, p2_features_1, p3_features_1, p4_features_1]
        experiment_2_features = [p1_features_2, p2_features_2, p3_features_2, p4_features_2]

        cnf_1, acc_1 = in_experiment_eval(experiment_1_features, exp_root_1, activities, "eye")
        cnf_2, acc_2 = in_experiment_eval(experiment_2_features, exp_root_2, activities, "eye")
        cnf_3, acc_3 = inter_experiment_eval(experiment_1_features, experiment_2_features, exp_root_1, activities, "eye", use_ward_metrics_flag=False)
        cnf_4, acc_4 = inter_experiment_eval(experiment_2_features, experiment_1_features, exp_root_2, activities, "eye", use_ward_metrics_flag=False)
        results.append({
            "window_size": window_size,
            "intra_1": acc_1,
            "intra_2": acc_2,
            "inter_1": acc_3,
            "inter_2": acc_4,
        })

    results = pd.DataFrame(results)
    print(results)

    width = 0.2
    x = np.array(list(range(0, len(windows))))

    plt.figure(figsize=(10, 3))
    plt.bar(x - 1 * width, results["inter_1"], width=width, label='independent 1', align='center')
    plt.bar(x + 0 * width, results["inter_2"], width=width, label='independent 2', align='center')
    plt.bar(x + 1 * width, results["intra_1"], width=width, label='dependent 1', align='center')
    plt.bar(x + 2 * width, results["intra_2"], width=width, label='dependent 2', align='center')
    plt.plot(x, results[["inter_1", "inter_2", "inter_1", "inter_2"]].mean(axis=1, numeric_only=True), linestyle=':', marker='o', color="black")
    plt.ylabel("Accuracy score")
    plt.xlabel("Window size in seconds")
    plt.xticks(x, results["window_size"]/1000)
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.show()


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
            "Large",
            "Precise",
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

    experiment_1_features = [p1_features_1, p2_features_1, p3_features_1, p4_features_1]
    experiment_2_features = [p1_features_2, p2_features_2, p3_features_2, p4_features_2]

    cnf_1, acc_1 = in_experiment_eval(experiment_1_features, exp_root_1, activities, "imu")
    cnf_2, acc_2 = in_experiment_eval(experiment_2_features, exp_root_2, activities, "imu")
    cnf_3, acc_3 = inter_experiment_eval(experiment_1_features, experiment_2_features, exp_root_1, activities, "imu")
    cnf_4, acc_4 = inter_experiment_eval(experiment_2_features, experiment_1_features, exp_root_2, activities, "imu")
    print({
        "intra_1": acc_1,
        "intra_2": acc_2,
        "inter_1": acc_3,
        "inter_2": acc_4,
    })

    plot_confusion_matrix(cnf_3, activities, normalize=False)


def test_sound_only(exp_root_1, exp_root_2, number_of_classes, use_ward_metrics_flag=False):
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
            "Large",
            "Precise",
        ]

    step_size = 1000
    window_size = 1000
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

    experiment_1_features = [p1_features_1, p2_features_1, p3_features_1, p4_features_1]
    experiment_2_features = [p1_features_2, p2_features_2, p3_features_2, p4_features_2]

    cnf_1, acc_1 = in_experiment_eval(experiment_1_features, exp_root_1, activities, "snd")
    cnf_2, acc_2 = in_experiment_eval(experiment_2_features, exp_root_2, activities, "snd")
    cnf_3, acc_3 = inter_experiment_eval(experiment_1_features, experiment_2_features, exp_root_1, activities, "snd", use_ward_metrics_flag=use_ward_metrics_flag)
    cnf_4, acc_4 = inter_experiment_eval(experiment_2_features, experiment_1_features, exp_root_2, activities, "snd", use_ward_metrics_flag=use_ward_metrics_flag)
    print({
        "intra_1": acc_1,
        "intra_2": acc_2,
        "inter_1": acc_3,
        "inter_2": acc_4,
    })


def test_imu_snd_sources(experiment_1, experiment_2, number_of_classes, use_ward_metrics_flag=False, wm_plots=False):
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
            "Large",
            "Precise",
        ]

    step_size = 1000
    w_eye = 10000
    w_imu = 1000
    w_sound = 1000
    p1_features_1 = read_merged_feature_file(experiment_1, "P1", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()
    p1_features_2 = read_merged_feature_file(experiment_2, "P1", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()

    p2_features_1 = read_merged_feature_file(experiment_1, "P2", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()
    p2_features_2 = read_merged_feature_file(experiment_2, "P2", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()

    p3_features_1 = read_merged_feature_file(experiment_1, "P3", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()
    p3_features_2 = read_merged_feature_file(experiment_2, "P3", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()

    p4_features_1 = read_merged_feature_file(experiment_1, "P4", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()
    p4_features_2 = read_merged_feature_file(experiment_2, "P4", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()

    if False:
        p1_features_1.drop(p1_features_1.loc[p1_features_1["label"] == "no activity"].index, inplace=True)
        p2_features_1.drop(p2_features_1.loc[p2_features_1["label"] == "no activity"].index, inplace=True)
        p3_features_1.drop(p3_features_1.loc[p3_features_1["label"] == "no activity"].index, inplace=True)
        p4_features_1.drop(p4_features_1.loc[p4_features_1["label"] == "no activity"].index, inplace=True)

        p1_features_2.drop(p1_features_2.loc[p1_features_2["label"] == "no activity"].index, inplace=True)
        p2_features_2.drop(p2_features_2.loc[p2_features_2["label"] == "no activity"].index, inplace=True)
        p3_features_2.drop(p3_features_2.loc[p3_features_2["label"] == "no activity"].index, inplace=True)
        p4_features_2.drop(p4_features_2.loc[p4_features_2["label"] == "no activity"].index, inplace=True)

    experiment_1_features = [p1_features_1, p2_features_1, p3_features_1, p4_features_1]
    experiment_2_features = [p1_features_2, p2_features_2, p3_features_2, p4_features_2]

    cnf_1, acc_1 = in_experiment_eval(experiment_1_features, experiment_1, activities, "imu+snd")
    cnf_2, acc_2 = in_experiment_eval(experiment_2_features, experiment_2, activities, "imu+snd")
    cnf_3, acc_3 = inter_experiment_eval(experiment_1_features, experiment_2_features, experiment_1, activities, "imu+snd", use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)
    cnf_4, acc_4 = inter_experiment_eval(experiment_2_features, experiment_1_features, experiment_2, activities, "imu+snd", use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)

    print({
        "intra_1": acc_1,
        "intra_2": acc_2,
        "inter_1": acc_3,
        "inter_2": acc_4,
    })


def test_all_sources(experiment_1, experiment_2, number_of_classes, use_ward_metrics_flag=False, wm_plots=False):
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
            "Large",
            "Precise",
        ]

    step_size = 1000
    w_eye = 15000
    w_imu = 5000
    w_sound = 1000
    p1_features_1 = read_merged_feature_file(experiment_1, "P1", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()
    p1_features_2 = read_merged_feature_file(experiment_2, "P1", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()

    p2_features_1 = read_merged_feature_file(experiment_1, "P2", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()
    p2_features_2 = read_merged_feature_file(experiment_2, "P2", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()

    p3_features_1 = read_merged_feature_file(experiment_1, "P3", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()
    p3_features_2 = read_merged_feature_file(experiment_2, "P3", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()

    p4_features_1 = read_merged_feature_file(experiment_1, "P4", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()
    p4_features_2 = read_merged_feature_file(experiment_2, "P4", w_eye, w_imu, w_sound, str(number_of_classes)).dropna()

    if False:
        p1_features_1.drop(p1_features_1.loc[p1_features_1["label"] == "no activity"].index, inplace=True)
        p2_features_1.drop(p2_features_1.loc[p2_features_1["label"] == "no activity"].index, inplace=True)
        p3_features_1.drop(p3_features_1.loc[p3_features_1["label"] == "no activity"].index, inplace=True)
        p4_features_1.drop(p4_features_1.loc[p4_features_1["label"] == "no activity"].index, inplace=True)

        p1_features_2.drop(p1_features_2.loc[p1_features_2["label"] == "no activity"].index, inplace=True)
        p2_features_2.drop(p2_features_2.loc[p2_features_2["label"] == "no activity"].index, inplace=True)
        p3_features_2.drop(p3_features_2.loc[p3_features_2["label"] == "no activity"].index, inplace=True)
        p4_features_2.drop(p4_features_2.loc[p4_features_2["label"] == "no activity"].index, inplace=True)

    experiment_1_features = [p1_features_1, p2_features_1, p3_features_1, p4_features_1]
    experiment_2_features = [p1_features_2, p2_features_2, p3_features_2, p4_features_2]

    cnf_1, acc_1 = in_experiment_eval(experiment_1_features, experiment_1, activities, "all")
    cnf_2, acc_2 = in_experiment_eval(experiment_2_features, experiment_2, activities, "all")
    cnf_3, acc_3 = inter_experiment_eval(experiment_1_features, experiment_2_features, experiment_1, activities, "all", use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)
    cnf_4, acc_4 = inter_experiment_eval(experiment_2_features, experiment_1_features, experiment_2, activities, "all", use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)
    print({
        "intra_1": acc_1,
        "intra_2": acc_2,
        "inter_1": acc_3,
        "inter_2": acc_4,
    })

    plot_confusion_matrix(cnf_4, activities, normalize=False)


def generate_temporary_feature_files_with_labels(exp_root, persons, window_sizes, number_of_classes, imu_windows, sound_windows):
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
                current_label = get_label_for_person_at_time(labels, p, row[default_time_point])
                data.loc[index, "label"] = current_label

            new_filename = p + "_eye_features_SW-" + str(w) + "-1000_" + str(number_of_classes) + "-labels.pickle"
            data.to_pickle(os.path.join(tmp_root, new_filename))

        # IMU feature file:
        position = "right"
        for imu_window in imu_windows:
            data = read_IMU_feature_file(exp_root, p, imu_window, position=position)
            labels = read_activity_labels_from_eyetracker_labelling(exp_root, "video", activity_label_replacements,
                                                                    activity_labels_to_remove)
            labels.drop_duplicates(inplace=True)

            for index, row in data.iterrows():
                current_label = get_label_for_person_at_time(labels, p, row[default_time_point])
                data.loc[index, "label"] = current_label

            new_filename = p + "_imu_" + position + "_SW-" + str(imu_window) + "-1000_" + str(number_of_classes) + "-labels.pickle"
            data.to_pickle(os.path.join(tmp_root, new_filename))

        # sound feature file:
        for sound_window in sound_windows:
            data = read_sound_feature_file(exp_root, p, sound_window)
            labels = read_activity_labels_from_eyetracker_labelling(exp_root, "video", activity_label_replacements,
                                                                    activity_labels_to_remove)
            labels.drop_duplicates(inplace=True)

            for index, row in data.iterrows():
                current_label = get_label_for_person_at_time(labels, p, row[default_time_point])
                data.loc[index, "label"] = current_label

            new_filename = p + "_sound_features_SW-" + str(sound_window) + "-1000_" + str(number_of_classes) + "-labels.pickle"
            data.to_pickle(os.path.join(tmp_root, new_filename))


def merge_feature_files(features):
    synced_features = [
        features[0],
    ]

    for f_i in range(1, len(features)):
        synced_features.append(pd.DataFrame())

    for row_index, row in features[0].iterrows():
        sample_time = row[default_time_point]

        for f_i in range(1, len(features)):
            features_to_merge = features[f_i]

            # Get closest time point
            index = (features_to_merge[default_time_point] - sample_time).abs().idxmin()

            current_row = features_to_merge.loc[index]
            current_row.drop(["t_start", "t_mid", "t_end", "label"], inplace=True)

            synced_features[f_i] = synced_features[f_i].append(current_row, ignore_index=True)

    merged_features = pd.concat(synced_features, axis=1)
    return merged_features


def generate_merged_features(exp_root, persons, eye_windows, imu_windows, sound_windows, labelset):
    print("start generation for " + exp_root + " with labelset: " + str(labelset) )
    out_root = os.path.join(exp_root, "processed_data", "merged_features")
    os.makedirs(out_root, exist_ok=True)

    labelset = str(labelset)
    for p in persons:
        for w in eye_windows:
            for imu_window in imu_windows:
                for sound_window in sound_windows:
                    eye_data = read_eyetracker_feature_file(exp_root, p, w, labelset=labelset)
                    imu_data = read_IMU_feature_file(exp_root, p, imu_window, labelset=labelset)
                    sound_data = read_sound_feature_file(exp_root, p, sound_window, labelset=labelset)

                    merged_data = merge_feature_files([eye_data, imu_data, sound_data])

                    new_filename = p + "_merged_features_SW-" + str(w) + "-" + str(imu_window) + "-" + str(sound_window) + "_" + labelset + "-labels.pickle"
                    merged_data.to_pickle(os.path.join(out_root, new_filename))


def run_labelled_feature_file_gen():
    experiment_1 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    experiment_2 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_9"
    persons = ["P1", "P2", "P3", "P4"]
    windows = [3000, 5000, 10000, 15000, 20000, 30000, 45000, 60000]
    imu_windows = [1000, 3000, 5000]
    sound_windows = [1000]

    generate_temporary_feature_files_with_labels(experiment_1, persons, windows, 6, imu_windows, sound_windows)
    generate_temporary_feature_files_with_labels(experiment_1, persons, windows, 2, imu_windows, sound_windows)
    generate_temporary_feature_files_with_labels(experiment_1, persons, windows, 1, imu_windows, sound_windows)
    generate_temporary_feature_files_with_labels(experiment_2, persons, windows, 6, imu_windows, sound_windows)
    generate_temporary_feature_files_with_labels(experiment_2, persons, windows, 2, imu_windows, sound_windows)
    generate_temporary_feature_files_with_labels(experiment_2, persons, windows, 1, imu_windows, sound_windows)


def run_merged_feature_file_gen():
    experiment_1 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    experiment_2 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_9"
    persons = ["P1", "P2", "P3", "P4"]
    windows = [3000, 5000, 10000, 15000, 20000, 30000, 45000, 60000]
    imu_windows = [1000, 3000, 5000]
    sound_windows = [1000]

    if False:
        generate_merged_features(experiment_1, persons, windows, imu_windows, sound_windows, 6)
        generate_merged_features(experiment_1, persons, windows, imu_windows, sound_windows, 2)
        generate_merged_features(experiment_1, persons, windows, imu_windows, sound_windows, 1)
    else:
        generate_merged_features(experiment_2, persons, windows, imu_windows, sound_windows, 6)
        generate_merged_features(experiment_2, persons, windows, imu_windows, sound_windows, 2)
        generate_merged_features(experiment_2, persons, windows, imu_windows, sound_windows, 1)


if __name__ == '__main__':
    # Regenerate features files with label sets:
    if False:
        run_labelled_feature_file_gen()
        exit()

    if False:
        run_merged_feature_file_gen()
        exit()

    experiment_1 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    experiment_2 = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_9"


    windows = [3000, 5000, 10000, 15000, 20000, 30000, 45000, 60000]
    number_of_classes = 1 # or 1

    use_ward_metrics_flag = True
    wm_plots = True # ward metric plots
    # find_best_window_size_for_classes(experiment_1, experiment_2, windows, number_of_classes)
    test_object_detections(experiment_1, experiment_2, 2000, number_of_classes)


    # test_eye_only(experiment_1, experiment_2, 10000, number_of_classes, use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)
    # test_imu_only(experiment_1, experiment_2, number_of_classes)
    # test_sound_only(experiment_1, experiment_2, number_of_classes)
    # test_imu_snd_sources(experiment_1, experiment_2, number_of_classes, use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)
    # test_all_sources(experiment_1, experiment_2, number_of_classes, use_ward_metrics_flag=use_ward_metrics_flag, wm_plots=wm_plots)
