import os
import json
import pandas as pd
from common_methods.time_format_conversions import convert_video_time_string_to_seconds
from experiment_handler.time_synchronisation import convert_timestamps


def read_experiment_phases(experiment_path):
    """
    Reads CSV file with phases of the experiment given in the main video's time.

    Parameters
    ----------
    experiment_path: str
        Path or the experiment's root folder

    Returns
    -------
        experiment phases: dictionary

    """
    label_file_path = os.path.join(experiment_path, "labels", "experiment_phases.csv")

    experiment_phases = {
        "setup/calib": [0, 0],
        "assembly": [0, 0],
        "disassembly": [0, 0]
    }
    with open(label_file_path, "r") as f:
        for line in f.readlines():
            items = line.strip("\n").split(",")
            if len(items) < 3 or items[0] not in experiment_phases.keys():
                continue

            experiment_phases[items[0]] = [
                convert_video_time_string_to_seconds(items[1]),
                convert_video_time_string_to_seconds(items[2])
            ]

        return experiment_phases
    return None


def read_location_labels(experiment_path):
    label_file_path = os.path.join(experiment_path, "labels", "location_labels.json")
    try:
        f = open(label_file_path, 'r')
        labels = json.load(f)

        location_infos = {}

        for label in labels:
            if label['person_id'] not in location_infos.keys():
                location_infos[label['person_id']] = [{"start": label['start'], "end": label['end'], "reference": label['reference'], "location": label['location']}]
            else:
                location_infos[label['person_id']].append({"start": label['start'], "end": label['end'], "reference": label['reference'], "location": label['location']})
        return location_infos

    except FileNotFoundError as e:
        print(e)
        return None


def read_activity_labels_from_eyetracker_labelling(experiment_path, convert_to, replace_labels=[], remove_labels=[]):
    """
    Reading activity labels file in format of the eyetracker labelling program and convert timestamp to selected reference time

    Parameters
    ----------
    experiment_path: str
        Path or the experiment's root folder
    convert_to: str
        Reference time's channel name (e.g. video)
    replace_labels: list of tuples
        (old_label, new_label) - rename all occurences of 'old_label' with 'new_label'
    remove_labels: list of str
        remove each occurence of these items

    Returns
    -------
        labels

    """
    path_to_labels = os.path.join(experiment_path, "labels", "activity_labels_eyetracker.csv")
    labels = pd.DataFrame.from_csv(path_to_labels)

    persons = labels["subject"].unique()
    for p in persons:
        et_reference = p + "_eyetracker"
        labels.loc[labels["subject"] == p, "start"] = convert_timestamps(experiment_path, labels.loc[
            labels["subject"] == p, "start"], et_reference, convert_to)
        labels.loc[labels["subject"] == p, "end"] = convert_timestamps(experiment_path, labels.loc[
            labels["subject"] == p, "end"], et_reference, convert_to)

    for to_replace in replace_labels:
        labels.loc[labels["label"] == to_replace[0], "label"] = to_replace[1]

    for to_remove in remove_labels:
        labels.drop(labels[labels['label'] == to_remove].index, inplace=True)

    return labels


def read_semantic_ground_truth_labels(experiment_path):#
    """
    Read label file used previously for semantic model

    Parameters
    ----------
    experiment_path: str
        Path or the experiment's root folder

    Returns
    -------
        labels:
    """

    label_file_path_1 = os.path.join(experiment_path, "labels", "semantic_ground_truth.csv")
    label_file_path_2 = os.path.join(experiment_path, "labels", "semantic_ground_truth_enc.csv")
    try:
        f_1 = open(label_file_path_1, 'r')
        f_2 = open(label_file_path_2, 'r')

        f_1_lines = f_1.readlines()
        f_2_lines = f_2.readlines()

        labels = []
        for index in range(min(len(f_1_lines), len(f_2_lines))):
            l1 = f_1_lines[index].strip('\n').split('\t')
            l2 = f_2_lines[index].strip('\n').split('\t')

            if l1[0] != l2[0]:
                print("Warning - time mismatch at index", index, l1, l2)

            label_row = {
                "timestamp": convert_video_time_string_to_seconds(l1[0]),
                "P1_activity": l1[1],
                "P1_class": l1[2],
                "P2_activity": l1[3],
                "P2_class": l1[4],
                "P3_activity": l1[5],
                "P3_class": l1[6],
                "P4_activity": l1[7],
                "P4_class": l1[8],
                "P1_activity_code": l2[1],
                "P1_class_code": l2[2],
                "P2_activity_code": l2[3],
                "P2_class_code": l2[4],
                "P3_activity_code": l2[5],
                "P3_class_code": l2[6],
                "P4_activity_code": l2[7],
                "P4_class_code": l2[8]
            }

            labels.append(label_row)

        labels = pd.DataFrame(labels)
        labels.set_index('timestamp', inplace=True)
        print(labels)

        return labels

    except FileNotFoundError as e:
        print(e)
        return None


if __name__ == '__main__':
    phases = read_experiment_phases("/Volumes/DataDrive/igroups_recordings/igroups_experiment_8")
    print(phases)

    locations = read_location_labels("/Volumes/DataDrive/igroups_recordings/igroups_experiment_8")
    print(locations)

    semantic_labels = read_semantic_ground_truth_labels("/Volumes/DataDrive/igroups_recordings/igroups_experiment_7")
