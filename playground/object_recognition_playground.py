import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from experiment_handler.object_recognition.object_detection_reader import read_filtered_object_detection_results, read_object_detections
from experiment_handler.label_data_reader import read_experiment_phases, read_location_labels, read_activity_labels_from_eyetracker_labelling


def object_rec_by_label_per_location(exp_root, model):
    phases = read_experiment_phases(exp_root)
    start = phases['assembly'][0]
    end = phases['disassembly'][1]

    object_recognitions = read_filtered_object_detection_results(exp_root, model, start, end, "video")
    loc_labels = read_location_labels(exp_root)
    print(loc_labels)

    infos = []
    for p in loc_labels.keys():
        detections_for_person = object_recognitions.loc[object_recognitions["person_id"] == p]
        for label in loc_labels[p]:
            during_label = detections_for_person.loc[detections_for_person["timestamp"].between(label["start"], label["end"])]

            current = {
                "person_id": p,
                "location": label["location"],
                "duration": label["end"] - label["start"],
                "screwdriver": during_label[during_label["label"] == "screwdriver"].size,
                "power_drill": during_label[during_label["label"] == "power_drill"].size
            }
            infos.append(current)

    infos = pd.DataFrame(infos)
    #print(infos)

    locations = infos["location"].unique()

    for loc in locations:
        at_location = infos.loc[infos["location"] == loc]

        print(loc, at_location["screwdriver"].sum()/at_location["duration"].sum(), at_location["power_drill"].sum()/at_location["duration"].sum(), at_location["duration"].sum())


def object_rec_by_label_per_activity(experiment, model, replacements, remove_labels):
    phases = read_experiment_phases(experiment)
    start = phases['assembly'][0]
    end = phases['disassembly'][1]

    """
    replacements = [
        ("TV lifting: taking out of the box", "no"),
        ("TV lifting: putting on the wall", "no"),
        ("TV lifting: taking off the wall", "no"),
        ("TV lifting: putting in the box", "no"),
        ("walking on the floor", "no"),
        ("carry tv spacer", "no"),
        ("carry tools", "no"),
        ("carry screen", "no"),
        ("screw: by screw driver", "Precise"),
        ("screw: by electric drill", "Precise"),
        ("screw: by hand", "Precise"),
        ("placing items", "Precise"),
        ("unpack tools", "Precise"),
    ]
    """



    activity_labels = read_activity_labels_from_eyetracker_labelling(experiment, "video", replacements, remove_labels)
    activity_labels.drop_duplicates(inplace=True)
    object_recognitions = read_filtered_object_detection_results(experiment, model, start, end, "video")

    object_recognitions.drop(object_recognitions[object_recognitions["top_index"] > 5].index, inplace=True)

    activity_labels["duration"] = activity_labels["end"] - activity_labels["start"]

    activities = activity_labels["label"].unique()
    persons = activity_labels["subject"].unique()

    infos = []
    for p in persons:
        detections_for_person = object_recognitions.loc[object_recognitions["person_id"] == p]

        for index, label in activity_labels.loc[activity_labels["subject"] == p].iterrows():

            during_label = detections_for_person.loc[detections_for_person["timestamp"].between(label["start"], label["end"])]

            current = {
                "person_id": p,
                "activity": label["label"],
                "duration": label["end"] - label["start"],
                "screwdriver": during_label[during_label["label"] == "screwdriver"].size,
                "power_drill": during_label[during_label["label"] == "power_drill"].size
            }
            infos.append(current)

    infos = pd.DataFrame(infos)
    #print(infos)

    for act in activities:
        at_location = infos.loc[infos["activity"] == act]

        print(act, at_location["screwdriver"].sum()/at_location["duration"].sum(), at_location["power_drill"].sum()/at_location["duration"].sum(), at_location["duration"].sum())



    f, axarr = plt.subplots(4, sharex=True, figsize=(16, 10))
    for idx, test_for in enumerate(["P1", "P2", "P3", "P4"]):

        person_dets = object_recognitions.loc[object_recognitions["person_id"] == test_for]
        screwdriver_det_times = person_dets.loc[person_dets["label"] == "screwdriver", "timestamp"].as_matrix()
        screwdriver_y = np.ones(len(screwdriver_det_times))*6.4
        drill_det_times = person_dets.loc[person_dets["label"] == "power_drill", "timestamp"].as_matrix()
        drill_y = np.ones(len(drill_det_times)) * 0.6

        axarr[idx].plot(screwdriver_det_times, screwdriver_y, '|', ms=10, color="red", label="Screwdriver")
        axarr[idx].plot(drill_det_times, drill_y, '|', ms=10, color="olive", label="Power drill")

        height = 0.05
        for index, label in activity_labels.loc[activity_labels["subject"] == test_for].iterrows():
            y_pos = (list(activities).index(label["label"])) / (len(activities)) + 0.08
            axarr[idx].axvspan(label["start"], label["end"], y_pos - height / 2, y_pos + height / 2, color="#1f77b4")

        axarr[idx].grid()
        axarr[idx].legend()
        axarr[idx].set_title(test_for)
        axarr[idx].set_ylabel("Activity")
        axarr[idx].set_yticks(range(1, len(activities) + 1))
        axarr[idx].set_ylim([0.5, 6.5])

        axarr[idx].set_yticklabels(activities)

    plt.xlabel("Time [s]")
    plt.show()


def train_activity_with_object_rec(experiment, model, replacements, to_remove):
    phases = read_experiment_phases(experiment)
    start = phases['assembly'][0]
    end = phases['disassembly'][1]

    activity_labels = read_activity_labels_from_eyetracker_labelling(experiment, "video", replacements, to_remove)
    activity_labels.drop_duplicates(inplace=True)
    object_recognitions = read_object_detections(experiment, model, start, end, "video")

    persons = ["P1", "P2", "P3", "P4"]

    features = {}
    for p in persons:
        feature_matrix = object_recognitions[p]
        #print(feature_matrix[:, 0])
        #print(feature_matrix[:, 1])
        for row in feature_matrix:
            print(row)



    # TODO: create feature matrix and samples

    # TODO: train with three person test on one




if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    model_name = "ResNet50"

    activity_label_replacements = [
        ("walking on the floor", "Walk"),
        ("carry tv spacer", "Walk"),
        ("carry tools", "Walk"),
        ("TV lifting: taking out of the box", "Screen placement"),
        ("TV lifting: putting on the wall", "Screen placement"),
        ("TV lifting: taking off the wall", "Screen placement"),
        ("TV lifting: putting in the box", "Screen placement"),
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

    object_rec_by_label_per_location(exp_root, model_name)
    object_rec_by_label_per_activity(exp_root, model_name, activity_label_replacements, activity_labels_to_remove)
    #train_activity_with_object_rec(exp_root, model_name, activity_label_replacements, activity_labels_to_remove)