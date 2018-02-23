import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from experiment_handler.label_data_reader import read_experiment_phases, read_location_labels, read_activity_labels_from_eyetracker_labelling
from experiment_handler.face_det_reader import get_face_detection_data
from feature_calculations.colocation.common import get_colocation_labels, get_location_of_persons_at_samples

def face_detection_vs_location(exp_root):
    phases = read_experiment_phases(experiment)
    start = phases['assembly'][0]
    end = phases['disassembly'][1]

    face_detections = get_face_detection_data(exp_root, start, end, "video")
    loc_labels = read_location_labels(exp_root)

    infos = []
    for p in loc_labels.keys():
        detections_for_person = face_detections.loc[face_detections["person_id"] == p]
        for label in loc_labels[p]:
            during_label = detections_for_person.loc[
                detections_for_person["timestamp"].between(label["start"], label["end"])]

            current = {
                "person_id": p,
                "location": label["location"],
                "duration": label["end"] - label["start"],
                "face_count": during_label["timestamp"].size,
            }
            infos.append(current)

    infos = pd.DataFrame(infos)

    locations = infos["location"].unique()
    for loc in locations:
        at_location = infos.loc[infos["location"] == loc]

        print(loc, at_location["face_count"].sum()/at_location["duration"].sum(), at_location["duration"].sum())


def face_detection_vs_activity(experiment, replacements, remove_labels):
    phases = read_experiment_phases(experiment)
    start = phases['assembly'][0]
    end = phases['disassembly'][1]

    activity_labels = read_activity_labels_from_eyetracker_labelling(experiment, "video", replacements, remove_labels)
    activity_labels.drop_duplicates(inplace=True)
    face_detections = get_face_detection_data(experiment, start, end, "video")

    activity_labels["duration"] = activity_labels["end"] - activity_labels["start"]

    activities = activity_labels["label"].unique()
    persons = activity_labels["subject"].unique()

    infos = []
    for p in persons:
        detections_for_person = face_detections.loc[face_detections["person_id"] == p]

        for index, label in activity_labels.loc[activity_labels["subject"] == p].iterrows():

            during_label = detections_for_person.loc[detections_for_person["timestamp"].between(label["start"], label["end"])]

            current = {
                "person_id": p,
                "activity": label["label"],
                "duration": label["end"] - label["start"],
                "face_count": during_label["timestamp"].size,
            }
            infos.append(current)

    infos = pd.DataFrame(infos)
    #print(infos)

    for act in activities:
        at_location = infos.loc[infos["activity"] == act]

        print(act, at_location["face_count"].sum()/at_location["duration"].sum(), at_location["duration"].sum())



    f, axarr = plt.subplots(4, sharex=True, figsize=(16, 10))
    for idx, test_for in enumerate(["P1", "P2", "P3", "P4"]):

        person_dets = face_detections.loc[face_detections["person_id"] == test_for]
        face_det_times = person_dets["timestamp"].as_matrix()
        face_det_y = np.ones(len(face_det_times))*6.4

        axarr[idx].plot(face_det_times, face_det_y, '|', ms=10, color="red", label="face detections")

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


def face_detection_vs_colocation(exp_root):
    step = 5
    window_length = 15 # seconds

    phases = read_experiment_phases(experiment)
    start = phases['assembly'][0]
    end = phases['disassembly'][1]

    sample_times = np.arange(start, end, step)

    location_labels = read_location_labels(exp_root)
    locations = get_location_of_persons_at_samples(location_labels, sample_times, exp_root)

    colocation_labels = get_colocation_labels(locations)
    face_detections = get_face_detection_data(experiment, start, end, "video")

    persons = ["P1", "P2", "P3", "P4"]

    results = []

    for p in persons:
        persons_face_detections = face_detections.loc[face_detections["person_id"] == p]
        for t in sample_times:
            faces_in_window = persons_face_detections.loc[
                persons_face_detections["timestamp"].between(t - window_length, t)]

            colocated = 0
            colocated_max = window_length / step + 1

            # check in windows (number of person colocated with)
            for pp in persons:
                if p == pp:
                    continue
                p_vs_pp_col = colocation_labels[p][pp]
                tmp = p_vs_pp_col[p_vs_pp_col[:, 0] >= t - window_length, :]
                tmp = tmp[tmp[:, 0] <= t, :]

                colocated += (np.sum(tmp, axis=0)[1] * 100 / colocated_max)


            facecount = faces_in_window["timestamp"].size
            current_result = {
                "person_id": p,
                "colocated": colocated,
                "face_count": facecount,
                "timestamp": t
            }
            results.append(current_result)

    results = pd.DataFrame(results)

    x_values = results["colocated"].unique()
    print(x_values)

    x_values = [100, 200, 300]
    titles = ["Co-located with one participant", "Co-located with two participants", "Co-located with three participants"]
    y_values = []
    for x in x_values:
        x_range = 10
        tmp = results.loc[results["colocated"].between(x - x_range, x + x_range), "face_count"].as_matrix()
        y_values.append(tmp[tmp < 1000])

    print(x_values)

    f, axarr = plt.subplots(1, 3, sharey=True, figsize=(16, 5))
    n_bin = 30
    for i in range(0, 3):
        axarr[i].hist(y_values[i], bins=n_bin)
        axarr[i].set_title(titles[i])
        axarr[i].grid()
        axarr[i].set_ylabel("Count")

    axarr[1].set_xlabel("Number of faces detected in a window of " + str(window_length) + " seconds")
    plt.show()

    # Create overall plot
    for p in persons:
        plt.figure()
        plt.plot(results.loc[results["person_id"] == p]["colocated"], results.loc[results["person_id"] == p]["face_count"], 'x')
        plt.title(p)
        plt.draw()
    plt.figure(figsize=(8, 3))
    plt.plot(results["colocated"], results["face_count"], 'x')
    plt.ylabel("Number of detected faces")
    plt.xlabel("Person co-located with other participant in window [%]")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    experiment = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"

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

    face_detection_vs_colocation(experiment)
    exit()
    face_detection_vs_location(experiment)
    face_detection_vs_activity(experiment, activity_label_replacements, activity_labels_to_remove)