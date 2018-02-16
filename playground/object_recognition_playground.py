import pandas as pd
from experiment_handler.object_recognition.object_detection_reader import read_filtered_object_detection_results
from experiment_handler.label_data_reader import read_experiment_phases, read_location_labels


def object_rec_by_label_per_location():
    model = "InceptionV3"
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
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
    print(infos)

    locations = infos["location"].unique()

    for loc in locations:
        at_location = infos.loc[infos["location"] == loc]

        print(loc, at_location["screwdriver"].sum()/at_location["duration"].sum(), at_location["power_drill"].sum()/at_location["duration"].sum(), at_location["duration"].sum())


if __name__ == '__main__':
    object_rec_by_label_per_location()