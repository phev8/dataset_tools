import os
import numpy as np
import pandas as pd

from experiment_handler.finder import get_object_recognition_infos, find_object_recognition_files, find_filtered_object_recognitions
from experiment_handler.time_synchronisation import convert_timestamps
from experiment_handler.object_recognition.class_index_handler import get_top_n_labels


def _load_single_object_recognition_file(filename):
    return pd.read_pickle(filename)


def read_object_detections(experiment_path, model_name, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read object detections from each subject's eyetracker in a time interval and convert the timestamps if necessary

    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    model_name: str
        Results using this models will be read (e.g. ResNet50, InceptionV3 ...)
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)
    convert_time: bool
        If set the returned array will contain timestamp in reference_time's values

    Returns
    -------
        parsed_data: dict
    """
    object_detection_files = find_object_recognition_files(experiment_path, model_name, "imagenet")

    parsed_data = {}
    for f in object_detection_files:
        fd = _load_single_object_recognition_file(f)
        person_id = os.path.basename(f).split('_')[0]

        et_reference = person_id + "_eyetracker"
        print("Reading data from: ", et_reference)

        # Convert start and end time
        if start is not None:
            start_timestamp = convert_timestamps(experiment_path, start, reference_time, et_reference)
            fd.drop(fd.loc[fd["timestamp"] < start_timestamp].index, inplace=True)

        if end is not None:
            end_timestamp = convert_timestamps(experiment_path, end, reference_time, et_reference)
            fd.drop(fd.loc[fd["timestamp"] > end_timestamp].index, inplace=True)

        if convert_time and reference_time is not None:
            fd.timestamp = convert_timestamps(experiment_path, fd.timestamp, et_reference, reference_time)

        parsed_data[person_id] = fd

    return parsed_data


def get_object_detections_by_label(exp_root, model_name, labels, n, save=True):
    """
    Find object detections across each subject's results for the specified labels

    Parameters
    ----------
    exp_root: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    labels: list of str
        Look for these labels in the top n detection, and if found select them
    model_name: str
        Results using this models will be read (e.g. ResNet50, InceptionV3 ...)
    n: int
        Top n classes will be checked for the labels
    save: bool
        if the found detection should be saved as csv file to the experiment folder (default=True)

    Returns
    -------
        results: pandas Dataframe
    """
    object_rec_files, persons, model_names, training_sets = get_object_recognition_infos(exp_root)
    data = read_object_detections(exp_root, model_name)

    detections = []

    for p_id in persons:
        persons_data = data[p_id]
        detections_for_label = []
        for index, row in persons_data.iterrows():
            found_labels = get_top_n_labels(row.predictions, "imagenet", n)

            for l in labels:
                if l in found_labels:
                    top_index = found_labels.index(l)

                    detections_for_label.append({
                        "label": l,
                        "timestamp": row.timestamp,
                        "top_index": top_index,
                        "person_id": p_id
                    })
        detections.append(pd.DataFrame(detections_for_label))

    detections_merged = pd.concat(detections)

    if save:
        output_file = os.path.join(exp_root, "processed_data", "object_recognition_results", "filtered_detections_" + model_name + "_imagenet.csv")
        detections_merged.to_csv(output_file)


def read_filtered_object_detection_results(experiment_path, model_name, start=None, end=None, reference_time=None):
    """
    Read object detections from each subject's eyetracker in a time interval and convert the timestamps if necessary

    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    model_name: str
        Results using this models will be read (e.g. ResNet50, InceptionV3 ...)
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)

    Returns
    -------
        parsed_data: pandas Dataframe
    """
    path_to_filtered_obj_rec = find_filtered_object_recognitions(experiment_path, model_name)[0]
    data = pd.read_csv(path_to_filtered_obj_rec)

    persons = data["person_id"].unique()
    for p in persons:
        et_reference = p + "_eyetracker"
        data.loc[data["person_id"] == p, "timestamp"] = convert_timestamps(experiment_path, data.loc[data["person_id"] == p, "timestamp"], et_reference, reference_time)

    data = data.drop(data[data["timestamp"] < start].index)
    data = data.drop(data[data["timestamp"] > end].index)

    return data


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    object_rec_files, persons, model_names, training_sets = get_object_recognition_infos(exp_root)
    print(object_rec_files)
    print(persons, model_names, training_sets)

    if False:
        labels = ["screwdriver", "power_drill"]

        for m in model_names:
            get_object_detections_by_label(exp_root, m, labels, 10)

    data = read_filtered_object_detection_results(exp_root, "ResNet50", 1000, 2000, "video")
    print(data.timestamp.min())
    print(data.timestamp.max())