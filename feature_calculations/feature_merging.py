import os, sys
import pandas as pd
import numpy as np

if __package__ is None:
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from experiment_handler.label_data_reader import read_experiment_phases


def _get_source_infos(experiment_root, person_id, sources):
    """
    Extract path of the features files and additional useful informations out of the selected feature name.

    Parameters
    ----------
    experiment_root: str
        path to the experiments root folder
    person_id: str
        which participant's data should be processed
    sources: list of str
        feature file names without person prefix and extension

    Returns
    -------
        source_info: dictionary
            Information extracted for feature name
    """

    source_info = {}
    for name in sources:
        source_info[name] = {
            "path_pickle": os.path.join(experiment_root, "processed_data", name.split("_")[0] + "_" + "features",
                                        person_id + "_" + name + ".pickle"),
            "path_csv": os.path.join(experiment_root, "processed_data", name.split("_")[0] + "_" + "features",
                                     person_id + "_" + name + ".csv")
        }
        if name.split("_")[0] == "imu" and name.split("_")[1] == "left":
            source_info[name]["col_prefix"] = "l"
        elif name.split("_")[0] == "imu" and name.split("_")[1] == "right":
            source_info[name]["col_prefix"] = "r"
        elif name.split("_")[0] == "imu" and name.split("_")[1] == "head":
            source_info[name]["col_prefix"] = "h"
        elif name.split("_")[0] == "eye":
            source_info[name]["col_prefix"] = "e"
        elif name.split("_")[0] == "sound":
            source_info[name]["col_prefix"] = "s"
        else:
            raise ValueError("Unknown feature source name.")

    return source_info


def merge_features(experiment_root, person_id, sources, sample_times, max_time_diff=7.0):
    # TODO: add docstring
    # TODO: add argument to choose between window start, mid or end as a reference
    source_and_infos = _get_source_infos(experiment_root, person_id, sources)

    # read feature files
    df_columns = []
    for k in source_and_infos.keys():
        source_and_infos[k]["features"] = pd.read_pickle(source_and_infos[k]["path_pickle"])
        new_names = [(i, source_and_infos[k]["col_prefix"] + "_" + i) for i in
                     source_and_infos[k]["features"].columns.values]
        df_columns.extend( [n[1] for n in new_names if "_t_" not in n[1] and "_label" not in n[1]] )
        source_and_infos[k]["features"].rename(columns=dict(new_names), inplace=True)

    merged_features = pd.DataFrame(index=sample_times, columns=df_columns)
    for sample_time in sample_times:
        samples = []
        for k in source_and_infos.keys():
            index = (source_and_infos[k]["features"][source_and_infos[k]["col_prefix"] + "_t_mid"] - sample_time).abs().idxmin()
            current_row = source_and_infos[k]["features"].iloc[index]
            current_time = current_row[source_and_infos[k]["col_prefix"] + "_t_mid"]
            if abs(current_time - sample_time) > max_time_diff:
                continue

            current_row = current_row.drop( [source_and_infos[k]["col_prefix"] + "_t_start",
                                             source_and_infos[k]["col_prefix"] + "_t_mid",
                                             source_and_infos[k]["col_prefix"] + "_t_end",
                                             source_and_infos[k]["col_prefix"] + "_label"])

            samples.append(current_row)

        if len(samples) <= 0:
            # TODO: what to do if no sample found: skip or write NaNs
            continue
        sample_row = pd.concat(samples, axis=0)
        merged_features.loc[sample_time] = sample_row

    return merged_features


if __name__ == '__main__':
    experiment_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"

    feature_files = ["imu_left_SW-5000-1000_empty-labels",
                     "imu_right_SW-5000-1000_empty-labels",
                     "imu_head_SW-5000-1000_empty-labels",
                     "eye_features_SW-15000-1000_empty-labels"]

    person_id = "P1"

    start = None
    end = None
    step = 1.0

    # read evaluation times from file
    if start is None and end is None:
        # read experiment phases to generate features in that interval
        experiment_phases = read_experiment_phases(experiment_root)
        start = experiment_phases['assembly'][0]
        end = experiment_phases['disassembly'][1]

    # generate sample times between start and end with stepsize
    sample_times = np.arange(start, end, step)

    merged_features = merge_features(experiment_root, person_id, feature_files, sample_times)
    print(merged_features)