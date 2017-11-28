import os
import json
from common_methods.time_format_conversions import convert_video_time_string_to_seconds

# TODO: read experiment phases

def read_experiment_phases(experiment_path):
    """
    Reads CSV file with phases of the experiment given in the main video's time.

    Parameters
    ----------
    experiment_path: str
        Path or the experiment#s root folder

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


if __name__ == '__main__':
    phases = read_experiment_phases("/Volumes/DataDrive/igroups_recordings/igroups_experiment_8")
    print(phases)

    locations = read_localtion_labels("/Volumes/DataDrive/igroups_recordings/igroups_experiment_8")
    print(locations)
