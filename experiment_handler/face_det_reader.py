import os
import pandas as pd
from experiment_handler.finder import find_face_detection_files
from experiment_handler.time_synchronisation import convert_timestamps


def _load_single_face_detection_file(filename):
    if not os.path.exists(filename):
        return None

    data = pd.read_pickle(filename)
    return data


def get_face_detection_data(experiment_path, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read face detections from each subject's eyetracker in a time interval and convert the timestamps if necessary

    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
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
        parsed_data: pandas Dataframe
            contains columns for:
                person_id: on which person's eyetracker image was the face detected
                timestamp: in the given reference time (use as index)
                filename: path to the extracted face image
                position: face's position in the world frame of the eyetracker
                size: size of the face (width, height, area) in pixel
                face_features: face's feature representation calculated by the neural network
    """
    face_detection_files = find_face_detection_files(experiment_path)

    data = []
    for f in face_detection_files:
        fd = _load_single_face_detection_file(f)

        person_id = fd.person_id.iloc[0]
        et_reference = person_id + "_eyetracker"

        # Convert start and end time
        if start is not None:
            start_timestamp = convert_timestamps(experiment_path, start, reference_time, et_reference)
            fd = fd.loc[fd.timestamp >= start_timestamp, :]

        if end is not None:
            end_timestamp = convert_timestamps(experiment_path, end, reference_time, et_reference)
            fd = fd.loc[fd.timestamp <= end_timestamp, :]

        if convert_time:
            fd.timestamp = convert_timestamps(experiment_path, fd.timestamp, et_reference, reference_time)

        data.append(fd)

    processed_data = pd.concat(data)
    return processed_data


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"

    data = get_face_detection_data(exp_root, start=602, end=668, reference_time="video", convert_time=True)
    print(data)