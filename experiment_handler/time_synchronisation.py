import os
import pandas as pd


def read_synchronisation_file(experiment_root):
    filepath = os.path.join(experiment_root, "labels", "synchronisation.csv")
    return pd.read_csv(filepath)


def convert_timestamps(experiment_root, timestamps, from_reference, to_reference):
    """
    Convert numeric timestamps (seconds for start of the video or posix timestamp) of a reference time (e.g. P3_eyetracker) to a different reference time (e.g. video time)
    
    Parameters
    ----------
    experiment_root: str
        Root of the current experiment (to find the right synchronisation matrix)
    timestamps: float or array like
        timestamps to be converted
    from_reference: str
        name of the reference of the original timestamps
    to_reference: str
        name of the reference time the timestamp has to be converted to

    Returns
    -------
        converted_timestamps: float or array like
            Timestamps given in to_reference time values 
    """
    synchronisation_file = read_synchronisation_file(experiment_root)

    offset = synchronisation_file.loc[synchronisation_file["from"] == from_reference, to_reference].values[0]

    converted_timestamps = timestamps + offset

    return converted_timestamps


if __name__ == '__main__':
    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"

    print(convert_timestamps(exp_root, [1482326641, 1482326642], "P3_eyetracker", "video"))