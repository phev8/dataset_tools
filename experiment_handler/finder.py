import os, sys


##################
# Eyetracker     #
##################
def get_eyetracker_participant_list(exp_path):
    """
    Find participants in the igroups experiments eyetracker folder and return each as a path
    """
    fd = os.path.join(exp_path, "eyetracker")
    list_of_person_paths = [os.path.join(fd, dir) for dir in os.listdir(fd) if
                            os.path.isdir(os.path.join(fd, dir))]
    return list_of_person_paths


def get_eyetracker_recording_list_in_folder(eyetracker_recording_path):
    """
    Find folders with pupil labs recordings (complete recording, including info.csv, world and eye videos and return each as a path
    """
    folders = [os.path.join(eyetracker_recording_path, dir) for dir in os.listdir(eyetracker_recording_path) if
                            os.path.isdir(os.path.join(eyetracker_recording_path, dir))]

    # Check if it is a complete recording:
    pupil_lab_recording_folder = []
    for f in folders:
        if et_recording_has_info_file(f) and et_recording_has_world_video(f):
            pupil_lab_recording_folder.append(f)

    return pupil_lab_recording_folder


def find_all_eyetracker_recordings_in_experiment(experiment_root):
    """
    Find all eyetracker recordings in the experiment
    :return: list of pathes to the pupil recording folder
    """
    participants = get_eyetracker_participant_list(experiment_root)

    list_of_recordings = []
    for person_path in participants:
        list_of_persons_recordings = get_eyetracker_recording_list_in_folder(person_path)
        list_of_persons_recordings.sort()
        list_of_recordings.extend(list_of_persons_recordings)

    return list_of_recordings


def et_recording_has_info_file(eyetracker_recording_path):
    """
    Check if a single pupil labs recording has an info file
    """
    return os.path.exists(os.path.join(eyetracker_recording_path, "info.csv"))


def et_recording_has_world_video(eyetracker_recording_path):
    """
    Check if a single pupil labs recording has a world video
    """
    files = [os.path.join(eyetracker_recording_path, dir) for dir in os.listdir(eyetracker_recording_path) if
               os.path.isfile(os.path.join(eyetracker_recording_path, dir))]
    for f in files:
        if os.path.basename(f).split(os.path.extsep)[0] == "world":
            return True
    return False


def et_recording_has_world_timestamps(eyetracker_recording_path):
    """
    Check if a single pupil labs recording has world timestamps
    """
    files = [os.path.join(eyetracker_recording_path, dir) for dir in os.listdir(eyetracker_recording_path) if
               os.path.isfile(os.path.join(eyetracker_recording_path, dir))]
    for f in files:
        if os.path.basename(f).split(os.path.extsep)[0] == "world_timestamps":
            return True
    return False


def check_if_eyetracker_recording_complete(eyetracker_recording_path):
    """
    Check if eyetracker recording has all files, we need for the processing

    Arguments:
         eyetracker_recording_path (str): path to the single eyetracker recording
    Returns:
        bool: true if complete false if anything is missing

    Raises:
        FileNotFoundError: if any if the important files are missing
    """
    complete = True

    if not et_recording_has_info_file(eyetracker_recording_path):
        complete = False
        if sys.version_info > (3, 0):
            raise FileNotFoundError("Info file cannot be found in " + eyetracker_recording_path)
        else:
            raise IOError("Info file cannot be found in " + eyetracker_recording_path)
    elif not et_recording_has_world_video(eyetracker_recording_path):
        complete = False
        if sys.version_info > (3, 0):
            raise FileNotFoundError("World video cannot be found in " + eyetracker_recording_path)
        else:
            raise IOError("world_timestamps.npy not found in " + eyetracker_recording_path)
    elif not et_recording_has_world_timestamps(eyetracker_recording_path):
        complete = False
        if sys.version_info > (3, 0):
            raise FileNotFoundError("world_timestamps.npy not found in " + eyetracker_recording_path)
        else:
            raise IOError("world_timestamps.npy not found in " + eyetracker_recording_path)
    return complete


##################
# Face detections#
##################
def find_face_detection_files(exp_root):
    face_det_dir = os.path.join(exp_root, "processed_data", "face_detections")
    fd_files = []
    for root, subdirs, files in os.walk(face_det_dir):
        for dir in subdirs:
            fd_files.append(os.path.join(root, dir, 'face_features.pkl'))
        break
    return fd_files


##################
# IMU            #
##################
def find_all_imu_files(exp_root):
    imu_dir = os.path.join(exp_root, "imu")
    files = [os.path.join(imu_dir, file) for file in os.listdir(imu_dir) if os.path.isfile(os.path.join(imu_dir, file)) and file.split(".")[-1] == "log"]
    return files


##################
# sound            #
##################
def find_all_sound_files(exp_root):
    snd_dir = os.path.join(exp_root, "audio")
    files = [os.path.join(snd_dir, file) for file in os.listdir(snd_dir) if os.path.isfile(os.path.join(snd_dir, file)) and file.split(".")[-1] == "wav"]
    return files



#######################
# Object recognitions #
#######################
def get_object_recognition_infos(exp_root):
    rec_dir = os.path.join(exp_root, "processed_data", "object_recognition_results")
    files = [os.path.join(rec_dir, file) for file in os.listdir(rec_dir) if
             os.path.isfile(os.path.join(rec_dir, file)) and file.split(".")[-1] == "npy"]

    persons = []
    models = []
    training_sets = []
    for f in files:
        filename = os.path.basename(f)
        persons.append(filename.split("_")[0])
        models.append(filename.split("_")[1])
        training_sets.append(filename.split("_")[2])

    persons = set(persons)
    models = set(models)
    training_sets = set(training_sets)
    return files, persons, models, training_sets


def find_object_recognition_files(exp_root, model_name, training_set):
    rec_dir = os.path.join(exp_root, "processed_data", "object_recognition_results")
    files = [os.path.join(rec_dir, file) for file in os.listdir(rec_dir) if
             os.path.isfile(os.path.join(rec_dir, file))
             and file.split(".")[-1] == "npy"
             and file.split("_")[1] == model_name
             and file.split("_")[2] == training_set
             ]

    return files


def find_filtered_object_recognitions(exp_root, model_name):
    rec_dir = os.path.join(exp_root, "processed_data", "object_recognition_results")
    files = [os.path.join(rec_dir, file) for file in os.listdir(rec_dir) if
             os.path.isfile(os.path.join(rec_dir, file)) and file.split("_")[0] == "filtered" and file.split("_")[2] == model_name]

    return files








