import os
import pandas as pd
from scipy.io import wavfile

from experiment_handler.time_synchronisation import convert_timestamps


    


def get_sound_data(experiment_path, source, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read sound data for a given source .wav file (e.g. P3 without the extension) in a time interval. (from subdir /audio)
    
    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    source: str
        Name of the sound file without extension e.g. P3
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
        parsed_data: numpy array
            sound data with columns order:  <timestamp>, <h>, <w>
    """

    wav_filepath = os.path.join(experiment_path, "audio", source + ".wav" )
    [rate, parsed_data] = wavfile.read(wav_filepath)
    print(["loading " + wav_filepath + " at " + str(rate) + " Hz"])

    parsed_data = pd.DataFrame( parsed_data, columns = ['h','w'] )
    # give it an index in seconds
    parsed_data.index = parsed_data.index / rate

    sound_reference_time = source.split("_")[0] + "_sound"
    
    # Convert start and end time (to sound_reference_time)
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, sound_reference_time)
        parsed_data = parsed_data.loc[parsed_data.index >= start_timestamp,:]

    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, sound_reference_time)
        parsed_data = parsed_data.loc[parsed_data.index <= end_timestamp,:]

    if convert_time:
        parsed_data.index = convert_timestamps(experiment_path, parsed_data.index, sound_reference_time, reference_time)


    return parsed_data



if __name__ == '__main__':
    exp_root = "/data/igroups_recordings/experiment_9"

    data = get_sound_data(exp_root, "P3", start=1145, end=1155, reference_time="video", convert_time=True)

#    wavfile.write( 'test.wav', 16000, data.values  )
#    data.plot()



