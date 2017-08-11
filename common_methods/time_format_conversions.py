import time


def convert_to_timestamp_p2(datetime_obj):
    """ Converts a datetime object to posix timestamp keeping microsecond information

    This method is used for python 2 runtime environment. For python 3 we can simply use datetime.timestamp().
    """
    return time.mktime(datetime_obj.timetuple()) + datetime_obj.microsecond / 1e6


def convert_video_time_string_to_seconds(video_time):
    """ Converting video time's string representation to seconds

    Arguments:
        video_time (str): Time for the beginning of a video with a string using "hh:mm:ss.ms" format.

    Examples:
        >>> convert_video_time_string_to_seconds("00:16")
        16
        >>> convert_video_time_string_to_seconds("03:30")
        210

    Returns:
        int: numeric video time in elapsed seconds from the beginning of the video
    """
    parsed_time = None
    items = video_time.split(":")
    if len(items) == 1:
        parsed_time = float(items[0])
    elif len(items) == 2:
        parsed_time = float(items[0])*60 + float(items[1])
    elif len(items) == 3:
        parsed_time = float(items[0])*3600 + float(items[1])*60 + float(items[2])

    return parsed_time


def convert_seconds_to_video_time_string(seconds):
    """ Converting elapsed seconds from the beginning of a video to video time's string representation

        Arguments:
            seconds (int): Time for the beginning of a video.

        Examples:
            >>> convert_seconds_to_video_time_string(16)
            "00:16"
            >>> convert_seconds_to_video_time_string(210)
            "03:30"

        Returns:
            str: string representation of the video time
    """
    minute = int((seconds / 60))
    seconds = seconds % 60
    return "%02d" % minute + ":" + "%02d" % seconds
