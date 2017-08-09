

def get_windows(start, end, method_name, time_reference="video", source=None):
    """
    Generating windows

    Parameters
    ----------
    start: float
        Start of the first window
    end: float
        End of the last window
    method_name: str
        Unique name of the window generator method
    time_reference: str
        Name of the time reference's channel (e.g. video), start and end values should be given in this reference time
    source: str
        Optional, use this signal to generate windows

    Returns
    -------
        windows: list
            list of windows given by their start, center and end time as a tuple

    """
    windows = None

    method_prefix = method_name.split("-")[0]

    if method_prefix == "SW":
        windows = generate_sliding_windows(start, end, method_name)
    elif method_name == "PEAK-TO-PEAK":
        # TODO
        pass
    # TODO: implement additional window generation methods

    return windows


def generate_sliding_windows(start, end, method_name):
    window_size = float(method_name.split("-")[1])/1000.0
    step_size = float(method_name.split("-")[2])/1000.0

    current_window = (start, start + window_size/2, start + window_size)
    windows = [current_window]

    while current_window[2] < end:
        current_window = (current_window[0] + step_size, current_window[1] + step_size, current_window[2] + step_size)
        windows.append(current_window)

    return windows

