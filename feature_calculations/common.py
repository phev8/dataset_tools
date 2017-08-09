# TODO: implement methods here, that are used by more modules


def get_values_in_window(raw_values_array, start, end):
    """
    Accessing rows of a numpy array in the interval start-end by the first column (timestamp)

    Parameters
    ----------
    raw_values_array: numpy array
        The first column should contain the timestamps
    start: float
        Beginning of the interval
    end: float
        End of the interval

    Returns
    -------
        window_values: numpy array
            Rows from the input numpy array, where the first column (timestamp) is in the start-end interval
    """
    window_values = raw_values_array[raw_values_array[:, 0] >= start, :]
    window_values = window_values[window_values[:, 0] <= end, :]
    return window_values
