import numpy as np

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


def mean_crossings(input_array):
    """
    Counts how many times the signal crosses the mean value

    Arguments:
        input_array (numpy array or list): 1d numpy array containing a time series signal

    Returns:
        int: number of mean crossings

    Raises:
        ValueError: if the input array has more than one dimension

    """
    if input_array.ndim > 1:
        raise ValueError("input_array must have only 1 dimension")
    value = np.mean(input_array)
    tmp = np.zeros(input_array.shape)

    tmp[input_array > value] = 1  # get value crossings

    value_crossings = np.diff(tmp)
    number_of_crossings = len(np.nonzero(value_crossings)[0])

    return number_of_crossings