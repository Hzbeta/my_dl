import numpy as np

def np_append(a, b, axis):
    """Appends two arrays together and returns the result.

    If the first array is None, the second array is returned. Otherwise, the
    two arrays are appended together and returned.

    Args:
        a (numpy.ndarray): The first array.
        b (numpy.ndarray): The second array.
        axis (int): The axis along which the arrays are appended.

    Returns:
        numpy.ndarray: The result of appending the two arrays.
    """
    if a is None:
        return b
    else:
        return np.append(a, b, axis=axis)