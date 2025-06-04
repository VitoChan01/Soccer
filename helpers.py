import numpy as np
import pandas as pd

# default values for grid dimensions
MAX_X = 1
MAX_Y = 1

def get_field_position(x, y, x_fields = 10, y_fields=10):
    """
    Calculate the position of a field in a grid based on its coordinates.

    Example use on an event dataframe:
    >>> df = pd.DataFrame({'Start X': [0.1, 0.5, 0.9], 'Start Y': [0.2, 0.5, 0.8]})
    >>> df['field_position'] = df.apply(lambda row: get_field_position(row['Start X'], row['Start Y']), axis=1)

    Args:
        x (int): The x-coordinate of the field.
        y (int): The y-coordinate of the field.
        x_fields (int): The number of fields along the x-axis.
        y_fields (int): The number of fields along the y-axis.

    Returns:
        str: A position string in the format "A1", "B2", etc.
    """
    assert x_fields < 26, "x_fields must be less than 26 for single character representation."

    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None

    x_pos = int(x * x_fields)
    y_pos = int(y * y_fields)

    # convert x-pos to string
    x_pos_letter = chr(ord('A') + x_pos)

    # convert y-pos to string
    pos_string = f"{x_pos_letter}{y_pos + 1}"

    return pos_string
