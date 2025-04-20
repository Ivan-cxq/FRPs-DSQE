import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from math import atan, degrees


def smooth_data(data, degree=5):
    """
    Smooth the data using a UnivariateSpline with the specified degree.

    :param data: DataFrame containing the data
    :param degree: Degree of the spline (default: 5)
    :return: Smoothed DataFrame
    """
    for col in data.columns:
        x = np.arange(len(data[col]))
        spl = UnivariateSpline(x, data[col], k=degree)
        data[col] = spl(x)
    return data


def calculate_line_angle(data_column):
    """
    Calculate the angle of the line formed by the first and last points of the data column.

    :param data_column: A pandas Series representing a column of data
    :return: Angle in degrees
    """
    x1, y1 = 0, data_column.iloc[0]
    x2, y2 = len(data_column) - 1, data_column.iloc[-1]
    return degrees(atan((y2 - y1) / (x2 - x1)))


def detect_mutation_points(data_column, threshold=0.5):
    """
    Detect mutation points in a data column based on a threshold.

    :param data_column: A pandas Series representing a column of data
    :param threshold: Threshold for detecting mutations (default: 0.5)
    :return: List of mutation points
    """
    mutation_points = []
    for i in range(1, len(data_column) - 1):
        if (abs(data_column.iloc[i] - data_column.iloc[i - 1]) > threshold and
                abs(data_column.iloc[i + 1] - data_column.iloc[i]) > threshold):
            mutation_points.append(i)
    return mutation_points


def calculate_mutation_angle(data_column, mutation_range):
    """
    Calculate the angle of the line formed by the start and end of the mutation range.

    :param data_column: A pandas Series representing a column of data
    :param mutation_range: Tuple (start, end) of the mutation range
    :return: Angle in degrees
    """
    start, end = mutation_range
    x1, y1 = start, data_column.iloc[start]
    x2, y2 = end, data_column.iloc[end]
    return degrees(atan((y2 - y1) / (x2 - x1)))


def calculate_angle_difference(angle1, angle2):
    """
    Calculate the smallest angle difference between two angles.

    :param angle1: First angle in degrees
    :param angle2: Second angle in degrees
    :return: Smallest angle difference in degrees
    """
    angle_diff = abs(angle1 - angle2)
    return min(angle_diff, 180 - angle_diff)


def main():
    # Read the CSV file
    data = pd.read_csv('Yarns/Sample_A_weft.csv')
    threshold = 0.5

    # Convert all columns to numeric values
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Smooth the data using a UnivariateSpline
    data = smooth_data(data, degree=5)

    # Process each column
    for col in data.columns:
        # Calculate the overall line angle
        line_angle = calculate_line_angle(data[col])

        # Detect mutation points
        mutation_points = detect_mutation_points(data[col], threshold)

        # Determine the overall mutation range
        overall_mutation_range = None
        if mutation_points:
            overall_mutation_range = (min(mutation_points), max(mutation_points) + 1)

        # Output mutation range and angle
        if overall_mutation_range:
            start, end = overall_mutation_range
            print(f"Column {col}: Overall mutation range from {start} to {end}")

            # Calculate the mutation angle
            mutation_angle = calculate_mutation_angle(data[col], overall_mutation_range)

            # Calculate the angle difference
            angle_diff = calculate_angle_difference(mutation_angle, line_angle)
            print(f"Angle between overall mutation range and line direction: {angle_diff} degrees")


if __name__ == "__main__":
    main()