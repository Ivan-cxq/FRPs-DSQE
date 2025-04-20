import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from math import sqrt


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


def calculate_mutation_distance(data_column, mutation_range):
    """
    Calculate the Euclidean distance between the start and end of the mutation range.

    :param data_column: A pandas Series representing a column of data
    :param mutation_range: Tuple (start, end) of the mutation range
    :return: Euclidean distance
    """
    start, end = mutation_range
    x_dist = end - start  # Horizontal distance
    y_dist = abs(data_column.iloc[end] - data_column.iloc[start])  # Vertical distance
    distance = sqrt(x_dist**2 + y_dist**2)  # Euclidean distance
    return distance


def main():
    # Read the CSV file
    data = pd.read_csv('Yarns/Sample_F_weft.CSV')
    threshold = 0.8

    # Convert all columns to numeric values
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Smooth the data using a UnivariateSpline
    data = smooth_data(data, degree=5)

    # Process each column
    for col in data.columns:
        # Detect mutation points
        mutation_points = detect_mutation_points(data[col], threshold)

        # Determine the overall mutation range
        overall_mutation_range = None
        if mutation_points:
            start = mutation_points[0]
            end = mutation_points[-1] + 1
            overall_mutation_range = (start, end)

        # Output mutation range and distance
        if overall_mutation_range:
            start, end = overall_mutation_range
            print(f"Column {col}: Overall mutation range from {start} to {end}")

            # Calculate the distance between the start and end of the mutation range
            distance = calculate_mutation_distance(data[col], overall_mutation_range)
            print(f"Distance between mutation range: {distance}")


if __name__ == "__main__":
    main()