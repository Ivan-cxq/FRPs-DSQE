import numpy as np
import cv2
import os
import csv
from math import atan2, degrees
from multiprocessing import Pool, cpu_count


def extract_contour_skeleton(args):
    """
    Extract contour skeleton points and calculate the maximum angle deviation from the base line.

    :param args: Tuple containing the image path and segment length
    :return: List of maximum angle deviations for each contour
    """
    img_path, segment_length = args
    img = cv2.imread(img_path, 0)  # Read the image in grayscale
    unique_values = np.unique(img)  # Get unique pixel values in the image

    # Filter unique values within the range [80, 130]
    contour_values = unique_values[(unique_values >= 80) & (unique_values <= 130)]

    # List to store contour points for each value
    contour_points_list = []

    for value in contour_values:
        mask = np.zeros_like(img)  # Create a mask for the current value
        mask[img == value] = 255  # Set pixels with the current value to 255

        contour_points = []  # List to store contour points for the current value

        # Extract contour points by averaging rows for each column
        for col in range(mask.shape[1]):
            rows = np.where(mask[:, col] != 0)[0]  # Find non-zero rows in the current column
            if rows.size > 0:
                avg_row = (rows[0] + rows[-1]) // 2  # Average the top and bottom rows
                contour_points.append((col, avg_row))  # Add the point to the contour list

        if len(contour_points) > 0:
            contour_points_list.append(contour_points)  # Add contour points to the list

    # Calculate the minimum area rectangle for all contour points
    all_x, all_y = [], []
    for contour_points in contour_points_list:
        x, y = zip(*contour_points)
        all_x.extend(x)
        all_y.extend(y)
    rect = cv2.minAreaRect(np.array([(x, y) for x, y in zip(all_x, all_y)], dtype=np.float32))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Find the edge with the smallest angle to the horizontal axis as the base line
    min_angle = 90
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = degrees(atan2(dy, dx))
        if abs(angle) < abs(min_angle):
            min_angle = angle

    # Calculate the maximum angle deviation for each segment of the contour
    max_angles = []
    for contour_points in contour_points_list:
        max_angle = 0
        for i in range(0, len(contour_points), segment_length):
            start_x, start_y = contour_points[i]
            end_x, end_y = contour_points[min(i + segment_length, len(contour_points) - 1)]
            dx = end_x - start_x
            dy = end_y - start_y
            angle = degrees(atan2(dy, dx))
            angle_diff = abs(angle - min_angle)
            max_angle = max(max_angle, angle_diff)
        max_angles.append(max_angle)

    return max_angles


def fit_polynomial(x, y, degree):
    """
    Fit a polynomial curve to the given data points.

    :param x: X-coordinates of the data points
    :param y: Y-coordinates of the data points
    :param degree: Degree of the polynomial
    :return: A polynomial function
    """
    coeffs = np.polyfit(x, y, degree)  # Fit the polynomial
    poly = np.poly1d(coeffs)  # Create a polynomial function
    return poly


def main():
    # Define input and output folders
    image_folder = 'Yarns/Sample_C_warp'
    output_folder = 'Yarns'

    # Get sorted list of file names in the input folder
    file_names = sorted(os.listdir(image_folder))

    # Define the path for the output CSV file
    csv_path = os.path.join(output_folder, f'{os.path.basename(image_folder)}_degree.csv')

    # Get all target gray values
    first_image = file_names[int(len(file_names) / 2)]
    img = cv2.imread(os.path.join(image_folder, first_image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image file: {first_image}")

    gray_values = np.unique(img)[np.unique(img) >= 81]  # Filter gray values >= 81

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header to the CSV file
        header = ['File Name'] + [f'Yarn {i + 1}' for i in range(len(gray_values))]
        writer.writerow(header)

        # Use multiprocessing to process images in parallel
        with Pool(processes=cpu_count()) as pool:
            args = [(os.path.join(image_folder, file_name), 35) for file_name in file_names]
            angles_list = pool.map(extract_contour_skeleton, args)

        # Fit polynomial curves to each column of the data
        num_columns = max(len(angles) for angles in angles_list)
        polyfits = []
        for col in range(num_columns):
            x = np.arange(len(file_names))
            y = [angles_list[i][col] if len(angles_list[i]) > col else 0 for i in range(len(file_names))]
            poly = fit_polynomial(x, y, 11)  # Adjust the degree of the polynomial here
            polyfits.append(poly)

        # Write the polynomial coefficients to the CSV file
        for i, file_name in enumerate(file_names):
            row = [file_name]
            for poly in polyfits:
                row.append(f"{poly(i):.2f}")
            writer.writerow(row)


if __name__ == '__main__':
    main()