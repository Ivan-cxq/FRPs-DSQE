import numpy as np
import cv2
import os
import csv
from multiprocessing import Pool, cpu_count


def extract_contour_skeleton(img_path):
    """
    Extract the skeleton points and calculate the minimum bounding rectangle height for each contour.

    :param img_path: Path to the image file
    :return: List of minimum bounding rectangle heights for each contour
    """
    img = cv2.imread(img_path, 0)  # Read the image in grayscale
    unique_values = np.unique(img)  # Get unique pixel values in the image

    # Filter unique values within the range [80, 130]
    contour_values = unique_values[(unique_values >= 80) & (unique_values <= 130)]
    skeleton = np.zeros_like(img)  # Initialize an empty skeleton image

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

    # Calculate the minimum bounding rectangle height for each contour
    min_bounding_heights = []
    for contour_points in contour_points_list:
        if len(contour_points) <= 1:  # If there's only one point, treat it as a single skeleton
            min_bounding_heights.append(0)
        else:
            points = np.array(contour_points, dtype=np.float32)  # Convert points to numpy array
            rect = cv2.minAreaRect(points)  # Calculate the minimum area rectangle
            box = cv2.boxPoints(rect)  # Get the rectangle's corner points
            box = np.int0(box)  # Convert points to integers
            min_bounding_height = max(box[:, 1]) - min(box[:, 1])  # Calculate the height of the rectangle
            min_bounding_heights.append(min_bounding_height)

    return min_bounding_heights


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
    image_folder = 'Yarns/Sample_A_warp'
    output_folder = 'Yarns'

    # Get sorted list of file names in the input folder
    file_names = sorted(os.listdir(image_folder))

    # Define the path for the output CSV file
    csv_path = os.path.join(output_folder, f'{os.path.basename(image_folder)}H.csv')

    # Get all target gray values
    first_image = file_names[int(len(file_names) / 2)]
    img = cv2.imread(os.path.join(image_folder, first_image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image file: {first_image}")

    gray_values = np.unique(img)[
        np.unique(img) >= 81]  # Filter warp yarn gray values >= 81,Filter weft yarn gray values >= 181

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header to the CSV file
        header = ['File Name'] + [f'Yarn {i + 1}' for i in range(len(gray_values))]
        writer.writerow(header)

        # Use multiprocessing to process images in parallel
        with Pool(processes=cpu_count()) as pool:
            args = [os.path.join(image_folder, file_name) for file_name in file_names]
            min_heights_list = pool.map(extract_contour_skeleton, args)

        # Fit polynomial curves to each column of the data
        num_columns = max(len(heights) for heights in min_heights_list)
        polyfits = []
        for col in range(num_columns):
            x = np.arange(len(file_names))
            y = [min_heights_list[i][col] if len(min_heights_list[i]) > col else 0 for i in range(len(file_names))]
            poly = fit_polynomial(x, y, 12)  # Adjust the degree of the polynomial here
            polyfits.append(poly)

        # Write the polynomial coefficients to the CSV file
        for i, file_name in enumerate(file_names):
            row = [file_name]
            for poly in polyfits:
                row.append(f"{poly(i):.2f}")
            writer.writerow(row)


if __name__ == '__main__':
    main()
