import cv2
import numpy as np
import csv
import os
from concurrent.futures import ThreadPoolExecutor


def calculate_centroid(img, gray_value):
    """
    Calculate the centroid position of the target with the specified gray value along the Y-axis.

    :param img: Input image (grayscale)
    :param gray_value: Target gray value
    :return: Centroid position (Y-axis coordinate)
    """
    # Create a mask to filter pixels with the specified gray value
    mask = (img == gray_value)
    coords = np.argwhere(mask)  # Get coordinates of all pixels in the mask
    if coords.size > 0:  # If target pixels exist
        centroid_y = int(np.mean(coords[:, 0]))  # Calculate the Y-axis centroid
    else:
        centroid_y = 0  # Return 0 if no target pixels are found
    return centroid_y


def process_image(image_path, gray_values):
    """
    Process a single image to calculate the centroid positions of specified gray values.

    :param image_path: Path to the image file
    :param gray_values: List of gray values to calculate centroids for
    :return: A list containing the image name and centroid positions
    """
    try:
        # Read the image and convert it to grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image file: {image_path}")

        # Initialize the result list with the image name
        row = [os.path.basename(image_path)]

        # Calculate the centroid position for each gray value
        for gray_value in gray_values:
            centroid_y = calculate_centroid(img, gray_value)
            row.append(centroid_y)

        return row

    except Exception as e:
        print(f"Error processing image: {image_path} - {e}")
        return None


def main():
    # Set the path to the image folder
    image_folder = './Yarns/Sample_F_weft'

    # Get the folder name
    folder_name = os.path.basename(image_folder)

    # Get all image file names
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                   f.endswith('.png') or f.endswith('.jpg')]

    if not image_files:
        raise FileNotFoundError(f"No image files found in: {image_folder}")

    # Get all target gray values
    first_image = image_files[int(len(image_files)/2)]
    img = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image file: {first_image}")

    gray_values = np.unique(img)[
        np.unique(img) >= 181]  # Filter weft yarn gray values >= 181

    # Create a CSV file
    csv_file = f'./Yarns/{folder_name}.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        header = ['Image Name'] + [f'Yarn {i + 1}' for i in range(len(gray_values))]
        writer.writerow(header)

        # Process images using multithreading
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image_path, gray_values) for image_path in image_files]
            for future in futures:
                result = future.result()
                if result:  # Write the result if processing is successful
                    writer.writerow(result)


if __name__ == "__main__":
    main()
