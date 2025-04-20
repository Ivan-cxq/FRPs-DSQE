import cv2
import numpy as np
import glob

# Configuration parameters
GRAY_VALUE = 80
WINDOW_WIDTH = 768
WINDOW_HEIGHT = 180
LABELING_ENABLED = True  # Enable to display fiber numbers

def update_and_remove_contour(key, best_contour, best_mask, count_list, count_plus_list, count_plus2_list):
    """Update contour information and remove from count lists"""
    targets[key] = best_mask
    cv2.drawContours(output_img, [best_contour], 0, key + GRAY_VALUE, -1)

    # Calculate new centroid
    moments = cv2.moments(best_contour)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        centers[key] = (center_x, center_y)
    else:
        # Handle division by zero if necessary
        pass

    contour_update_counters[key] = 0
    # Remove key from all non-updated lists
    count_list = [x for x in count_list if x != key]
    count_plus_list = [x for x in count_plus_list if x != key]
    count_plus2_list = [x for x in count_plus2_list if x != key]

    return count_list, count_plus_list, count_plus2_list

if __name__ == '__main__':
    input_files = glob.glob('./Tracking/Sample_A_warp/*.png')
    output_directory = './Tracked/'

    target_count = 1
    targets = {}
    centers = {}
    non_updated_count = []
    non_updated_count_plus = []
    non_updated_count_plus2 = []

    start_index = 300
    end_index = 1381

    # Initialize seed image and contour library
    seed_image = cv2.imread(input_files[start_index], cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(seed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_update_counters = {idx + 1: 0 for idx in range(len(contours))}

    for contour in contours:
        base_mask = np.zeros_like(seed_image)
        cv2.drawContours(base_mask, [contour], 0, 255, -1)
        targets[target_count] = base_mask

        # Calculate contour centroid
        moments = cv2.moments(contour)
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        centers[target_count] = (center_x, center_y)

        target_count += 1

    # Main tracking loop
    for frame_index in range(start_index, end_index):
        current_file = input_files[frame_index]
        gray_frame = cv2.imread(current_file, cv2.IMREAD_GRAYSCALE)
        _, threshold = cv2.threshold(gray_frame, 0, 255, 0)
        output_img = np.zeros_like(gray_frame)

        for contour_id, contour_mask in targets.items():
            working_mask = np.zeros_like(gray_frame)
            center_x, center_y = centers[contour_id]

            # Define ROI boundaries
            start_x = max(0, center_x - WINDOW_WIDTH // 2)
            end_x = min(gray_frame.shape[1], center_x + WINDOW_WIDTH // 2)
            start_y = max(0, center_y - WINDOW_HEIGHT // 2)
            end_y = min(gray_frame.shape[0], center_y + WINDOW_HEIGHT // 2)

            # Process ROI
            roi = gray_frame[start_y:end_y, start_x:end_x]
            working_mask[start_y:end_y, start_x:end_x] = roi

            detected_contours, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_iou = 0
            best_contour = None
            best_mask = None

            for detected_contour in detected_contours:
                temp_mask = np.zeros_like(gray_frame)
                cv2.drawContours(temp_mask, [detected_contour], 0, 255, -1)

                intersection = np.logical_and(contour_mask, temp_mask)
                union = np.logical_or(contour_mask, temp_mask)
                iou = np.sum(intersection) / np.sum(union)

                if iou > max_iou:
                    max_iou = iou
                    best_contour = detected_contour
                    best_mask = temp_mask

            # Update logic based on IOU thresholds
            if max_iou > 0.90:
                non_updated_count, non_updated_count_plus, non_updated_count_plus2 = update_and_remove_contour(
                    contour_id, best_contour, best_mask, non_updated_count, non_updated_count_plus, non_updated_count_plus2
                )

            elif contour_id in non_updated_count and max_iou > 0.8:
                non_updated_count, non_updated_count_plus, non_updated_count_plus2 = update_and_remove_contour(
                    contour_id, best_contour, best_mask, non_updated_count, non_updated_count_plus, non_updated_count_plus2
                )

            elif contour_id in non_updated_count_plus and max_iou > 0.7:
                non_updated_count, non_updated_count_plus, non_updated_count_plus2 = update_and_remove_contour(
                    contour_id, best_contour, best_mask, non_updated_count, non_updated_count_plus, non_updated_count_plus2
                )

            elif contour_id in non_updated_count_plus2 and max_iou > 0.6:
                non_updated_count, non_updated_count_plus, non_updated_count_plus2 = update_and_remove_contour(
                    contour_id, best_contour, best_mask, non_updated_count, non_updated_count_plus, non_updated_count_plus2
                )

        # Final update checking
        for contour_id, contour_mask in targets.items():
            if (output_img == (contour_id + GRAY_VALUE)).sum() == 0:
                output_img[contour_mask == 255] = contour_id + GRAY_VALUE
                contour_update_counters[contour_id] += 1

                if contour_update_counters[contour_id] > 2:
                    non_updated_count.append(contour_id)
                if contour_update_counters[contour_id] > 5:
                    non_updated_count_plus.append(contour_id)
                if contour_update_counters[contour_id] > 10:
                    non_updated_count_plus2.append(contour_id)
            if LABELING_ENABLED:
                cv2.putText(output_img, str(contour_id), centers[contour_id], cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        # Save results
        output_filename = output_directory + current_file.split('/')[-1]
        cv2.imwrite(output_filename, output_img)

        print(output_filename)
        print(non_updated_count)
        print(non_updated_count_plus)
        print(non_updated_count_plus2)