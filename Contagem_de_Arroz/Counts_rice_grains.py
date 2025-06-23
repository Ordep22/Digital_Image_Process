import cv2 as cv
import numpy as np
import os


def count_grains_final(image_path, debug=False):
    """
    Final Version - Counts rice grains with high precision.
    This function combines a robust segmentation pipeline with an area-based
    heuristic to estimate the count in dense clusters, using a correction
    factor to account for grain overlap.
    """
    # --- Final Configuration Parameters ---
    MIN_GRAIN_AREA = 80
    MAX_GRAIN_AREA = 450
    MIN_SOLIDITY = 0.94
    MIN_ASPECT_RATIO = 1.5
    MAX_ASPECT_RATIO = 4.0
    DEFAULT_AVG_AREA = 150

    # Factor to compensate for the area lost due to overlap in clusters.
    # A value < 1 increases the estimated count. E.g., 0.9 means we consider
    # that a grain in a cluster occupies, on average, 90% of the area of an isolated grain.
    OVERLAP_CORRECTION_FACTOR = 0.90

    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return 0

    filename = os.path.basename(image_path)
    # This assumes a global GROUND_TRUTH dictionary is defined, as in previous versions.
    # For standalone use, you might pass GROUND_TRUTH as an argument.
    # expected_count = GROUND_TRUTH.get(filename, 0)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Step 1: Segmentation with Top-hat, Otsu, and Watershed
    kernel_tophat = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel_tophat)
    _, thresh = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel_opening = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_opening, iterations=2)
    sure_bg = cv.dilate(opening, kernel_opening, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    threshold_value = 0.45999 * dist_transform.max()
    _, sure_fg = cv.threshold(dist_transform, threshold_value, 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers)

    # Step 2: Post-processing with Area-based Heuristic
    good_grains_contours = []
    cluster_contours = []

    for i in range(2, len(np.unique(markers))):
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == i] = 255
        # Find contours of the segmented object
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)

        if area < MIN_GRAIN_AREA:  # Filter out noise
            continue

        hull = cv.convexHull(c)
        solidity = float(area) / cv.contourArea(hull) if cv.contourArea(hull) > 0 else 0

        aspect_ratio = 0
        try:
            (x, y), (w, h), angle = cv.minAreaRect(c)
            if min(w, h) > 0:
                aspect_ratio = max(w, h) / min(w, h)
        except (cv.error, ValueError):
            continue

        # Classify as a "good" grain or a potential cluster
        if (MAX_GRAIN_AREA > area and solidity > MIN_SOLIDITY and
                MAX_ASPECT_RATIO > aspect_ratio > MIN_ASPECT_RATIO):
            good_grains_contours.append(c)
        else:
            cluster_contours.append(c)

    # Calculate the average area of a single "good" grain for this specific image
    avg_grain_area = sum(cv.contourArea(c) for c in good_grains_contours) / len(
        good_grains_contours) if good_grains_contours else DEFAULT_AVG_AREA

    # Start the count with the number of confirmed good grains
    final_grain_count = len(good_grains_contours)

    # Calculate the corrected area to account for overlap
    corrected_area_per_grain = avg_grain_area * OVERLAP_CORRECTION_FACTOR

    # Estimate grains in clusters
    for c in cluster_contours:
        cluster_area = cv.contourArea(c)
        # Estimate how many grains fit in the cluster's area using the corrected factor
        estimated_grains = round(cluster_area / corrected_area_per_grain)
        final_grain_count += max(1, estimated_grains)  # Ensure we count at least 1

    # --- Debug Visualization ---
    if debug:
        expected_count = GROUND_TRUTH.get(filename, 0)
        result_image = image.copy()
        cv.drawContours(result_image, good_grains_contours, -1, (0, 255, 0), 2)  # Green for good grains
        cv.drawContours(result_image, cluster_contours, -1, (0, 255, 255), 2)  # Yellow for clusters

        text_expected = f"Ground Truth: {expected_count}"
        text_counted = f"Counted (est.): {int(final_grain_count)}"
        error_percent = (abs(final_grain_count - expected_count) / expected_count) * 100 if expected_count > 0 else 0
        text_error = f"Image Error: {error_percent:.2f}%"

        cv.putText(result_image, text_expected, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(result_image, text_counted, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(result_image, text_error, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.imshow(f"Final Result - {filename}", result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return int(final_grain_count)


# ==============================================================================
# Main execution block for testing
# ==============================================================================
if __name__ == "__main__":
    # Define the paths to the image files
    PATH_ARRAY = [
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\60.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\82.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\114.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\150.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\205.bmp"
    ]
    # Define the ground truth for each image
    GROUND_TRUTH = {
        "60.bmp": 60, "82.bmp": 82, "114.bmp": 114, "150.bmp": 150, "205.bmp": 205
    }

    print("Starting grain count - Calculating Overall Percentage Error...\n")

    total_absolute_error = 0
    total_expected_grains = 0

    for image_path in PATH_ARRAY:
        filename = os.path.basename(image_path)

        detected_count = count_grains_final(image_path, debug=True)
        expected_count = GROUND_TRUTH.get(filename, 0)

        # Calculate the error in number of grains (absolute error)
        absolute_error = abs(detected_count - expected_count)

        # Accumulate absolute errors and total expected grains
        total_absolute_error += absolute_error
        total_expected_grains += expected_count

        print(f"Image: {filename}")
        print(f" -> Expected count: {expected_count}")
        print(f" -> Detected grains (estimated): {detected_count}")
        print(f" -> Absolute error (grains): {absolute_error}")
        if expected_count > 0:
            print(f" -> Image percentage error: {(absolute_error / expected_count) * 100:.2f} %\n")
        else:
            print("\n")

    # Calculate the overall percentage error based on the totals
    overall_percentage_error = (total_absolute_error / total_expected_grains) * 100 if total_expected_grains > 0 else 0

    print(f"=============================================")
    print(f"Total Expected Grains: {total_expected_grains}")
    print(f"Total Errors (in grains): {total_absolute_error}")
    print(f"Overall Percentage Error (Total): {overall_percentage_error:.2f} %")
    print(f"=============================================")