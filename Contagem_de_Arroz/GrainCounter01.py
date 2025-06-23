import cv2 as cv
import numpy as np
import os


def analyze_and_count_grains_final(image_path, min_noise_area=50, min_circularity=0.45, max_circularity=0.8,
                                   debug=False):
    """
    Final version that performs counting and classification of rice grains.
    - Uses the Top-hat + Watershed approach for precise segmentation.
    - Classifies objects based on a range of area and circularity for greater robustness.
    """
    image = cv.imread(image_path)
    filename = os.path.basename(image_path)
    expected_count = GROUND_TRUTH.get(filename, 0)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return 0, 0

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Step 1: Robust preprocessing with Top-hat and Otsu (unchanged)
    kernel_tophat = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel_tophat)
    _, thresh = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Step 2: Preparation for Watershed (unchanged)
    kernel_opening = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_opening, iterations=2)
    sure_bg = cv.dilate(opening, kernel_opening, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    threshold_value = 0.45999 * dist_transform.max()
    _, sure_fg = cv.threshold(dist_transform, threshold_value, 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Step 3: Application of Watershed (unchanged)
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers)

    # --- STEP 4: ANALYSIS AND CLASSIFICATION WITH ENHANCED LOGIC ---
    total_objects = len(np.unique(markers)) - 1
    whole_grains = 0
    other_particles = 0

    result_image_debug = image.copy()

    for i in range(2, total_objects + 2):
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == i] = 255
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts: continue

        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)

        # 1. Initial noise filter
        if area < min_noise_area:
            # Does not draw anything for noise, just ignores it
            continue

        perimeter = cv.arcLength(c, True)
        if perimeter == 0: continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # 2. Main criterion based on shape (circularity)
        if min_circularity < circularity < max_circularity:
            whole_grains += 1
            cv.drawContours(result_image_debug, [c], -1, (0, 255, 0), 2)  # Green for whole grains
        else:
            other_particles += 1
            cv.drawContours(result_image_debug, [c], -1, (0, 255, 255), 2)  # Yellow for broken/clustered particles

    if debug:
        # Display the results on the image for easy visualization
        text_defaut_grains = f"Total grains: {expected_count}"
        text_whole = f"Amout of grains: {whole_grains + other_particles}"
        text_error = f"Error: {((expected_count - (whole_grains + other_particles)) / expected_count) * 100}%"
        cv.putText(result_image_debug, text_defaut_grains, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(result_image_debug, text_whole, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(result_image_debug, text_error, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.imshow("Final Classification", result_image_debug)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Return the count of whole grains and other particles
    return whole_grains, other_particles


if __name__ == "__main__":
    PATH_ARRAY = [
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\60.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\82.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\114.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\150.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\205.bmp"
    ]
    GROUND_TRUTH = {
        "60.bmp": 60, "82.bmp": 82, "114.bmp": 114, "150.bmp": 150, "205.bmp": 205
    }

    print("Starting final analysis and grain classification...\n")

    total_error_final = 0
    for image_path in PATH_ARRAY:
        filename = os.path.basename(image_path)
        # Call the final function
        whole, broken = analyze_and_count_grains_final(image_path, debug=True)

        expected_count = GROUND_TRUTH.get(filename, 0)
        # The error is now the difference between the expected count and the grains classified as whole
        error = abs(whole - expected_count)
        total_error_final += error

        print(f"--- Image: {filename} ---")
        print(f"Expected Count (whole): {expected_count}")
        print(f" -> Found Whole Grains: {whole} (Error: {error})")
        print(f" -> Other Particles (broken/clustered): {broken}\n")

    print(f"======================================")
    print(f"ANALYSIS COMPLETE")
    print(f"Total Accumulated Error in Whole Grain Count: {total_error_final}")
    print(f"======================================")