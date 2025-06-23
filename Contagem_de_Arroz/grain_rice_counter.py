import cv2 as cv
import numpy as np
import os

# --- Configuration Parameters ---
# These parameters were defined based on empirical tests and literature review,
# particularly inspired by the referenced work from Peterson Adriano Belan,
# as well as iterative tuning with sample images.

MIN_GRAIN_AREA = 80                  # Minimum area to consider a valid single grain (defined empirically)
MAX_GRAIN_AREA = 450                 # Maximum area for a single grain (helps identify clusters; empirically tuned)
MIN_SOLIDITY = 0.94                  # Minimum solidity threshold for valid grain contours (from shape analysis)
MIN_ASPECT_RATIO = 1.5              # Minimum aspect ratio to filter out too-round shapes (empirically chosen)
MAX_ASPECT_RATIO = 4.0              # Maximum aspect ratio for a valid grain (based on grain shape)
DEFAULT_GRAIN_AREA = 150            # Default area fallback when no valid grains detected (average grain size)
OVERLAP_CORRECTION_FACTOR = 0.90    # Correction factor to compensate overlapping grains in clusters (estimated)


# Image paths used for the experiment
IMAGE_PATHS = [
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\60.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\82.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\114.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\150.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\205.bmp"
]

# Ground truth grain counts for each image
GROUND_TRUTH = {
    "60.bmp": 60, "82.bmp": 82, "114.bmp": 114, "150.bmp": 150, "205.bmp": 205
}


class GrainCounter:
    def count_grains(self, image_path, debug):

        image = cv.imread(image_path)
        if image is None:
            return 0

        filename = os.path.basename(image_path)
        expected_count = GROUND_TRUTH.get(filename, 0)

        # --- Step 1: Grayscale Conversion ---
        # Convert to grayscale to simplify further processing,
        # as color information is not necessary for shape-based segmentation.
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # --- Step 2: Top-hat Morphological Filtering ---
        # Enhance small bright objects (e.g., rice grains) against a darker background.
        # The elliptical kernel mimics the elongated shape of rice grains.
        kernel_tophat = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
        tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel_tophat)

        # --- Step 3: Thresholding using Otsu's Method ---
        # Otsu’s algorithm automatically determines a threshold to separate foreground
        # (grains) from background by minimizing intra-class variance.
        _, thresh = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # --- Step 4: Morphological Opening ---
        # Removes small white noise and disconnects weakly connected objects (false positives).
        # The number of iterations was defined based on empirical testing across several code versions.
        kernel_opening = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_opening, iterations=2)

        # --- Step 5: Background and Foreground Estimation ---
        # Dilation ensures that the background regions are expanded and defined.
        # The number of iterations was defined based on empirical testing across several code versions.
        sure_bg = cv.dilate(opening, kernel_opening, iterations=3)

        # Compute the distance transform of the opened image.
        # This highlights the center of each object (grain) with high intensity values.
        # maskSize = 5 provides a smoother and more precise distance map than 3, which improves the stability
        # of the foreground segmentation in the next step.
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

        # Use a fixed ratio to select sure foreground regions (the core of the grains).
        # The value 0.45999 was determined through empirical testing across multiple code versions.
        threshold_value = 0.45999 * dist_transform.max()
        _, sure_fg = cv.threshold(dist_transform, threshold_value, 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Subtract the foreground from the background to obtain unknown regions,
        # which likely represent the boundaries between touching grains.
        unknown = cv.subtract(sure_bg, sure_fg)

        # --- Step 6: Marker Labelling and Watershed Segmentation ---
        # Label connected components as initial markers.
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1  # Ensure background is not labeled as 0.
        markers[unknown == 255] = 0  # Mark the unknown regions with 0.

        # Apply watershed algorithm to separate overlapping grains.
        # It treats the grayscale image as a topographic surface and floods it from the markers.
        markers = cv.watershed(image, markers)

        # --- Step 7: Heuristic Post-processing with Area and Shape Filters ---
        # This step applies shape-based heuristics to distinguish single grains from clusters or noise.
        # The heuristics are inspired by the work:
        # "Sistema de Visão Computacional para Inspeção da Qualidade de Grãos de Feijão"
        # by Peterson Adriano Belan, which guides grain quality inspection based on contour properties.

        good_grains_contours = []  # List to hold contours likely representing single grains
        cluster_contours = []  # List to hold contours that likely represent overlapping grain clusters

        # Loop through each detected marker region from the watershed segmentation,
        # starting from 2 to ignore background and border labels.
        for i in range(2, len(np.unique(markers))):
            # Create a mask isolating the current marker region
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == i] = 255

            # Find external contours in this isolated region
            cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue  # Skip if no contours found (defensive check)

            # Select the largest contour in this region, assuming it corresponds to a grain or cluster
            c = max(cnts, key=cv.contourArea)
            area = cv.contourArea(c)

            if area < MIN_GRAIN_AREA:
                # Ignore very small areas considered noise or irrelevant objects
                continue

            # Calculate solidity: the ratio between the contour area and its convex hull area
            # Solidity measures how 'solid' or 'compact' the shape is.
            # A high solidity (close to 1) suggests the contour is convex and grain-like,
            # whereas lower values indicate irregular or concave shapes typical of clusters or noise.
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # Compute the aspect ratio using the minimum area rectangle enclosing the contour
            # Aspect ratio helps filter out shapes that are too elongated or too round to be grains,
            # as grains generally have an elliptical shape within a certain ratio range.
            aspect_ratio = 0
            try:
                (x, y), (w, h), angle = cv.minAreaRect(c)
                if min(w, h) > 0:
                    aspect_ratio = max(w, h) / min(w, h)
            except:
                # Defensive: in case minAreaRect fails, skip this contour
                continue

            # Apply filtering heuristics based on area, solidity, and aspect ratio thresholds
            # - Area filters remove contours that are too big or too small
            # - Solidity filters remove contours that are not compact enough
            # - Aspect ratio filters remove shapes too elongated or too round for grains
            if (MAX_GRAIN_AREA > area and solidity > MIN_SOLIDITY and
                    MAX_ASPECT_RATIO > aspect_ratio > MIN_ASPECT_RATIO):
                good_grains_contours.append(c)  # Likely a single grain
            else:
                cluster_contours.append(c)  # Likely overlapping grains or noise

        # --- Step 8: Count Estimation Using Median Grain Area ---
        # Compute the median area of valid single grains to avoid influence from outliers
        # Median is more robust than mean, especially when clusters distort the distribution
        if good_grains_contours:
            good_grain_areas = [cv.contourArea(c) for c in good_grains_contours]
            median_grain_area = np.median(good_grain_areas)
        else:
            median_grain_area = DEFAULT_GRAIN_AREA  # Fallback to a safe default

        final_grain_count = len(good_grains_contours)

        # Estimate number of grains in clustered contours by dividing cluster area
        # by the corrected median area (considering overlaps)
        corrected_area_per_grain = median_grain_area * OVERLAP_CORRECTION_FACTOR
        for c in cluster_contours:
            cluster_area = cv.contourArea(c)
            estimated_grains = round(cluster_area / corrected_area_per_grain)
            final_grain_count += max(1, estimated_grains)  # Avoid zero estimation

        # --- Step 9: Optional Debug Visualization ---
        if debug:
            result_image = image.copy()

            # Draw single grains in green, clusters in yellow
            cv.drawContours(result_image, good_grains_contours, -1, (0, 255, 0), 2)
            cv.drawContours(result_image, cluster_contours, -1, (0, 255, 255), 2)

            # Overlay expected and estimated counts
            text_expected = f"Ground Truth: {expected_count}"
            text_counted = f"Counted (est.): {int(final_grain_count)}"
            error_percent = (
                abs(final_grain_count - expected_count) / expected_count * 100
                if expected_count > 0 else 0
            )
            text_error = f"Image Error: {error_percent:.2f}%"

            cv.putText(result_image, text_expected, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(result_image, text_counted, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv.putText(result_image, text_error, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show the final image with annotations
            cv.imshow(f"Final Result - {filename}", result_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return int(final_grain_count)


def main():
    """
    Runs grain counting on all provided images and evaluates total error.
    """
    counter = GrainCounter()

    print("Starting grain count analysis...\n")

    total_error = 0
    total_truth = 0

    for path in IMAGE_PATHS:
        filename = os.path.basename(path)
        estimated = counter.count_grains(path, debug=True)
        expected = GROUND_TRUTH.get(filename, 0)

        error = abs(estimated - expected)
        total_error += error
        total_truth += expected

        print(f"Image: {filename}")
        print(f" -> Expected: {expected}")
        print(f" -> Estimated: {estimated}")
        print(f" -> Absolute Error: {error}")
        if expected > 0:
            print(f" -> Error Percentage: {(error / expected) * 100:.2f}%\n")

    overall_error = (total_error / total_truth) * 100 if total_truth > 0 else 0

    print("=============================================")
    print(f"Total Expected Grains: {total_truth}")
    print(f"Total Error (grains): {total_error}")
    print(f"Overall Percentage Error: {overall_error:.2f}%")
    print("=============================================")


if __name__ == "__main__":
    main()
