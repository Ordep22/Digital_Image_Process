import cv2 as cv
import numpy as np

# Directory where result images will be saved
DIR_PATH = r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\Result\\"

# List of input images with green screen background
PATHS = [
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\0.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\1.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\2.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\3.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\4.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\5.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\6.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\7.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\8.bmp"
]

# New background to be inserted behind the extracted subject
PATH_NEW_BACKGROUND = r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\faria_lima_original.png"

class ImageProcess:
    def __init__(self):
        pass

    def read_image(self, img_path):
        return cv.imread(img_path)

    def show_image(self, title_window, img):
        cv.imshow(title_window, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def save_image(self, img, title):
        img = img.astype(np.uint8)
        cv.imwrite(DIR_PATH + title + ".png", img)

    def extract_foreground(self, img, debug=True):
        """
        Applies chroma key to isolate the foreground object from a green background.
        Outputs original image and an alpha mask for blending.
        """
        # Slight blur reduces high-frequency noise before masking
        img_blurred = cv.blur(img, ksize=(9, 9))

        # Separate color channels
        b = img_blurred[:, :, 0]
        g = img_blurred[:, :, 1]
        r = img_blurred[:, :, 2]

        # Adaptive green screen thresholding based on image intensity
        offset_r, offset_b = self.compute_dynamic_threshold(img)
        mask_logic = (g > r + offset_r) & (g > b + offset_b)

        # Create binary mask for green screen
        green_mask = np.zeros_like(g, dtype=np.uint8)
        green_mask[mask_logic] = 255

        # Invert to get the foreground mask
        foreground_mask = cv.bitwise_not(green_mask)

        # Refine edges to handle transition zones and partial transparency
        refined_mask = self.refine_mask(
            binary_mask=foreground_mask,
            open_close_kernel=(3, 3),
            gaussian_ksize=(15, 15),
            debug=False
        )

        # Normalize mask to [0,1] range for alpha blending
        alpha = refined_mask.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        if debug:
            self.show_image("Refined Foreground Mask", refined_mask)

        return img, alpha

    def refine_mask(self, binary_mask, open_close_kernel=(3, 3), gaussian_ksize=(3, 3), debug=True):
        """
        Refines binary mask using morphological operations and Gaussian blur.
        - Opening: removes small noise spots
        - Closing: fills small holes
        - Gaussian blur: smooths transitions at the edges
        """
        kernel = np.ones(open_close_kernel, np.uint8)

        refined = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)
        refined = cv.morphologyEx(refined, cv.MORPH_CLOSE, kernel)
        refined = cv.erode(refined, kernel, iterations=2)

        # Gaussian blur helps reduce harsh edges and aliasing
        refined = cv.GaussianBlur(refined, gaussian_ksize, 0)

        if debug:
            self.show_image("Mask after Refinement", refined)

        return refined

    def compute_dynamic_threshold(self, img):
        """
        Dynamically calculates offset thresholds to identify green dominance.
        This makes the method robust to variations in lighting and saturation.
        """
        g = img[:, :, 1].flatten()
        r = img[:, :, 2].flatten()
        b = img[:, :, 0].flatten()

        g_ref = np.mean(g)

        # Offsets are clipped to avoid removing foreground content in bright/dark scenes
        offset_r = max(g_ref - np.median(r), 45)
        offset_b = max(g_ref - np.median(b), 45)

        return offset_r, offset_b

    def assble_new_image(self, foreground, alpha, background_path, debug=True):
        """
        Performs alpha blending to combine the original foreground with a new background.
        """
        foreground = foreground.astype(np.float32)
        background = self.read_image(background_path)
        background = cv.resize(background, (foreground.shape[1], foreground.shape[0])).astype(np.float32)

        blended = alpha * foreground + (1 - alpha) * background
        new_image = blended.astype(np.uint8)

        if debug:
            self.show_image("Final Composite", new_image)

        return new_image


def main():
    process = ImageProcess()
    for path in PATHS:
        print(f"Processing image: {path[-9:]}")
        img = process.read_image(path)
        foreground, alpha = process.extract_foreground(img, debug=False)
        new_image = process.assble_new_image(foreground, alpha, PATH_NEW_BACKGROUND, debug=True)
        filename = path.split("\\")[-1].split(".")[0]
        process.save_image(new_image, title="result_image_" + filename)

if __name__ == "__main__":
    main()
