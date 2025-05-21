'''
Challenge
Objective: Implement the bloom effect in two versions, using Gaussian filtering and box blur.
-> You MAY (and SHOULD!) use OpenCV’s built-in functions for the filters.
-> For the bright-pass, do not perform binarization independently on the 3 channels!
-> Note that the σ values increase significantly between filters.
-> Remember that the substitution is not a one-to-one replacement of the Gaussian filter with the mean (box) filter;
each application of the Gaussian filter is approximated by several successive applications of the mean filter!
'''

import cv2 as cv
import numpy as np

PATH = r'C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Bloom\Images\zelda.png'
DIR_PATH = r'C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Bloom\Images\\'

class HandleImage:
    def __init__(self):
        self.img = None

    def read_image(self, path):
        self.img = cv.imread(cv.samples.findFile(path))
        if self.img is None:
            raise FileNotFoundError(f"Image not found at {path}")
        self.img = self.img.astype(np.float32) / 255
        return self.img

    def save_image(self, img, title):
        img = (img).astype(np.uint8)
        cv.imwrite(DIR_PATH + title + ".png", img)

    def show_image(self, img, title_window):
        cv.imshow(title_window, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

class BloomEffect(HandleImage):
    def __init__(self):
        super().__init__()
        self.bright_mask = None
        self.threshold = None
        self.blur = None
        self.result = None

    def extract_bright_regions_lab(self, threshold=210): #threshold tested with differente values, but the better result was threshold = 210
        '''
        Using Lab provides a more accurate representation of brightness, as the L channel reflects true
        luminance independently of color or saturation, making it more reliable for detecting bright regions.
        '''
        img_uint8 = (self.img * 255).astype(np.uint8)
        lab = cv.cvtColor(img_uint8, cv.COLOR_BGR2Lab)
        l_channel, _, _ = cv.split(lab)
        _, mask = cv.threshold(l_channel, threshold, 255, cv.THRESH_BINARY)
        self.bright_mask = mask

    def apply_linear_filter(self, method, sigma=30, kernel_size=21): #sigma could be 20,30,40 or 50
        if method == 0:
            self.blur = cv.GaussianBlur(self.bright_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        elif method == 1:
            self.blur = self.bright_mask.copy()
            for _ in range(10):
                self.blur = cv.blur(self.blur, (kernel_size, kernel_size))
        else:
            raise ValueError("Invalid blur method: use 0 for Gaussian or 1 for Box blur.")
        self.blur = cv.cvtColor(self.blur, cv.COLOR_GRAY2BGR)

    def blend_image(self, gain=3.0): #Teste with gain 1.5, 3.0, 4 and 6
        img = (self.img * 255).astype(np.float32)
        blur = self.blur.astype(np.float32)
        self.result = cv.addWeighted(img, 1.0, blur, gain, 0.0)
        self.result = np.clip(self.result, 0, 255).astype(np.uint8)


def main():
    bloom = BloomEffect()
    img = bloom.read_image(PATH)
    bloom.show_image(img, "Original Image")

    bloom.extract_bright_regions_lab(threshold=210)
    bloom.show_image(bloom.bright_mask, "Bright Mask")
    bloom.save_image(bloom.bright_mask, "Bright_Mask_Lab")

    # Gaussian Blur
    bloom.method = 0
    bloom.apply_linear_filter(bloom.method)
    bloom.show_image(bloom.blur, "Gaussian Blur")
    bloom.save_image(bloom.blur, "GaussianBlur")
    bloom.blend_image()
    bloom.show_image(bloom.result, "Bloom Image with Gaussian Blur")
    bloom.save_image(bloom.result, "Bloom_image_with_Gaussian_Blur")

    # Box Blur
    bloom.method = 1
    bloom.apply_linear_filter(bloom.method)
    bloom.show_image(bloom.blur, "Box Blur")
    bloom.save_image(bloom.blur, "BoxBlur")
    bloom.blend_image()
    bloom.show_image(bloom.result, "Bloom Image with Box Blur")
    bloom.save_image(bloom.result, "Bloom_image_with_Box_Blur")


if __name__ == "__main__":
    main()
