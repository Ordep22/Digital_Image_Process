'''
Challeng
Objective: Implement the bloom effect in two versions, using Gaussian filtering and box blur.
-> You MAY (and SHOULD!) use OpenCV’s built-in functions for the filters.
-> For the bright-pass, do not perform binarization independently on the 3 channels!
-> Note that the σ values increase significantly between filters.
-> Remember that the substitution is not a one-to-one replacement of the Gaussian filter with the mean (box) filter;
each application of the Gaussian filter is approximated by several successive applications of the mean filter!
'''

import time
import cv2 as cv
import numpy as np

PATH = r'C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Bloom\Images\zelda.png'


class HandleImage:
    def __init__(self):
        self.img = None
        pass
    def readImage(self,path):
        img = cv.imread(cv.samples.findFile(path), cv.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255
        return img
    def showImage(self, img, titleWindow):
        cv.imshow(titleWindow, (img * 255).astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()

    def SaveImage(self, img, titleWindow):
        img = (img * 255).astype(np.uint8)
        cv.imwrite(titleWindow, img)
        time.sleep(0.1)


def main():
    handleimage = HandleImage()
    img = handleimage.readImage(PATH)
    handleimage.showImage(img, "Zelda Image")


if __name__ == "__main__":
    main()
