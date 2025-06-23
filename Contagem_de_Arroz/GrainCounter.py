import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class RaiceGrainCounter:
    def __init__(self, sigma, block_size, C, min_area_px=80, max_area_px=1200):
        self.sigma  = sigma
        self.block_size = block_size
        self.C = C
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px

    def preprocess(self,image):
        image = cv.imread(image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (0, 0), sigmaX=self.sigma)
        return gray, blur

    def threshold_and_morphology(self,blur):
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 31, 5)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
        return thresh, opening

    def watershed_segmentation(self, image, opening):

        pass