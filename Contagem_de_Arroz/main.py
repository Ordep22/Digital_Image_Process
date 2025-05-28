import time

import cv2
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


#TODO:Implement a better way to read more than one paths at running time
#TODO:Implement a generic case for all images in to data base do count the elements


PATH_60 = r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\60.bmp"
PATH_82 = r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\82.bmp"
PATH_114 = r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\114.bmp"
PATH_150 = r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\150.bmp"
PATH_205 = r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\205.bmp"

class HandleImage:
    def __init__(self):
        pass

    def read_image(self, path):
        self.img = cv.imread(cv.samples.findFile(path),cv.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileExistsError(f'Image not found at {path}')
        else:
            self.img.astype(np.float32) / 255

    def save_image(self, img, file_name):
        img = img.astype(np.uint8)
        cv.imwrite(file_name, img)
        time.sleep(2)

    def show_image(self, text_title, img):
        cv.imshow(text_title, img)
        cv.waitKey(0)
        cv.destroyAllWindows()




def main():
    handle_image = HandleImage()
    handle_image.read_image(PATH_60)
    #handle_image.show_image(r"Show first image", handle_image.img)
    thresh_img = cv.adaptiveThreshold(handle_image.img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,27,-10)
    #handle_image.show_image(r"Show threshInv image", thresh_img)

    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    min_area  = 50 #TODO: create a max and min value based on the found contours
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

    object_count  = len(filtered_contours)
    cv.drawContours(handle_image.img,filtered_contours,-1,(0,100,0),1)
    cv.putText(handle_image.img, f"Object:{object_count}", (10,30), cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    handle_image.show_image(r"Thresh image", thresh_img)
    handle_image.show_image("Final image",handle_image.img)


if __name__ == "__main__":
    main()
