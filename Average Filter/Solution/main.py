'''
Challenge:
- Naive algorithm.
- Separable filter (choose whether you want to do it with or without using previous sums).
- Algorithm with integral images.
'''

import sys
import time

import cv2
import cv2 as cv
import numpy as np

imagePath = r"/Users/PedroVitorPereira/Documents/GitHub/Digital_Image_Process/Average Filter/Instructions/C lib/Exemplos/a01 - Original.bmp"
THRESHOLD = 0.75


class HandleImage:
    def __init__(self):
        pass

    def showImage(self, img, titleWindow):
        cv.imshow(titleWindow, (img * 255).astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()

    def SaveImage(self, img, titleWindow):
        img = (img * 255).astype(np.uint8)
        cv.imwrite(titleWindow, img)
        time.sleep(0.1)


class ImageProcessor:
    def __init__(self):
        pass

    def ApplyAverageFilterBasic(self, img, kernel_size):

        offset = kernel_size // 2
        img_out = np.zeros_like(img)

        for row in range(img.shape[0]):
            for column in range(img.shape[1]):
                sum = 0
                for krow in range(row - offset, row + offset + 1):
                    for kcolumn in range(column - offset, column + offset + 1):
                        if 0 <= krow < img.shape[0] and 0 <= kcolumn < img.shape[1]:
                            sum += img[krow][kcolumn]
                        else:
                            sum += 0
                img_out[row][column] = sum / (kernel_size ** 2)

        return img_out

    def ApplySeparableAverageFilterBasic(self, img, height, width):

        offset_vertical = height // 2
        offset_horizontal = width // 2
        img_out_horizintal_filter = np.zeros_like(img)
        img_out = np.zeros_like(img)

        for row in range(img.shape[0]):
            for column in range(img.shape[1]):
                sum = 0
                for kcolumn in range(column - offset_horizontal, column + offset_horizontal + 1):
                    if 0 <= kcolumn < img.shape[1]:
                        sum += img[row][kcolumn]
                    else:
                        sum += 0
                img_out_horizintal_filter[row][column] = sum / width

        for column in range(img.shape[1]):
            for row in range(img.shape[0]):
                sum = 0
                for krow in range(row - offset_vertical, row + offset_vertical + 1):
                    if 0 <= krow < img.shape[0]:
                        sum += img_out_horizintal_filter[krow][column]
                    else:
                        sum += 0
                img_out[row][column] = sum / height

        return img_out

    def ApplySeparableAverageFilterSmart(self, img, height, width):
        offset_vertical = height // 2
        offset_horizontal = width // 2

        img_out_horizontal_filter = np.zeros_like(img, dtype=float)
        img_out = np.zeros_like(img, dtype=float)

        for row in range(img.shape[0]):
            sum = np.sum(img[row, 0:width])
            for column in range(offset_horizontal, img.shape[1] - offset_horizontal):
                if column != offset_horizontal:
                    sum = sum - img[row, column - offset_horizontal - 1] + img[row, column + offset_horizontal]
                img_out_horizontal_filter[row, column] = sum / width

        for column in range(offset_horizontal, img.shape[1] - offset_horizontal):
            sum = np.sum(img_out_horizontal_filter[0:height, column])
            for row in range(offset_vertical, img.shape[0] - offset_vertical):
                if row != offset_vertical:
                    sum = sum - img_out_horizontal_filter[row - offset_vertical - 1, column] + \
                          img_out_horizontal_filter[
                              row + offset_vertical, column]
                img_out[row, column] = sum / height

        return img_out

    def ApplyIntegralImage(self, img, height, width):

        offset_vertical = height // 2
        offset_horizontal = width // 2

        img = img.astype(np.float32)
        img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant').astype(np.float32)
        img_integral = np.zeros_like(img_pad,dtype=np.float64)
        img_out = np.zeros_like(img_integral,dtype=np.float64)

        for row in range(1, img_pad.shape[0] ):
            for column in range(1, img_pad.shape[1]):
                img_integral[row][column] = (img_pad[row, column] +
                                             img_integral[row - 1, column] +
                                             img_integral[row, column - 1] -
                                             img_integral[row - 1, column - 1])

        for row in range(img_integral.shape[0]):
            sum = 0
            for column in range(img_integral.shape[1]):
                if ((row + 1 + offset_vertical )< img_integral.shape[0] and (column + 1 + offset_horizontal) < img_integral.shape[1] and (row + 1 - offset_vertical - 1) >= 0 and (column + 1 + offset_horizontal -1 ) >= 0 ):
                    sum = (img_integral[row + 1 + offset_vertical, column + 1 + offset_horizontal]
                           - img_integral[row + 1 - offset_vertical - 1, column + 1 + offset_horizontal]
                           - img_integral[row + 1 + offset_vertical, column + 1 - offset_horizontal - 1]
                           + img_integral[row + 1 - offset_vertical - 1, column + 1 - offset_horizontal - 1])
                    img_out[row][column] = sum / (height * width)

        return img_out


def main():
    height = 5
    width = 5
    imgProcessor = ImageProcessor()
    handleImage = HandleImage()

    img = cv.imread(cv.samples.findFile(imagePath), cv.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255

    if img is None:
        print("Could not load img")
        sys.exit()

    img_blur_basic = imgProcessor.ApplyAverageFilterBasic(img, 5)

    handleImage.showImage(img_blur_basic, "Image Blur with the basic implemmentation")

    handleImage.SaveImage(img_blur_basic, "Blur_Basic_" + str(height) +"_"+ str(width) +".png")

    #img_blur_separable_basic = imgProcessor.ApplySeparableAverageFilterBasic(img, height, width)

    #handleImage.showImage(img_blur_separable_basic, "Image Blur Basic")

    #handleImage.SaveImage(img_blur_basic, "Blur_Separable_Basic_" + str(height) + "_" + str(width) + ".png")

    #img_blur_separable_smart = imgProcessor.ApplySeparableAverageFilterSmart(img, height, width)

    #handleImage.showImage(img_blur_separable_smart, "Image Blur Smart")

    #handleImage.SaveImage(img_blur_basic, "Blur_Separable_Smart_" + str(height) + "_" + str(width) + ".png")

    #image_blur_integral_image = imgProcessor.ApplyIntegralImage(img, height, width)

    #handleImage.showImage(image_blur_integral_image, "Image Blur with integral img")

    #handleImage.SaveImage(img_blur_basic, "Blur_Integral_Image_" + str(height) + "_" + str(width) + ".png")

    #image_blur_open_cv = cv.blur(img,(height,width))

    #handleImage.showImage(image_blur_open_cv, "Image Blur with OpenCv implementation")

    #handleImage.SaveImage(img_blur_basic, "Image_Blur_Open_CV" + str(height) + "_" + str(width) + ".png")

if __name__ == '__main__':
    main()
