'''
Challenge:
- Naive algorithm.
- Separable filter (choose whether you want to do it with or without using previous sums).
- Algorithm with integral images.
'''

import sys
import cv2 as cv
import numpy as np

imagePath = r"/Users/PedroVitorPereira/Documents/GitHub/Digital_Image_Process/Average Filter/Instructions/C lib/Exemplos/a01 - Original.bmp"
THRESHOLD = 0.75


class HandleImage:
    def __init__(self):
        pass

    def showImage(self, image, titleWindow):
        cv.imshow(titleWindow, (image * 255).astype('uint8'))
        cv.waitKey(0)
        cv.destroyAllWindows()

    def SaveImage(self):
        pass


class ImageProcessor:
    def __init__(self):
        pass

    def Makebiniarizaed(self, image):
        binarizedImage = np.where(image >= THRESHOLD, 1, 0).astype('uint16')
        return binarizedImage

    def ApplyAverageFilterBasic(self, image, kernel_size):

        offset = kernel_size // 2
        img_out = np.zeros_like(image)

        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                sum = 0
                for krow in range(row - offset, row + offset + 1):
                    for kcolumn in range(column - offset, column + offset + 1):
                        if 0 <= krow < image.shape[0] and 0 <= kcolumn < image.shape[1]:
                            sum += image[krow][kcolumn]
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

        img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
        img_integral = np.zeros_like(img_pad,dtype=np.float32)
        img_out = np.zeros_like(img,dtype=np.float32)

        #Calculing integral image
        for row in range(1, img_pad.shape[0] ):
            sum = np.sum(img_pad[row, 0:width])
            for column in range(1, img_pad.shape[1]):
                img_integral[row][column] = (img_pad[row, column] +
                                                           img_integral[row - 1, column] +
                                                           img_integral[row, column - 1] +
                                                           img_integral[row - 1, column - 1])
        #Blur image
        for row in range(img_integral.shape[0]):
            for column in range(img_integral.shape[1]):
                sum = 0
                r1 = row + 1 - offset_vertical
                r2 = row + 1 + offset_vertical
                c1 = column + 1 - offset_horizontal
                c2 = column + 1 + offset_horizontal

                r1 = max(r1, 0)
                c1 = max(c1, 0)
                r2 = min(r2, img_integral.shape[0] - 1)
                c2 = min(c2, img_integral.shape[1] - 1)

                sum = (img_integral[r2, c2] - img_integral[r1 - 1, c2] - img_integral[r2, c1 - 1] + img_integral[r1 - 1, c1 - 1])
                img_out[row][column] = sum / (height * width)

        return img_out


def main():
    imgProcessor = ImageProcessor()
    handleImage = HandleImage()

    img = cv.imread(cv.samples.findFile(imagePath), cv.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255

    if img is None:
        print("Could not load image")
        sys.exit()

    #img_binarized = imgProcessor.Makebiniarizaed(img)

    #img_blur_basic = imgProcessor.ApplyAverageFilterBasic(img, 5)

    #img_blur_separable_basic = imgProcessor.ApplySeparableAverageFilterBasic(img, 3, 3)

    #handleImage.showImage(img_blur_separable_basic, "Image Blur Basic")

    #img_blur_separable_smart = imgProcessor.ApplySeparableAverageFilterSmart(img, 3, 3)

    #handleImage.showImage(img_blur_separable_smart, "Image Blur Smart")

    image_blur_integral_image = imgProcessor.ApplyIntegralImage(img, 3, 3)

    handleImage.showImage(image_blur_integral_image, "Image Blur with integral image")


if __name__ == '__main__':
    main()
