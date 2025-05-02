import sys
import cv2 as cv
import numpy as np

imagePath  = r"/Users/PedroVitorPereira/Documents/GitHub/Digital_Image_Process/Average Filter/Instructions/C lib/Exemplos/a01 - Original.bmp"
THRESHOLD = 0.75


class HandleImage:
    def __init__(self):
        pass
    def showImage(self, image,titleWindow):
        cv.imshow("Binarized_image", (image * 255).astype('uint8'))
        cv.waitKey(0)
        cv.destroyAllWindows()
    def SaveImage(self):
        pass
class ImageProcessor:
    def __init__(self):
        pass
    def Makebiniarizaed(self, image):
        binarizedImage = np.where(image >= THRESHOLD, 1,0).astype('uint16')
        return binarizedImage
    def ApplyAverageFilterBasic(self,image, kernel_size):

        offset = kernel_size // 2
        img_out = np.zeros_like(image)

        for row in range(image.shape[0]):
            for colunm in range(image.shape[1]):
                soma = 0
                for krow in range(row - offset, row + offset + 1):
                    for kcolunm in range(colunm - offset, colunm + offset + 1):
                        if 0 <= krow < image.shape[0] and 0 <= kcolunm < image.shape[1]:
                            soma += image[krow][kcolunm]
                        else:
                            soma += 255
                img_out[row][colunm] = soma /(kernel_size**2)

        return img_out

def main():
    imgProcessor = ImageProcessor()
    handleImage = HandleImage()

    img = cv.imread(cv.samples.findFile(imagePath), cv.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)/255

    if img is None:
        print("Could not load image")
        sys.exit()

    #img_binarized = imgProcessor.Makebiniarizaed(img)

    img_blur_Newby  = imgProcessor.ApplyAverageFilterBasic(img, 5)

    handleImage.showImage(img_blur_Newby,"Binarized_image")




if __name__ == '__main__':
    main()
