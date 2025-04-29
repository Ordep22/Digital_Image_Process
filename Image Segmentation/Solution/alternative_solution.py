import cv2 as cv
import numpy as np
import sys

THRESHOLD = 0.75
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 1
IMGPATH = r''

class ManageStack():

    def __init__(self):
        self.stack = []
    def push(self, pixel_coordinate):
        self.stack.append(pixel_coordinate)
    def pop(self):
        if not self.isEmpty():
            return self.stack.pop()
        else:
            print("Stack  is empyt")
            return False

    def peek(self):
        if (self.isEmpty()):
            print("Stack  is empyt")
            return False
        else:
            return self.stack[self.size() - 1]
    def isEmpty(self):

        if len(self.stack) == 0:
            return True
        else:
            return False
    def size(self):
        return len(self.stack)

class ImageProcessing:

    def __init__(self):
        self.mngStack = ManageStack()
        self.components = []
        self.number_pixels = 0

    def BrowseImage(self, img, label=2):

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x,y] == 1:
                    self.FloodFill(label,img,x,y)
                    label += 1
        return self.components

    def FloodFill(self, label_position, img, x0, y0):

        self.mngStack.push([x0, y0])
        top, left = x0, y0
        bottom, right = x0, y0
        number_pixels = 0

        while not self.mngStack.isEmpty():

            x,y = self.mngStack.pop()
            if ((0 <=x  < img.shape[0]) and (0 <= y < img.shape[1])):

                if img[x][y] == 1:

                    img[x][y] = label_position
                    number_pixels += 1

                    top = min(top, x)
                    bottom = max(bottom, x)
                    left = min(left, y)
                    right = max(right, y)

                    self.mngStack.push([x + 1,y])
                    self.mngStack.push([x - 1, y])
                    self.mngStack.push([x, y + 1])
                    self.mngStack.push([x, y -1])
                else:
                    pass

        if number_pixels <= N_PIXELS_MIN:

            pass
        else:
            self.components.append(
                {
                    'Label': label_position,
                    'n_pixels': number_pixels,
                    'top': top,
                    'left': left,
                    'bottom': bottom,
                    'right': right
                }
            )

    def Labeling(self, component, img):

        for comp in component:
            top = comp['top']
            left = comp['left']
            bottom = comp['bottom']
            right = comp['right']
            cv.rectangle(img,(left,top),(right, bottom), (0,0,255),1)


class Run:

    def __init__(self):
        self.imgProc = ImageProcessing()
        pass
    def main(self):

        img = cv.imread(cv.samples.findFile( IMGPATH),cv.IMREAD_GRAYSCALE)
        img_out = cv.cvtColor (img, cv.COLOR_GRAY2BGR)
        img = img.astype(np.float32) / 255

        if img is None:
            print('Could not read the image. \n')
            sys.exit()

        img_bin = np.where(img >= THRESHOLD, 1, 0).astype('uint16')

        cv.imshow("Binarized image", (img_bin*255).astype('uint8'))

        key = cv.waitKey(0)
        if key == ord('d'):
            cv.destroyAllWindows()

        cv.imwrite("Binarized_image.png", (img_bin*255).astype('uint8'))

        coordinat  = self.imgProc.BrowseImage(img_bin)

        self.imgProc.Labeling(coordinat,img_out)

        cv.imshow("Flood Fill", img_out)

        key = cv.waitKey(0)
        if key == ord('d'):
            cv.destroyAllWindows()

        cv.imwrite("Flood_Fill_result.png", img_out)


if __name__ == '__main__':
    run = Run()
    run.main()

