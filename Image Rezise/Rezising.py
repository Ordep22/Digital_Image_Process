import cv2 as cv
def main():
    img = cv.imread(r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Image Rezise\Images\a01 - Original.bmp")
    print(img.shape)
    cv.imshow("Orignal Image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_rezised_nearest = cv.resize(img,(500,500), interpolation=cv.INTER_NEAREST)
    print(img_rezised_nearest.shape)
    cv.imshow("Resized Image - Interpolation Nearest", img_rezised_nearest)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_rezised_linear = cv.resize(img,(500,500), interpolation=cv.INTER_LINEAR)
    print(img_rezised_linear.shape)
    cv.imshow("Resized Image - Linear interpolation", img_rezised_linear)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    img_rezised_cubic = cv.resize(img,(500,500), interpolation=cv.INTER_CUBIC)
    print(img_rezised_cubic.shape)
    cv.imshow("Resized Image - Cubic interpolation", img_rezised_cubic)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_rezised_inter_area = cv.resize(img,(500,500), interpolation=cv.INTER_AREA)
    print(img_rezised_inter_area.shape)
    cv.imshow("Resized Image - Inter Area interpolation", img_rezised_inter_area)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_rezised_lanczos = cv.resize(img, (500, 500), interpolation=cv.INTER_LANCZOS4)
    print(img_rezised_lanczos.shape)
    cv.imshow("Resized Image - Lanczos interpolation", img_rezised_lanczos)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
