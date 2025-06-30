import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

PATHS = [
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\0.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\1.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\2.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\3.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\4.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\5.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\6.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\7.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\8.BMP",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\9.png"
]

class image_process:
    def __init__(self):
        pass

    def read_image(self, img_path):
        img = cv.imread(img_path)
        return img

    def show_image(self, title_window, img):
        cv.imshow(title_window, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def extract_foreground(self, img, debug=True):
        """
        Extrai a máscara do foreground com base no fundo verde, utilizando suavização e morfologia.
        Retorna a imagem original e a máscara alpha suavizada.
        """
        img_blurred = cv.blur(img, (9, 9))

        # Separação dos canais
        b = img_blurred[:, :, 0]
        g = img_blurred[:, :, 1]
        r = img_blurred[:, :, 2]

        # Critério de verde dominante
        offset = 125
        mask_logic = (g > r + offset) & (g > b + offset)

        # Máscara binária inicial
        green_mask = np.zeros_like(g, dtype=np.uint8)
        green_mask[mask_logic] = 255

        # Inverte para pegar foreground e aplica refinamento
        foreground_mask = cv.bitwise_not(green_mask)
        kernel = np.ones((5, 5), np.uint8)
        refined_mask = cv.morphologyEx(foreground_mask, cv.MORPH_CLOSE, kernel)
        refined_mask = cv.GaussianBlur(refined_mask, (7, 7), 0)

        # Gera máscara alpha normalizada para blending
        alpha = refined_mask.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        if debug:
            self.show_image("Foreground Mask (refinada)", refined_mask)

        return img, alpha  # foreground image e alpha mask

    def assble_new_image(self, foreground, alpha, background, debug=True):
        """
        Aplica a composição da imagem usando a máscara alpha suavizada.
        """
        foreground = foreground.astype(np.float32)
        background = cv.resize(background, (foreground.shape[1], foreground.shape[0])).astype(np.float32)

        # Composição por alpha blending
        blended = alpha * foreground + (1 - alpha) * background
        new_image = blended.astype(np.uint8)

        if debug:
            self.show_image("Imagem final com novo fundo", new_image)

        return new_image

def main():
    img_process = image_process()
    img = img_process.read_image(PATHS[2])
    new_background = img_process.read_image(PATHS[9])

    foreground, alpha = img_process.extract_foreground(img)
    img_process.assble_new_image(foreground, alpha, new_background)

if __name__ == "__main__":
    main()
