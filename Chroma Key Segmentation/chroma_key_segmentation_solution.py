import cv2 as cv
import numpy as np
DIR_PATH = r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\Result\\"
PATHS = [
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\0.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\1.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\2.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\3.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\4.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\5.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\6.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\7.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\8.bmp"
]
PATH_NEW_BACKGROUND = [r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Chroma Key Segmentation\img\9.png"]

class ImageProcess:
    def __init__(self):
        pass

    def read_image(self, img_path):
        return cv.imread(img_path)

    def show_image(self, title_window, img):
        cv.imshow(title_window, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def save_image(self, img, title):
        img = (img).astype(np.uint8)
        cv.imwrite(DIR_PATH + title + ".png", img)

    def extract_foreground(self, img, debug=True):
        img_blurred = cv.blur(img, ksize=(5, 5))

        # Separação dos canais
        b = img_blurred[:, :, 0]
        g = img_blurred[:, :, 1]
        r = img_blurred[:, :, 2]

        # Threshold adaptativo
        offset_r, offset_b = self.compute_dynamic_threshold(img)
        mask_logic = (g > r + offset_r) & (g > b + offset_b)

        # Máscara binária inicial
        green_mask = np.zeros_like(g, dtype=np.uint8)
        green_mask[mask_logic] = 255

        # Inverter para capturar apenas o objeto (foreground)
        foreground_mask = cv.bitwise_not(green_mask)

        # Refinar a máscara com método modular
        refined_mask = self.refine_mask(
            binary_mask=foreground_mask,
            open_close_kernel=(5, 5),
            gaussian_ksize=(7, 7),
            erode_dilate=True,
            debug=debug
        )

        # Criar máscara alpha
        alpha = refined_mask.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        if debug:
            self.show_image("Foreground Mask (refinada)", refined_mask)

        return img, alpha

    def refine_mask(self, binary_mask, open_close_kernel=(3, 3), gaussian_ksize=(3, 3), erode_dilate=False, debug=True):
        kernel = np.ones(open_close_kernel, np.uint8)

        # Abrir para remover ruído e fechar para preencher buracos
        refined = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)
        refined = cv.morphologyEx(refined, cv.MORPH_CLOSE, kernel)

        if erode_dilate:
            refined = cv.erode(refined, kernel, iterations=1)
            refined = cv.dilate(refined, kernel, iterations=2)

        refined = cv.GaussianBlur(refined, gaussian_ksize, 0)

        if debug:
            self.show_image("Refined Mask", refined)

        return refined

    def compute_dynamic_threshold(self, img):
        g = img[:, :, 1].flatten()
        r = img[:, :, 2].flatten()
        b = img[:, :, 0].flatten()

        g_ref = np.median(g)
        offset_r = max(g_ref - np.max(r), 20)
        offset_b = max(g_ref - np.max(b), 20)

        return offset_r, offset_b

    def assble_new_image(self, foreground, alpha, background, debug=True):
        foreground = foreground.astype(np.float32)
        background = cv.resize(background, (foreground.shape[1], foreground.shape[0])).astype(np.float32)

        blended = alpha * foreground + (1 - alpha) * background
        new_image = blended.astype(np.uint8)

        if debug:
            self.show_image("Imagem final com novo fundo", new_image)

        return new_image


def main():
    process = ImageProcess()

    for path in PATHS[:-1]:
        print(f"Processando imagem {path[len(path)-5:]}")
        img = process.read_image(path)
        foreground, alpha = process.extract_foreground(img, debug=True)
        new_image  = process.assble_new_image(foreground, alpha, PATH_NEW_BACKGROUND, debug=True)
        process.save_image(new_image,title = "result_image_" + path[len(path)-5:] )
if __name__ == "__main__":
    main()
