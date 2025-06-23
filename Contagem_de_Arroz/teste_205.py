import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def contar_graos_watershed(image_path, min_area=88, max_area=1000, debug=False):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY_INV, 11, 6)
    cv.imshow("Thresh", thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Remover ruído com abertura morfológica
    kernel = np.ones((1, 1), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    cv.imshow("Opening", opening)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Background seguro
    kernel = np.ones((2, 2), np.uint8)
    sure_bg = cv.dilate(opening, kernel, iterations=1)
    cv.imshow("sure_bg", sure_bg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Distância para separar grãos
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    cv.imshow("dist_transform", dist_transform)
    cv.waitKey(0)
    cv.destroyAllWindows()
    _, sure_fg = cv.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)
    cv.imshow("sure_fg", sure_fg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Marcação
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv.watershed(image, markers)
    result = image.copy()
    result[markers == -1] = [0, 0, 255]  # contorno vermelho opcional
    cv.imshow("Result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Encontrar contornos nos marcadores segmentados
    mask = np.uint8(markers > 1) * 255
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if min_area < cv.contourArea(c) < max_area]

    for c in valid:
        cv.drawContours(result, [c], -1, (0, 255, 0), 1)

    if debug:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
        plt.title(f"{os.path.basename(image_path)} - Grãos detectados: {len(valid)}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return len(valid)

# Exemplo de uso
image_path = r"C:\Work\PyProjects\Teste\205.bmp"
total = contar_graos_watershed(image_path, debug=True)
print(f"Total de grãos detectados: {total}")
