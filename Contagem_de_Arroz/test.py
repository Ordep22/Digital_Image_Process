import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def contar_graos_simples(image_path, min_area=100, max_area=2000):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 35, 5)
    kernel = np.ones((3, 3), np.uint8)

    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    cnts, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if min_area < cv.contourArea(c) < max_area]
    result = image.copy()

    cv.drawContours(result, valid, -1, (0, 255, 0), 1)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title(f"{os.path.basename(image_path)} - Grãos detectados: {len(valid)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return len(valid)

num = contar_graos_simples(r"C:\Work\PyProjects\Teste\114.bmp")
print(f"Total de grãos detectados: {num}")