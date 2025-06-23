import cv2 as cv
import numpy as np
import os


def analyze_and_count_grains(image_path, debug=False):
    """
    Versão V05 - Abordagem Final com Técnica Clássica.
    - Utiliza um método ADAPTATIVO para encontrar os marcadores do Watershed.
    - Aplica o limiar de Otsu na imagem de Transformada de Distância para
      determinar os marcadores de forma automática para cada imagem.
    """
    # --- Parâmetros de Classificação (revertendo MIN_GRAIN_AREA) ---
    MIN_GRAIN_AREA = 80  # Revertido para 80, pois o novo método de marcadores é mais estável.
    MAX_GRAIN_AREA = 450
    MIN_SOLIDITY = 0.94
    MIN_ASPECT_RATIO = 1.5
    MAX_ASPECT_RATIO = 4.0

    image = cv.imread(image_path)
    if image is None:
        return 0, 0, 0

    filename = os.path.basename(image_path)
    expected_count = GROUND_TRUTH.get(filename, 0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # --- Pré-processamento (Inalterado) ---
    kernel_tophat = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel_tophat)
    _, thresh = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel_opening = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_opening, iterations=2)
    sure_bg = cv.dilate(opening, kernel_opening, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

    # --- GERAÇÃO DE MARCADORES ADAPTATIVOS (A GRANDE MUDANÇA) ---
    # 1. Normalizar a imagem de distância para a faixa 0-255 para que possamos usar Otsu.
    dist_transform_norm = cv.normalize(dist_transform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

    # 2. Aplicar o limiar de Otsu para encontrar os marcadores de forma automática.
    # Esta é a etapa que substitui o `fator * dist_transform.max()`.
    _, sure_fg = cv.threshold(dist_transform_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # --- Watershed e Classificação (sem alterações na lógica) ---
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers)

    # O restante do código de análise, debug e main permanece idêntico ao da V04.
    whole_grains = 0
    other_particles = 0
    result_image_debug = image.copy()
    for i in range(2, len(np.unique(markers))):
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == i] = 255
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)
        if area < MIN_GRAIN_AREA: continue
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        aspect_ratio = 0
        try:
            (x, y), (w, h), angle = cv.minAreaRect(c)
            if min(w, h) > 0:
                aspect_ratio = max(w, h) / min(w, h)
        except (cv.error, ValueError):
            continue
        is_whole_grain = ((MAX_GRAIN_AREA > area > MIN_GRAIN_AREA) and
                          (solidity > MIN_SOLIDITY) and
                          (MAX_ASPECT_RATIO > aspect_ratio > MIN_ASPECT_RATIO))
        if is_whole_grain:
            whole_grains += 1
            cv.drawContours(result_image_debug, [c], -1, (0, 255, 0), 2)
        else:
            other_particles += 1
            cv.drawContours(result_image_debug, [c], -1, (0, 0, 255), 2)
    if debug:
        total_detected = whole_grains + other_particles
        error_count = abs(expected_count - total_detected)
        error_percent = (error_count / expected_count) * 100 if expected_count > 0 else 0
        text_gt = f"Referencia: {expected_count}"
        text_total = f"Total Encontrado: {total_detected}"
        text_error = f"Erro Total: {error_count} graos ({error_percent:.2f}%)"
        cv.putText(result_image_debug, text_gt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(result_image_debug, text_total, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(result_image_debug, text_error, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.imshow(f"Analise - {filename}", result_image_debug)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return whole_grains, other_particles, expected_count


if __name__ == "__main__":
    # O bloco principal para executar os testes permanece o mesmo.
    PATH_ARRAY = [
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\60.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\82.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\114.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\150.bmp",
        r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\205.bmp"
    ]
    GROUND_TRUTH = {
        "60.bmp": 60, "82.bmp": 82, "114.bmp": 114, "150.bmp": 150, "205.bmp": 205
    }
    print("Iniciando analise de graos (V05 - Marcadores Adaptativos com Otsu)...")
    # ... (o resto do loop de teste é igual ao da V04)
    total_absolute_error = 0
    total_expected_grains = 0
    for image_path in PATH_ARRAY:
        if not os.path.exists(image_path): continue
        filename = os.path.basename(image_path)
        whole, broken, expected = analyze_and_count_grains(image_path, debug=True)
        total_found = whole + broken
        error = abs(total_found - expected)
        error_percent = (error / expected) * 100 if expected > 0 else 0
        total_absolute_error += error
        total_expected_grains += expected
        print(f"\n--- Imagem: {filename} ---")
        print(f"Contagem de Referencia: {expected}")
        print(f" -> TOTAL ENCONTRADO: {total_found}")
        print(f" -> ERRO: {error} ( {error_percent:.2f}% )")
    average_error_rate = (total_absolute_error / total_expected_grains) * 100 if total_expected_grains > 0 else 0
    print(f"\n======================================")
    print(f"ANALISE COMPLETA")
    print(f"Erro Absoluto Total Acumulado: {total_absolute_error} graos")
    print(f"Taxa de Erro Media: {average_error_rate:.2f}%")
    print(f"======================================")