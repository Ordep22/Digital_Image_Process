import cv2 as cv
import numpy as np
import os


def analyze_and_count_grains(image_path, debug=False):
    """
    Versão V03.
    - Mantém a classificação detalhada entre grãos inteiros e outras partículas.
    - O objetivo final é que a SOMA de todas as partículas detectadas (inteiros + outros)
      se aproxime da contagem de referência.
    """
    # --- PARÂMETROS DE CLASSIFICAÇÃO (sem alteração) ---
    MIN_GRAIN_AREA = 80
    MAX_GRAIN_AREA = 450
    MIN_SOLIDITY = 0.94
    MIN_ASPECT_RATIO = 1.5
    MAX_ASPECT_RATIO = 4.0

    image = cv.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return 0, 0, 0

    filename = os.path.basename(image_path)
    expected_count = GROUND_TRUTH.get(filename, 0)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # --- PASSOS 1, 2, 3: Pré-processamento e Watershed (sem alteração) ---
    kernel_tophat = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel_tophat)
    _, thresh = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel_opening = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_opening, iterations=2)
    sure_bg = cv.dilate(opening, kernel_opening, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

    threshold_value = 0.45999 * dist_transform.max()
    _, sure_fg = cv.threshold(dist_transform, threshold_value, 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers)

    # --- PASSO 4: ANÁLISE E CLASSIFICAÇÃO (lógica interna sem alteração) ---
    total_objects_found = len(np.unique(markers)) - 1
    whole_grains = 0
    other_particles = 0

    result_image_debug = image.copy()

    for i in range(2, total_objects_found + 2):
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == i] = 255
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)

        if area < MIN_GRAIN_AREA:
            continue

        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        aspect_ratio = 0
        try:
            (x, y), (width, height), angle = cv.minAreaRect(c)
            if min(width, height) > 0:
                aspect_ratio = max(width, height) / min(width, height)
        except (cv.error, ValueError):
            continue

        is_whole_grain = (
                (MAX_GRAIN_AREA > area > MIN_GRAIN_AREA) and
                (solidity > MIN_SOLIDITY) and
                (MAX_ASPECT_RATIO > aspect_ratio > MIN_ASPECT_RATIO)
        )

        if is_whole_grain:
            whole_grains += 1
            cv.drawContours(result_image_debug, [c], -1, (0, 255, 0), 2)
        else:
            other_particles += 1
            if area >= MAX_GRAIN_AREA or solidity <= MIN_SOLIDITY:
                cv.drawContours(result_image_debug, [c], -1, (0, 255, 255), 2)
            else:
                cv.drawContours(result_image_debug, [c], -1, (0, 0, 255), 2)

    # --- Visualização de Debug (Ajustada para a nova métrica) ---
    if debug:
        # **MUDANÇA AQUI**: O total agora é a soma de tudo que foi detectado.
        total_detected = whole_grains + other_particles
        error_count = abs(expected_count - total_detected)
        error_percent = (error_count / expected_count) * 100 if expected_count > 0 else 0

        text_gt = f"Referencia: {expected_count}"
        text_total = f"Total Encontrado: {total_detected}"  # NOVA LINHA
        text_error = f"Erro Total: {error_count} graos ({error_percent:.2f}%)"  # ERRO BASEADO NO TOTAL

        cv.putText(result_image_debug, text_gt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(result_image_debug, text_total, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(result_image_debug, text_error, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.imshow(f"Analise - {filename}", result_image_debug)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return whole_grains, other_particles, expected_count


if __name__ == "__main__":
    # IMPORTANTE: Garanta que os caminhos para as imagens estão corretos.
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

    print("Iniciando analise de graos (V03 - Erro baseado na contagem total)...\n")

    total_absolute_error = 0
    total_expected_grains = 0

    for image_path in PATH_ARRAY:
        if not os.path.exists(image_path):
            print(f"AVISO: Imagem nao encontrada em '{image_path}'. Pulando.")
            continue

        filename = os.path.basename(image_path)

        whole, broken, expected = analyze_and_count_grains(image_path, debug=True)

        # **MUDANÇA AQUI**: O total encontrado é a soma das duas categorias.
        total_found = whole + broken

        # **MUDANÇA AQUI**: O erro é calculado com base no total encontrado.
        error = abs(total_found - expected)
        error_percent = (error / expected) * 100 if expected > 0 else 0

        total_absolute_error += error
        total_expected_grains += expected

        print(f"--- Imagem: {filename} ---")
        print(f"Contagem de Referencia: {expected}")
        print(f" -> Grãos Inteiros: {whole}")
        print(f" -> Outras Partículas: {broken}")
        print(f" -> TOTAL ENCONTRADO: {total_found}")
        print(f" -> ERRO: {error} ( {error_percent:.2f}% )")
        print("")

    average_error_rate = (total_absolute_error / total_expected_grains) * 100 if total_expected_grains > 0 else 0

    print(f"======================================")
    print(f"ANALISE COMPLETA")
    print(f"Erro Absoluto Total Acumulado: {total_absolute_error} graos")
    print(f"Taxa de Erro Media (baseado na contagem total): {average_error_rate:.2f}%")
    print(f"======================================")