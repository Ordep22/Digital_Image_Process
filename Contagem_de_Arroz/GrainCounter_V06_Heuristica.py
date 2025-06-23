import cv2 as cv
import numpy as np
import os


def contar_graos_definitivo(image_path, debug=False):
    """
    Versão 06 - Abordagem final com heurística de área.
    1. Segmenta os grãos com o pipeline V03.
    2. Identifica grãos "perfeitos" para calcular uma área média por imagem.
    3. Usa a área média para estimar quantos grãos existem em aglomerados maiores.
    """
    # --- Parâmetros de Classificação para identificar grãos de referência ---
    MIN_GRAIN_AREA = 80
    MAX_GRAIN_AREA = 450
    MIN_SOLIDITY = 0.94
    MIN_ASPECT_RATIO = 1.5
    MAX_ASPECT_RATIO = 4.0
    DEFAULT_AVG_AREA = 150  # Fallback caso nenhum grão perfeito seja encontrado

    image = cv.imread(image_path)
    filename = os.path.basename(image_path)
    expected_count = GROUND_TRUTH.get(filename, 0)
    if image is None: return 0

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # --- ETAPA 1 e 2: Pré-processamento e Segmentação (Baseado na V03) ---
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

    # --- ETAPA 3: Pós-processamento com Heurística de Área (A NOVA LÓGICA) ---

    # 3.1: PASSO DE CALIBRAÇÃO - Encontrar a área média de um grão perfeito
    good_grains_contours = []
    cluster_contours = []

    for i in range(2, len(np.unique(markers))):
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == i] = 255
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts: continue

        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)

        if area < MIN_GRAIN_AREA: continue  # Ignora ruídos

        hull = cv.convexHull(c)
        solidity = float(area) / cv.contourArea(hull) if cv.contourArea(hull) > 0 else 0

        aspect_ratio = 0
        try:
            (x, y), (w, h), angle = cv.minAreaRect(c)
            if min(w, h) > 0:
                aspect_ratio = max(w, h) / min(w, h)
        except (cv.error, ValueError):
            continue

        # Verifica se é um grão "perfeito"
        if (MAX_GRAIN_AREA > area and solidity > MIN_SOLIDITY and
                MAX_ASPECT_RATIO > aspect_ratio > MIN_ASPECT_RATIO):
            good_grains_contours.append(c)
        else:  # Se não for perfeito (e não for ruído), é um aglomerado
            cluster_contours.append(c)

    # Calcula a área média dos grãos perfeitos encontrados nesta imagem
    avg_grain_area = 0
    if len(good_grains_contours) > 0:
        total_area_good_grains = sum(cv.contourArea(c) for c in good_grains_contours)
        avg_grain_area = total_area_good_grains / len(good_grains_contours)
    else:
        # Se nenhum grão perfeito for encontrado, usa um valor padrão
        avg_grain_area = DEFAULT_AVG_AREA

    # 3.2: PASSO DE CONTAGEM FINAL - Estimar grãos em aglomerados
    final_grain_count = len(good_grains_contours)  # Começa com os grãos perfeitos

    for c in cluster_contours:
        cluster_area = cv.contourArea(c)
        # Estima quantos grãos cabem na área do aglomerado
        estimated_grains = round(cluster_area / avg_grain_area)
        # Adiciona à contagem (garante que conte pelo menos 1)
        final_grain_count += max(1, estimated_grains)

    if debug:
        result_image = image.copy()
        # Desenha os grãos perfeitos em verde
        cv.drawContours(result_image, good_grains_contours, -1, (0, 255, 0), 2)
        # Desenha os aglomerados em amarelo
        cv.drawContours(result_image, cluster_contours, -1, (0, 255, 255), 2)

        text_couted_grain = f"Graos contados (est): {final_grain_count}"
        text_percent_error = f"Error (est): {(abs(final_grain_count - expected_count) / expected_count) * 100:.2f} %"
        cv.putText(result_image, text_couted_grain, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(result_image, text_percent_error, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.imshow(f"V06 - Heuristica de Area", result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return final_grain_count


if __name__ == "__main__":
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

    print("Iniciando contagem de grãos (V06 - Heurística de Área)...\n")

    total_absolute_error = 0
    total_expected_grains = 0

    for image_path in PATH_ARRAY:
        filename = os.path.basename(image_path)
        detected_count = contar_graos_definitivo(image_path, debug=True)
        expected_count = GROUND_TRUTH.get(filename, 0)

        absolute_error = abs(detected_count - expected_count)

        total_absolute_error += absolute_error
        total_expected_grains += expected_count

        print(f"Imagem: {filename}")
        print(f" -> Contagem esperada: {expected_count}")
        print(f" -> Grãos detectados (estimado): {detected_count}")
        print(f" -> Erro absoluto (grãos): {absolute_error}")
        if expected_count > 0:
            print(f" -> Erro percentual da imagem: {(absolute_error / expected_count) * 100:.2f} %\n")
        else:
            print("\n")

    overall_percentage_error = (total_absolute_error / total_expected_grains) * 100

    print(f"=============================================")
    print(f"Total de Grãos Esperados: {total_expected_grains}")
    print(f"Total de Erros (em grãos): {total_absolute_error}")
    print(f"Erro Percentual Geral (Total): {overall_percentage_error:.2f} %")
    print(f"=============================================")