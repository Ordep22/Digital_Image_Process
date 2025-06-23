import cv2 as cv
import numpy as np
import os


def contar_graos_definitivo(image_path, debug=False):
    """
    Conta grãos de arroz com máxima precisão usando uma abordagem híbrida:
    - Top-hat + Otsu para uma binarização robusta.
    - Distance Transform + Watershed para separar grãos agrupados.
    """
    image = cv.imread(image_path)
    filename = os.path.basename(image_path)
    expected_count = GROUND_TRUTH.get(filename, 0)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return 0

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Etapa 1: Pré-processamento robusto
    kernel_tophat = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel_tophat)
    _, thresh = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Etapa 2: Limpeza e preparação para o Watershed
    kernel_opening = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_opening, iterations=2)
    sure_bg = cv.dilate(opening, kernel_opening, iterations=3)

    # Etapa 3: Encontrar marcadores com o ajuste final
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

    # ***** ALTERAÇÃO CHAVE AQUI *****
    # Diminuímos o multiplicador para criar mais marcadores e separar aglomerados densos.
    threshold_value = 0.45999 * dist_transform.max()
    _, sure_fg = cv.threshold(dist_transform, threshold_value, 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Etapa 4: Aplicar o Watershed
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers)

    count = len(np.unique(markers)) - 1

    if debug:
        result_image = image.copy()
        result_image[markers == -1] = [0, 0, 255]
        filename = os.path.basename(image_path)
        text_defaut_grains = f"Total grains in to the image: {expected_count}"
        text_couted_grain = f"Graos contados: {count}"
        text_percent_error = f"Error: {(abs(count- expected_count)/expected_count)*100:.2f} %"
        cv.putText(result_image, text_defaut_grains, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(result_image, text_couted_grain, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(result_image, text_percent_error, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.imshow("Resultado Final com Ajuste Fino", result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return count


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

    print("Iniciando contagem de grãos - Calculando Erro Percentual Geral...\n")

    total_absolute_error = 0
    total_expected_grains = 0

    for image_path in PATH_ARRAY:
        filename = os.path.basename(image_path)
        # Supondo que a função 'contar_graos_definitivo' existe e foi definida anteriormente
        detected_count = contar_graos_definitivo(image_path, debug=True)
        expected_count = GROUND_TRUTH.get(filename, 0)

        # Calcula o erro em número de grãos (erro absoluto)
        absolute_error = abs(detected_count - expected_count)

        # Acumula os erros absolutos e os totais esperados
        total_absolute_error += absolute_error
        total_expected_grains += expected_count

        print(f"Imagem: {filename}")
        print(f" -> Contagem esperada: {expected_count}")
        print(f" -> Grãos detectados: {detected_count}")
        print(f" -> Erro absoluto (grãos): {absolute_error}")
        # Esta é a sua excelente adição para detalhar o erro por imagem
        print(f" -> Erro percentual da imagem: {(absolute_error / expected_count) * 100:.2f} %\n")

    # Calcula o erro percentual geral com base nos totais
    overall_percentage_error = (total_absolute_error / total_expected_grains) * 100

    print(f"=============================================")
    print(f"Total de Grãos Esperados: {total_expected_grains}")
    print(f"Total de Erros (em grãos): {total_absolute_error}")
    print(f"Erro Percentual Geral (Total): {overall_percentage_error:.2f} %")
    print(f"=============================================")