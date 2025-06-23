import os
import time
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Caminhos das imagens
# ATENÇÃO: Certifique-se de que estes caminhos estão corretos para o seu ambiente!
PATH_ARRAY = [
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\60.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\82.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\114.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\150.bmp",
    r"C:\Users\Pedro Pereira\Documents\Git\Digital_Image_Process\Contagem_de_Arroz\Image\205.bmp"
]

# Valores esperados (Ground Truth) para cada imagem
GROUND_TRUTH = {
    "60.bmp": 60,
    "82.bmp": 82,
    "114.bmp": 114,
    "150.bmp": 150,
    "205.bmp": 205
}


class HandleImage:
    def __init__(self):
        pass

    def read_image(self, path):
        """
        Lê uma imagem em escala de cinza de um determinado caminho.
        """
        self.img = cv.imread(cv.samples.findFile(path), cv.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f'Image not found at {path}')
        # Garante que a imagem seja do tipo np.uint8 para funções OpenCV
        self.img = self.img.astype(np.uint8)

    def process_image(self, img, gaussian_blur_sigma, adaptive_thresh_block_size, C, min_area_px=100, max_area_px=10000):
        """
        Processa uma imagem para contar grãos de arroz usando thresholding adaptativo e watershed.

        Args:
            img (np.array): A imagem de entrada em escala de cinza.
            gaussian_blur_sigma (float): Valor de sigma para o desfoque Gaussiano.
            adaptive_thresh_block_size (int): Tamanho do bloco para thresholding adaptativo (deve ser ímpar).
            C (int): Constante subtraída da média ou média ponderada.
            min_area_px (int): Área mínima para um grão detectado (em pixels).
            max_area_px (int): Área máxima para um grão detectado (em pixels).

        Returns:
            tuple: (imagem_threshold, imagem_opening, contornos, contagem)
        """
        # 1. Pré-processamento: Desfoque Gaussiano e Thresholding Adaptativo
        # O desfoque Gaussiano ajuda a reduzir o ruído e suavizar a imagem.
        # Kernel (25, 25) para melhor suavização.
        blured = cv.GaussianBlur(img, (5, 5), sigmaX=gaussian_blur_sigma)
        # O thresholding adaptativo lida com variações de iluminação.
        thresh = cv.adaptiveThreshold(
            blured, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV, adaptive_thresh_block_size, C
        )

        # 2. Operações Morfológicas
        # Define um kernel para as operações morfológicas.
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # 'Opening' remove pequenos objetos e ruídos, e separa objetos que estão tocando.
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

        # 3. Fundo Certo (Sure Background) e Primeiro Plano Certo (Sure Foreground) para Watershed
        # O 'fundo certo' é obtido dilatando o resultado do 'opening'.
        sure_bg = cv.dilate(opening, kernel, iterations=3)

        # A transformada de distância encontra a distância de cada pixel até o pixel zero mais próximo.
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        # Threshold na transformada de distância para obter o primeiro plano certo.
        _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 4. Região Desconhecida
        # A região desconhecida é a diferença entre o fundo certo e o primeiro plano certo.
        unknown = cv.subtract(sure_bg, sure_fg)

        # 5. Preparação dos Marcadores para o Watershed
        # Rotula os componentes conectados no primeiro plano certo.
        # Pixels de fundo são 0, outros objetos são 1, 2, ...
        _, markers = cv.connectedComponents(sure_fg)
        # Adiciona 1 a todos os rótulos para garantir que o fundo seja 1 (e não 0, que é geralmente para o desconhecido).
        markers += 1
        # Marca as regiões desconhecidas como 0.
        markers[unknown == 255] = 0

        # 6. Algoritmo Watershed
        # Converte a imagem em escala de cinza para BGR para desenhar as bordas coloridas do watershed.
        color_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        # Aplica o algoritmo watershed. Os marcadores são atualizados no local.
        # As fronteiras são marcadas com -1.
        markers = cv.watershed(color_img, markers)

        # 7. Contagem dos Grãos e Extração dos Contornos
        count = 0
        rice_contours = []
        # Percorre todos os rótulos únicos encontrados pelo watershed.
        # Exclui o fundo (1) e as bordas (-1).
        for label in np.unique(markers):
            if label == -1 or label == 1:  # Exclui bordas e fundo
                continue

            # Cria uma máscara para a região rotulada atual.
            mask = np.zeros(img.shape, dtype="uint8")
            mask[markers == label] = 255

            # Encontra os contornos para a região atual.
            cnts, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if len(cnts) > 0:
                # Assume que cada rótulo deve corresponder a um contorno para um grão.
                # Pega o maior contorno se múltiplos forem encontrados para um único rótulo.
                c = max(cnts, key=cv.contourArea)
                area = cv.contourArea(c)

                # Filtra os contornos com base na área para excluir ruídos ou grãos mesclados.
                # Condição de filtro agora inclui min_area_px E max_area_px
                if min_area_px < area < max_area_px:
                    rice_contours.append(c)
                    count += 1

        return thresh, opening, rice_contours, count

    def define_parameters(self):
        """
        Itera por diferentes combinações de parâmetros para encontrar as melhores configurações.
        """
        results = []
        # Defina um intervalo de parâmetros para teste.
        # Estes são os valores que você mencionou na sua pergunta anterior.
        sigmas = [5.0, 7.0, 9.0]
        block_sizes = [15, 21, 27]  # Deve ser ímpar
        Cs = [-10, -5, 0, 5]

        # Ajuste min_area_px e max_area_px para o tamanho provável dos seus grãos de arroz.
        # Estes são valores iniciais, você pode precisar ajustá-los.
        initial_min_area_px = 30
        initial_max_area_px = 10000

        print("Iniciando a busca de parâmetros... Isso pode levar alguns minutos.")
        start_time = time.time()

        for path in PATH_ARRAY:
            file_name = os.path.basename(path)
            self.read_image(path)
            original = self.img

            for sigma in sigmas:
                for block_size in block_sizes:
                    for C_val in Cs:
                        thresh, opening, contours, count = self.process_image(
                            original, sigma, block_size, C_val,
                            min_area_px=initial_min_area_px,
                            max_area_px=initial_max_area_px
                        )
                        results.append({
                            "image": file_name,
                            "sigma": sigma,
                            "block_size": block_size,
                            "C": C_val,
                            "count": count
                        })
        end_time = time.time()
        print(f"Busca de parâmetros concluída em {end_time - start_time:.2f} segundos.")

        # Construção do DataFrame
        df = pd.DataFrame(results)
        df["ground_truth"] = df["image"].map(GROUND_TRUTH)
        df["error"] = (df["count"] - df["ground_truth"]).abs()
        df["acerto"] = df["count"] == df["ground_truth"]
        df["acerto_tolerancia_2"] = (df["count"] - df["ground_truth"]).abs() <= 2

        # Agrupamento por parâmetros para encontrar os melhores em termos de CONSISTÊNCIA
        grouped_tol = df.groupby(["sigma", "block_size", "C"])["acerto_tolerancia_2"].sum().reset_index()
        grouped_tol["param"] = grouped_tol[["sigma", "block_size", "C"]].astype(str).agg("_".join, axis=1)

        # Plotar o número de acertos com tolerância para visualização
        plt.figure(figsize=(12, 6))
        plt.bar(grouped_tol["param"], grouped_tol["acerto_tolerancia_2"], color='lightgreen', edgecolor='black')
        plt.xticks(rotation=90)
        plt.title("Número de acertos com tolerância ±2 por combinação de parâmetros")
        plt.ylabel("Número de acertos (0 a 5)") # 5 é o número total de imagens no PATH_ARRAY
        plt.xlabel("Parâmetros (sigma_block_C)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Encontrar a(s) combinação(ões) de parâmetros com o maior número de acertos consistentes
        max_acertos_consistent = grouped_tol["acerto_tolerancia_2"].max()
        self.best_consistent_params = grouped_tol[grouped_tol["acerto_tolerancia_2"] == max_acertos_consistent]
        print("\nCombinação(ões) de parâmetros com o MAIOR número de acertos (tolerância ±2) em todas as imagens:")
        print(self.best_consistent_params)

        # Salvando resultados (opcional)
        df.to_csv("resultado_analise_parametros.csv", index=False)


def main():
    handle_image = HandleImage()

    # --- PASSO 1: ENCONTRAR PARÂMETROS ÓTIMOS (DESCOMENTE PARA EXECUTAR) ---
    # É ABSOLUTAMENTE CRÍTICO EXECUTAR ESTA FUNÇÃO Pelo menos uma vez
    # após qualquer mudança no código de processamento da imagem,
    # para que os parâmetros ótimos sejam recalculados.
    #handle_image.define_parameters()

    # --- PASSO 2: APLICAR PARÂMETROS ÓTIMOS A UMA IMAGEM ESPECÍFICA ---
    # Uma vez que você tenha rodado define_parameters() acima,
    # olhe a saída no console sob "Combinação(ões) de parâmetros com o MAIOR número de acertos..."
    # e use esses valores para definir as variáveis abaixo.

    # SUBSTITUA ESTES VALORES pelos parâmetros que você encontrou como os melhores
    # para a CONSISTÊNCIA (maior 'acerto_tolerancia_2' total).
    optimal_sigma = 7.0       # Exemplo, ajuste com base na sua análise
    optimal_block_size = 21   # Exemplo, ajuste com base na sua análise (deve ser ímpar)
    optimal_C = 0             # Exemplo, ajuste com base na sua análise
    optimal_min_area = 30     # Ajuste este valor se grãos muito pequenos forem contados como ruído
    optimal_max_area = 10000  # Ajuste este valor se grãos mesclados ou objetos grandes forem contados

    # Escolha uma imagem para testar (ex: 60.bmp, 82.bmp, etc.)
    # Você pode alterar o índice para testar outras imagens do PATH_ARRAY
    image_path_to_process = PATH_ARRAY[1] # Testando com a primeira imagem (60.bmp)
    file_name = os.path.basename(image_path_to_process)
    ground_truth_count = GROUND_TRUTH.get(file_name, "N/A")

    print(f"\nProcessando imagem: {file_name} com parâmetros selecionados:")
    print(f"  Sigma={optimal_sigma}, Block Size={optimal_block_size}, C={optimal_C}")
    print(f"  Min Area={optimal_min_area}, Max Area={optimal_max_area}")

    handle_image.read_image(image_path_to_process)
    original_image = handle_image.img

    # Processa a imagem com os parâmetros escolhidos
    thresh_img, opening_img, rice_contours, count = handle_image.process_image(
        original_image,
        optimal_sigma,
        optimal_block_size,
        optimal_C,
        min_area_px=optimal_min_area,
        max_area_px=optimal_max_area
    )

    # --- Visualização dos Resultados ---
    # Converte a imagem original para BGR para desenhar contornos coloridos
    annotated_image = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
    cv.drawContours(annotated_image, rice_contours, -1, (0, 255, 0), 2)  # Desenha contornos verdes

    # Exibe a contagem detectada e o valor real
    cv.putText(annotated_image, f"Detectados: {count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(annotated_image, f"Valor Real: {ground_truth_count}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.putText(annotated_image, f"Imagem: {file_name}", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv.imshow("Imagem Threshold", thresh_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow("Imagem Opening", opening_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow("Graos de Arroz Anotados", annotated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()