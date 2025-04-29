#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
# Adaptado por: Pedro Pereira
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

NEGATIVO = False
THRESHOLD = 0.75
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 1
INPUT_IMAGE = r''

components = []

#===============================================================================
class ManageStack():

    def __init__(self):
        self.stack = []

    def push(self, pixel_coordinate):
        self.stack.append(pixel_coordinate)

    def pop(self):
        if not self.isEmpty():
            return self.stack.pop()
        else:
            return False

    def peek(self):
        if self.isEmpty():
            return False
        else:
            return self.stack[self.size() - 1]

    def isEmpty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

mngStack = ManageStack()

def FloodFill(label_position, img, x0, y0):
    ''' Implementa flood fill usando pilha. Marca os pixels com o valor do label. '''
    mngStack.push([x0, y0])
    top, left = x0, y0
    bottom, right = x0, y0
    number_pixels = 0

    while not mngStack.isEmpty():
        x, y = mngStack.pop()
        if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
            if img[x][y] == 1:
                img[x][y] = label_position
                number_pixels += 1

                top = min(top, x)
                bottom = max(bottom, x)
                left = min(left, y)
                right = max(right, y)

                mngStack.push([x + 1, y])
                mngStack.push([x - 1, y])
                mngStack.push([x, y + 1])
                mngStack.push([x, y - 1])

    largura = right - left + 1
    altura = bottom - top + 1

    if number_pixels >= N_PIXELS_MIN and largura >= LARGURA_MIN and altura >= ALTURA_MIN:
        components.append(
            {
                'Label': label_position,
                'n_pixels': number_pixels,
                'T': top,
                'L': left,
                'B': bottom,
                'R': right
            }
        )
#===============================================================================
def binariza(img, threshold):
    ''' Binarização simples por limiarização. '''
    return np.where(img >= threshold, 1, 0).astype('uint16')
#-------------------------------------------------------------------------------

def rotula(img, largura_min, altura_min, n_pixels_min, label=2):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores únicos.'''
    components.clear()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 1:
                FloodFill(label, img, x, y)
                label += 1
    return components
#===============================================================================
def main():
    img = cv2.imread(cv2.samples.findFile(INPUT_IMAGE), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    img = img.reshape((img.shape[0], img.shape[1], 1))
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.astype(np.float32) / 255

    if NEGATIVO:
        img = 1 - img

    img_bin = binariza(img, THRESHOLD)
    cv2.imshow('01 - binarizada',(img_bin * 255).astype(np.uint8))
    cv2.imwrite('01 - binarizada.png', (img_bin * 255).astype(np.uint8))

    start_time = timeit.default_timer()
    componentes = rotula(img_bin, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    elapsed = timeit.default_timer() - start_time

    print('Tempo: %.4f segundos' % elapsed)
    print('%d componentes detectados.' % len(componentes))

    for c in componentes:
        cv2.rectangle(img_out, (c['L'], c['T']), (c['R'], c['B']), (0, 0, 255), 1)

    cv2.imshow('02 - out', img_out)
    cv2.imwrite('02 - out.png', img_out)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
#===============================================================================