from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
from tests import get_results


def main():
    img = Image.open('../statics/img/img.png')
    gray = img.convert('L')
    gray = gray.resize((8,8), Image.LANCZOS)
    bw = gray.point(lambda x: 0 if x<210 else 255, '1')
    new_im = np.asarray(bw)
    metodos, predicciones,accs  = get_results(new_im.flatten())
    for metodo, prediccion, acc in zip(metodos, predicciones,accs):
        print '%s\t%s\t%s\t' % (metodo, prediccion, acc)




if __name__ == '__main__':
    main()