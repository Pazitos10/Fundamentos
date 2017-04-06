from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split
from classifiersTest import get_results


def main():
    digits = datasets.load_digits()
    
    im = Image.open('static/img/img.png')
    im = im.convert('L')
    im.thumbnail((8,8))
    im_a = np.asarray(im, dtype=float)
    
    inv = ImageOps.invert(im)
    inv.save('invert.png')
    im_a = np.asarray(inv, dtype=float)
    
    im_a = list(map((lambda x: (x//16)+1 ), im_a))
    im_a = np.asarray(im_a)

    #otra = Image.fromarray(res, mode='L')
    #otra.save('otra.png')
    
    #im = Image.fromarray(res, 'L')

    
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    
    cl1 = svm.SVC(gamma=0.001, probability=True)
    
    cl2 = KNeighborsClassifier(n_neighbors=9, algorithm='auto')
    cl3 = ExtraTreesClassifier(max_features='auto',
                                     n_jobs=-1,
                                     random_state=1)
    cl4 = DummyClassifier(strategy='most_frequent',random_state=np.random.randint(0,9))

    cls = [cl1,cl2,cl3,cl4]

    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.1)


    cls = list(map((lambda cl: cl.fit(X_train, y_train)), cls))
    expected = y_test
    cls = list(map((lambda cl: pred(cl, im_a.flatten(), digits.images[2].flatten())), cls))

    #print(res) #np.asarray(im_a2,dtype=float)
    print(im_a)
    print(digits.images[2])

    print_plots(digits, im_a)


def pred(cl, data, ds_digit):
    print("="*20)
    print(cl.__class__.__name__)
    results = cl.predict(data.reshape(1, -1))[0]
    #probs = {"prob_" + str(i) : prob for i, prob in enumerate(cl.predict_proba(data.reshape(1, -1))[0])}
    #print("parece un: %s con %s" % (results, probs))
    print("parece un: %s" % (results))

    results = cl.predict(ds_digit.reshape(1, -1))[0]
    #probs = {"prob_" + str(i) : prob for i, prob in enumerate(cl.predict_proba(ds_digit.reshape(1, -1))[0])}
    #print("ds_digit parece un: %s con %s" % (results, probs))
    print("ds_digit parece un: %s " % (results))

def print_plots(digits, res):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(res, cmap = plt.get_cmap('gray'))
    ax2.imshow(digits.images[1], cmap= plt.get_cmap('gray'))
    plt.show()

if __name__ == '__main__':
    main()
