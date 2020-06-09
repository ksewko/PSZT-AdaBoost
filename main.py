from time import time

import numpy as np
from AdaBoost import AdaBoost
from scikt import testv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():

    #wczytanie danych
    test = np.loadtxt('test.txt', delimiter=",")
    x = test[:, 0:13]
    train = np.loadtxt('train.txt', delimiter=",")
    y = train[:, 8]

    #przygotowanie danych
    xTrenuj, xTestuj, yTrenuj, yTestuj = train_test_split(x, y, test_size=0.2, random_state=0)

    #przygotoanie danych do testu i do trenowania
    xTrenuj=xTrenuj.transpose()
    yTrenuj[yTrenuj == 1] = 1
    yTrenuj[yTrenuj == 0] = -1

    xTestuj=xTestuj.transpose()
    yTestuj[yTestuj == 1] = 1
    yTestuj[yTestuj == 0] = -1

    # train
    adaBoost=AdaBoost(xTrenuj, yTrenuj)
    start_trenuj = time()
    adaBoost.trenuj(16)
    end_trenuj = time()
    total_trenuj = end_trenuj - start_trenuj

    # prognoza
    y_pred = adaBoost.prognoza(xTestuj)

    print(f"Czas trenuj AdaBoost: {total_trenuj} s")
    print ("prognoza:", len(y_pred[y_pred == yTestuj]))
    print ("dokladnosc:", accuracy_score(yTestuj, y_pred))

    test_result = testv2()

if __name__=='__main__':
    main()