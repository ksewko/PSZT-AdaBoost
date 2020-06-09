from time import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import numpy as np


# Wczytywanie danych
test = np.loadtxt('test.txt', delimiter=",")
X = test[:, 0:13]
train = np.loadtxt('train.txt', delimiter=",")
y = train[:, 8]
# Podzial zestawu danych na trenujące i testujace
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Tworzenie obiktu Adaboost
czas_start = time()
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Trenowanie obiektu Adaboost
model = abc.fit(X_train, y_train)
czas_end = time()
czas_total = czas_end - czas_start

#Prognoza odpoweidzi dla zestawu danych
y_pred = model.predict(X_test)
print(f"Czas trenuj AdaBoost: {czas_total} s")
print("Dokladnosc:",metrics.accuracy_score(y_test, y_pred))