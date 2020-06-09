import numpy as np
'''
Słaby klasyfikator algorytmu jednowarstwowego drzewa decyzyjnego
'''
class DrzewoDecyzyjne:
    def __init__(self,X,y):
        self.X=np.array(X)
        self.y=np.array(y)
        self.N=self.X.shape[0]
    
    def trenuj(self, W, steps=100):  #Zwraca najniższy próg spośród wszystkich parametrów
        '''    
        W jest wektorem o długości N, który reprezentuje masę N próbek
        threshold_value- Próg
        threshold_pos- Pozycja progu Przez kilka pierwszych parametrów
        threshold_tag- Wynosi 1 lub -1. Jeśli jest większy niż próg, jest dzielony na próg_tag, jeśli
        jest mniejszy niż próg, odwrotnie
        '''
        min = float("inf")    #Zainicjuj min do nieskończoności
        progWartosc=0;
        progPozycja=0;
        progFlaga=0;
        self.W=np.array(W)
        for i in range(self.N):  #  wartosc- wskazuje próg, liczbaBledow wskazuje liczbę błędów
            wartosc, liczbaBledow = self.znajdzMinimalny(i, 1, steps)
            if (liczbaBledow < min):
                min = liczbaBledow
                progWartosc = wartosc
                progPozycja = i
                progFlaga = 1
        for i in range(self.N):  # -1
            wartosc, liczbaBledow= self.znajdzMinimalny(i, -1, steps)
            if (liczbaBledow < min):
                min = liczbaBledow
                progWartosc = wartosc
                progPozycja = i
                progFlaga = -1
        #Ostatnia aktualizacja
        self.progWartosc=progWartosc
        self.progPozycja=progPozycja
        self.progFlaga=progFlaga
        print(self.progWartosc, self.progPozycja, self.progFlaga)
        return min
    
    def znajdzMinimalny(self, i, flaga, kroki):  #Znajdź minimalny próg i-tego parametru, flaga progu wynosi 1 lub -1
        j = 0
        pom = self.prognozaTrening(self.X, i, j, flaga).transpose()
        liczbaBledow = np.sum((pom != self.y) * self.W)
        #print now
        dolnaGranica=np.min(self.X[i, :])  #Minimalna wartość tego atrybutu, dolna granica
        gornaGranica=np.max(self.X[i, :])      #Maksymalna wartość tego atrybutu, górna granica
        minerr = float("inf")       #Zainicjuj minerr do nieskończoności
        wartosc=0                     #value reprezentuje próg
        st= (gornaGranica - dolnaGranica) / kroki        #interwał
        for j in np.arange(dolnaGranica, gornaGranica, st):
            pom = self.prognozaTrening(self.X, i, j, flaga).transpose()
            liczbaBledow = np.sum((pom != self.y) * self.W)
            if liczbaBledow < minerr:
                minerr=liczbaBledow
                wartosc=j
        return wartosc, minerr
    
    def prognozaTrening(self, testSet, i, j, tag): #Wytypuj wynik podczas treningu zgodnie z wartością progową j
        testSet=np.array(testSet).reshape(self.N, -1)
        prognoza_y = np.ones((np.array(testSet).shape[1], 1))
        prognoza_y[testSet[i, :] * tag < j * tag]=-1
        return prognoza_y

    def prognoza(self, testX):  #Słaba prognoza klasyfikatora
        testX=np.array(testX).reshape(self.N, -1) #Przekształcone na N wierszy i X kolumn, -1 jest zbyt leniwe, aby liczyć
        prognoza_y = np.ones((np.array(testX).shape[1], 1))
        prognoza_y[testX[self.progPozycja, :] * self.progFlaga < self.progWartosc * self.progFlaga]=-1
        return prognoza_y
	