import numpy as np
from SlabyKlasyfikator import DrzewoDecyzyjne
from sklearn.metrics import accuracy_score

class AdaBoost:
    def __init__(self, X, y, Slabszy=DrzewoDecyzyjne):
        self.X=np.array(X)
        self.y=np.array(y).flatten(1) #Zwraca kopię tablicy zwiniętej do jednego wymiaru.
        self.Slabszy=Slabszy
        self.sums=np.zeros(self.y.shape)
        
        '''
        W jest wagą, początkowa sytuacja testowa jest równomiernie rozłożona, to znaczy wszystkie próbki mają wartość 1 / n
        '''
        self.W=np.ones((self.X.shape[1],1)).flatten(1)/self.X.shape[1]

        self.Q=0  #Rzeczywista liczba słabych klasyfikatorów
        
       # M to maksymalna liczba słabych klasyfikatorów, które można modyfikować w funkcji głównej
        
    def trenuj(self, M=8):
        self.G={}         # Słownik reprezentujący słaby klasyfikator
        self.alpha={}     # Parametry każdego słabego klasyfikatora
        for i in range(M):
            self.G.setdefault(i)
            self.alpha.setdefault(i)
        for i in range(M):   # self.G [i] jest i-tym słabym klasyfikatorem
            self.G[i]=self.Slabszy(self.X,self.y)
            e=self.G[i].trenuj(self.W) #Wytrenuj słaby klasyfikator zgodnie z aktualną wagą
            self.alpha[i]=1.0/2*np.log((1-e)/e) #Oblicz współczynniki tego klasyfikatora
            res=self.G[i].prognoza(self.X)  #res jest wyjściem z klasyfikatora

            #Oblicz bieżącą liczbę dokładności treningu
            print (" słaby klasyfkator dokładność", accuracy_score(self.y, res), "\n*******************************************************************")

            # Z jest współczynnikiem normalizacji
            Z=self.W*np.exp(-self.alpha[i]*self.y*res.transpose())
            self.W=(Z/Z.sum()).flatten(1) #Zaktualizuj wagę
            self.Q=i
            # errorcnt zwraca liczbę punktów błędu, 0 oznacza idealne
            if (self.blednePunkty(i)==0):
                print("%dSłabe klasyfikatory mogą obniżyć poziom błędu do 0"%(i+1))
                break

    def blednePunkty(self, t):   #Zwróć błędnie sklasyfikowane punkty
        self.sums= self.sums + self.G[t].prognoza(self.X).flatten(1) * self.alpha[t]
        
        prognoza_y=np.zeros(np.array(self.sums).shape)
        prognoza_y[self.sums >= 0]=1
        prognoza_y[self.sums < 0]=-1
        
        t=(prognoza_y != self.y).sum()
        return t
    
    def prognoza(self, test_X):  #Przetestuj ostateczny klasyfikator
        test_X=np.array(test_X)
        sumy=np.zeros(test_X.shape[1])
        for i in range(self.Q+1):
            sumy= sumy + self.G[i].prognoza(test_X).flatten(1) * self.alpha[i]
        prognoza_y=np.zeros(np.array(sumy).shape)
        prognoza_y[sumy >= 0]=1
        prognoza_y[sumy < 0]=-1
        return prognoza_y
