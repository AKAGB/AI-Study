import numpy as np
import pandas as pd

from math import e
from math import pi
from math import log
from math import inf
from math import isinf

class GMM:
    def __init__(self, Data, K, limit=10000, ep=1e-10):
        self.name = Data.iloc[:, 0]
        self.DataSet = np.mat(Data.iloc[:,1:])
        self.K = K
        self.N, self.D = np.shape(self.DataSet)
        self.gammas = np.zeros([self.N, self.K])
        self.mius = np.zeros([self.K, self.D])
        self.pi_ = np.zeros(self.K)
        self.sigmas = np.zeros([self.K, self.D, self.D])
        self.logs = np.zeros(self.N)
        self.iterLimit = limit
        self.ep = ep

    def Gaussian(self, miu, sigma):
        """ return Gaussian Distribution """
        def distribution(x):
            weight = 1 / ((2 * pi)**(self.D / 2))
            weight /= (np.linalg.det(sigma))**0.5
            x_ = np.transpose(x - miu)
            s_ = np.linalg.inv(sigma)
            ex = float(np.dot(np.transpose(x_) , np.dot(s_, x_)))
            ex *= -0.5
            if ex < -700:
                return e**-700
            if ex > 700:
                return e**700
            return weight * e**ex
        return distribution
    
    def initialize(self, selectData):
        for i in range(self.K):
            self.mius[i] = self.DataSet[selectData[i]]
            self.sigmas[i] = np.eye(self.D) * 20
            self.pi_[i] = 1 / self.K

    def Estep(self):
        # Get Gaussian distributions
        Ns = [self.Gaussian(self.mius[i], self.sigmas[i]) for i in range(self.K)]

        for i in range(self.N):
            tmp = [self.pi_[k] * Ns[k](self.DataSet[i]) for k in range(self.K)]
            s = sum(tmp)
            tmp = [x / s for x in tmp]
            self.gammas[i] = np.mat(tmp)
            self.logs[i] = log(s)

    def Mstep(self):
        Nk = np.sum(self.gammas, axis=0)
        self.pi_ = np.array([Nk[k] / self.N for k in range(self.K)])
        # Update miu
        # self.mius = np.transpose(np.dot(self.DataSet.T, self.gammas) / Nk)
        # Update sigams
        for i in range(self.K):
            tmp = self.DataSet - self.mius[i]
            s = np.zeros([self.D, self.D])
            m = np.zeros(self.D)
            for j in range(self.N):
                m = m + self.gammas[j, i] * self.DataSet[j]
                s = s + self.gammas[j, i] * np.dot(np.transpose(tmp[j]), tmp[j])
            if Nk[i] == 0:
                Nk[i] = 1e-50
            self.mius[i] = np.mat(m / Nk[i])
            self.sigmas[i] = s / Nk[i]
            while np.linalg.det(self.sigmas[i]) <= 0:
                self.sigmas[i] = self.sigmas[i] + np.eye(self.D) * 0.1

    def likelihood(self):
        return np.sum(self.logs)

    def run(self, selectData):
        """Run EM Algorithm"""
        self.initialize(selectData)
        new_lh = inf
        old_lh = inf
        for i in range(self.iterLimit):
            if abs(new_lh - old_lh) < self.ep:
                break
            old_lh = new_lh
            self.Estep()
            self.Mstep()
            new_lh = self.likelihood()
        else:
            print('Iteration reaches the upper limit!')
    
    def cluster(self):
        result = [[] for i in range(self.K)]
        for i in range(self.N):
            result[np.argmax(self.gammas[i])].append(self.name[i])
        level = np.sum(self.mius, axis=1)
        level = sorted(enumerate(level), key=lambda x : x[1])
        level = [x[0] for x in level]
        labels = []
        for i in range(self.K):
            labels.append(result[level[i]])
        
        return labels

if __name__ == "__main__":
    df = pd.read_csv('football.csv')
    model = GMM(df, 3, 1000, 1e-10)
    model.run([0, 5, 8])
    res = model.cluster()
    labels = ['First-class', 'Second-class', 'Third-class']
    for i in range(3):
        print(labels[i], ':\t', sorted(res[i]), sep='')