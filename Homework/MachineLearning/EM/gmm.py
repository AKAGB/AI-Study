import numpy as np
import pandas as pd

from math import e
from math import pi
from math import log
from math import inf

class GMM:
    def __init__(self, DataSet, K, limit=10000, ep=1e-10):
        self.DataSet = np.mat(DataSet)
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
            ex = np.sum(np.dot(np.transpose(x_) , np.dot(s_, x_)))
            ex *= -0.5
            return weight * e**ex
        return distribution
    
    def initialize(self, selectData):
        for i in range(self.K):
            self.mius[i] = self.DataSet[selectData[i]]
            self.sigmas[i] = np.eye(self.D) * 50
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
        self.mius = np.transpose(np.dot(self.DataSet.T, self.gammas) / Nk)
        # Update sigams
        for i in range(self.K):
            tmp = self.DataSet - self.mius[i]
            s = np.zeros([self.D, self.D])
            for j in range(self.N):
                s = s + self.gammas[j, i] * np.dot(np.transpose(tmp[j]), tmp[j])
            self.sigmas[i] = s / Nk[i]

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
            print('迭代达到上限！')

if __name__ == "__main__":
    df = pd.read_csv('football.csv')
    data = np.mat(df.iloc[:,1:])
    model = GMM(data, 3, 1000, 1e-6)
    model.run([0, 2, 8])
