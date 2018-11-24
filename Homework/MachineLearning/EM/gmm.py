import numpy as np
import pandas as pd

from math import e
from math import pi

class GMM:
    def __init__(self, DataSet, K):
        self.DataSet = np.array(DataSet)
        self.K = K
        self.N, self.D = np.shape(self.DataSet)
        self.gammas = np.zeros(self.N, self.D)
        self.mius = np.zeros([self.K, self.D])
        self.pi_ = np.zeros(self.K)
        self.sigmas = np.zeros([self.K, self.D, self.D])
        self.Nk = np.zeros(self.K)
        self.iterLimit = 10000

    def Gaussian(self, miu, sigma):
        """ return Gaussian Distribution """
        def distribution(x):
            weight = 1 / ((2 * pi)**(self.D / 2))
            weight /= (np.linalg.det(sigma))**0.5
            x_ = x - miu
            s_ = np.linalg.inv(sigma)
            ex = np.dot(x_.T , np.dot(s_, x_))
            ex *= -0.5
            return weight * e**ex
        return distribution
    
    def initialize(self, selectData):
        for i in range(self.K):
            self.mius[i] = self.DataSet[selectData[i]]
            self.sigmas[i] = np.eye(self.D) * 0.1
            self.pi_[i] = 1 / self.K
