from abc import ABC, abstractmethod
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

class WilliamsPlusKernel:
    def __init__(self, R):
        self.name = 'WilliamsPlusKernel'
        self.R = R

    def get_kernel(self, p1, p2):
        r = np.linalg.norm(p1-p2)
        return 2*np.power(r, 3)+3*self.R * np.power(r, 2)+np.power(self.R, 3)

    def dK(self, p1, p2):
        r = np.linalg.norm(p1-p2)
        return 6*np.power(r, 2)+6*self.R*r


class WilliamsMinusKernel:
    def __init__(self, R):
        self.name = 'WilliamsMinusKernel'
        self.R = R

    def get_kernel(self, p1, p2):
        r = np.linalg.norm(p1-p2)
        return 2*np.power(r, 3)-3*self.R * np.power(r, 2)+np.power(self.R, 3)

    def dK(self, p1, p2):
        r = np.linalg.norm(p1-p2)
        return 6*np.power(r, 2)-6*self.R*r


class RadialBasisKernel:
    def __init__(self, R=None):
        self.name = "RadialBasisKernel"
        self.R = R

    def get_kernel(self, p1, p2):
        r = np.linalg.norm(p1-p2)
        return np.exp(-np.power(r, 2))

    def dK(self, p1, p2):
        r = np.linalg.norm(p1-p2)
        t1 = np.power(r, 2)/(2*np.power(1/self.R, 2))
        t2 = -2*r*np.exp(-t1)/(2*np.power(1/self.R, 2))
        return t2
