# Copyright (C) 2022 Jianjie Lin
# 
# This file is part of python.
# 
# python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with python.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from scipy.spatial.distance import pdist, cdist, squareform


class SKWilliamsPlusKernel(Kernel):
    def __init__(self, R):
        self.name = 'SKWilliamsPlusKernel'
        self.R = R

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        print(X.shape)
        if Y is None:
            d1 = pdist(X, metric='euclidean')
            dist1 = 2*np.power(d1, 3)+3*self.R * np.power(d1, 2)+np.power(self.R, 3)
            # convert from upper-triangular matrix to square matrix
            K = squareform(dist1)
            np.fill_diagonal(K, np.power(self.R, 3))
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            d1 = cdist(X, Y,metric='euclidean')
            dist1 = 2*np.power(d1, 3)+3*self.R * np.power(d1, 2)+np.power(self.R, 3)
            K = dist1
        if eval_gradient:
            d1 = cdist(X, Y,metric='euclidean')
            K_gradient=6*(d1+self.R)
            return K, K_gradient
        else:
            return K
    def gradient(self, X, Y):
        d1 = cdist(X, Y,metric='euclidean')
        K_gradient=6*(d1-self.R)
        return K_gradient
    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False
    def diag(self, X):
        return np.ones(X.shape[0])*np.power(self.R, 3)


class SKWilliamsMinusKernel(Kernel):
    def __init__(self, R):
        self.name = 'SKWilliamsMinusKernel'
        self.R = R

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            d1 = pdist(X, metric='euclidean')
            dist1 = 2*np.power(d1, 3)+3*self.R * np.power(d1, 2)+np.power(self.R, 3)
            # convert from upper-triangular matrix to square matrix
            K = squareform(dist1)
            np.fill_diagonal(K, np.power(self.R, 3))
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            d1 = cdist(X, Y,metric='euclidean')
            dist1 = 2*np.power(d1, 3)-3*self.R * np.power(d1, 2)+np.power(self.R, 3)
            K = dist1
        if eval_gradient:
            d1 = cdist(X, Y,metric='euclidean')
            K_gradient=6*(d1-self.R)
            return K, K_gradient
        else:
            return K
    def gradient(self,X,Y):
        d1 = cdist(X, Y,metric='euclidean')
        K_gradient=6*(d1-self.R)
        return K_gradient
    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False
    def diag(self, X):
        return np.ones(X.shape[0])*np.power(self.R, 3)
if __name__ == '__main__':
    x=np.asarray([[1,2,3],[2,3,4],[4,5,6],[7,6,8]])
    y=np.random.random([2,3])
    skw=SKWilliamsMinusKernel(3)
