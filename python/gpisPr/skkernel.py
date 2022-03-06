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
from sklearn.gaussian_process.kernels import Kernel,RBF
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
            tmp1=np.power(d1,2)
            tmp2=np.power(self.R, 3)
            dist1 = 2*tmp1*d1+3*self.R * tmp1+tmp2
            # convert from upper-triangular matrix to square matrix
            K = squareform(dist1)
            Y=np.sum(np.abs(K),axis=1)/X.shape[0]
            print(Y)
            np.fill_diagonal(K, tmp2+Y)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            d1 = cdist(X, Y,metric='euclidean')
            tmp1=np.power(d1,2)
            dist1 = 2*tmp1*d1+3*self.R * tmp1+np.power(self.R, 3)
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
            tmp1=np.power(d1, 2)
            tmp2=np.power(self.R, 3)
            dist1 = 2*tmp1*d1-3*self.R * tmp1+tmp2
            # convert from upper-triangular matrix to square matrix
            K = squareform(dist1)
            Y=np.sum(np.abs(K),axis=1)/X.shape[0]
            np.fill_diagonal(K, tmp2+Y)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            d1 = cdist(X, Y,metric='euclidean')
            tmp1=np.power(d1, 2)
            dist1 = 2*tmp1*d1-3*self.R * tmp1+np.power(self.R, 3)
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

def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale
class SKRBF(RBF):
    def __init__(self, length_scale=1, length_scale_bounds=(1e-5, 1e5)):
        super().__init__(length_scale, length_scale_bounds)
        self.length_scale=length_scale
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale ** 2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K
    def gradient(self,X,Y):
        dists = cdist(X / self.length_scale, Y / self.length_scale,metric='euclidean')
        K_gradient=np.exp(-0.5 * np.power(dists,2))*-1*dists
        return K_gradient

if __name__ == '__main__':
    x=np.asarray([[1,2,3],[2,3,4],[4,5,6],[7,6,8]])
    y=np.random.random([2,3])
    skw=SKWilliamsMinusKernel(3)
