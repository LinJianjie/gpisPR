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

from dis import dis
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from sklearn.gaussian_process.kernels import Kernel,RBF,Matern
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import kv, gamma

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
    def __init__(self, R, alpha=0.1):
        self.name = 'SKWilliamsMinusKernel'
        self.R = R
        self.has_kxx=False
        self.KXX=None
        self.alpha=alpha
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        
        if Y is None:
            if not self.has_kxx:
                d1 = pdist(X, metric='euclidean')
                self.has_kxx=True
                tmp1=np.power(d1, 2)
                tmp2=np.power(self.R, 3)
                self.kxx_dist = 2*tmp1*d1-3*self.R * tmp1+tmp2
                # convert from upper-triangular matrix to square matrix
                K1 = squareform(self.kxx_dist)
                np.fill_diagonal(K1, tmp2+self.alpha)
                self.KXX=K1
            else:
                K1 = self.KXX
            K=K1
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
        K_gradient=6*(d1-self.R)*d1
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
        self.has_kxx=False
        self.KXX=None
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            if not self.has_kxx:
                dists = pdist(X / length_scale, metric="sqeuclidean")
                K1 = np.exp(-0.5 * dists)
                
                # convert from upper-triangular matrix to square matrix
                K1 = squareform(K1)
                np.fill_diagonal(K1, 1)
                self.KXX=K1
                self.has_kxx=True
            else:
                K1 = self.KXX
            K=K1
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

def _approx_fprime(xk, f, epsilon, args=()):
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad

class SKMatern(Matern):
    def __init__(self, length_scale=1, length_scale_bounds=..., nu=1.5):
        super().__init__(length_scale, length_scale_bounds, nu)
        self.has_kxx=False
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="euclidean")
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * np.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * np.sqrt(5)
            K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists ** 2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = np.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale ** 2
                )
            else:
                D = squareform(dists ** 2)[:, :, np.newaxis]

            if self.nu == 0.5:
                denominator = np.sqrt(D.sum(axis=2))[:, :, np.newaxis]
                K_gradient = K[..., np.newaxis] * np.divide(
                    D, denominator, where=denominator != 0
                )
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            elif self.nu == np.inf:
                K_gradient = D * K[..., np.newaxis]
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K
    def gradient(self,X,Y):
        dists = cdist(X / self.length_scale, Y / self.length_scale, metric="euclidean")
        if self.nu == 0.5:
            K_gradient=np.exp(-dists)*-1
            return K_gradient
        if self.nu==1.5:
            tmp1=np.sqrt(3)
            tmp2 = dists * tmp1
            K_gradient=np.exp(-tmp2)*tmp1+(1.0 + tmp2) * np.exp(-tmp2)*(-1)*tmp1
            return K_gradient
        if self.nu==2.5:
            tmp1=np.sqrt(5)
            K = dists * np.sqrt(5)
            K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
            K_gradient=(tmp1+2/3.0*tmp1*K)*np.exp(-K)+(1.0 + K + K ** 2 / 3.0)*np.exp(-K)*-1*tmp1
            return K_gradient
        else:
            raise ValueError("not defined nu")
if __name__ == '__main__':
    x=np.asarray([[1,2,3],[2,3,4],[4,5,6],[7,6,8]])
    y=np.random.random([2,3])
    skw=SKWilliamsMinusKernel(3)
