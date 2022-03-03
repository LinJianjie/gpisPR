# Copyright (c) 2022 Jianjie Lin
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from tkinter.messagebox import NO
from pointCloud import PointCloud
import numpy as np
from skkernel import SKWilliamsMinusKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class GPIS(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GPIS, self).__init__()
        self._X = None
        self._Y = None
        self._X_target=None

    @property
    def X_source(self):
        return self._X

    @X_source.setter
    def X_source(self, v):
        self._X = v

    @property
    def Y_source_value(self):
        return self._Y

    @Y_source_value.setter
    def Y_source_value(self, v):
        self._Y = v

    @property
    def X_target(self):
        return self._X_target

    @X_target.setter
    def X_target(self, v):
        self._X_target = v


if __name__ == '__main__':
    X = np.random.rand(10, 3)
    y = np.random.rand(10, 1)
    kernel = SKWilliamsMinusKernel(3)
    gpis = GPIS(kernel=SKWilliamsMinusKernel(3), random_state=0)
    gpis.fit(X, y)
