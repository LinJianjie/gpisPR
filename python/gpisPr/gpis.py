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

from pointCloud import PointCloud
import numpy as np
from skkernel import SKWilliamsMinusKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import pdist
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class GPISData:
    def __init__(self) -> None:
        pass

    def compute_max_radius(self):
        radius = pdist(self.surface_points, metric="euclidean")
        self.__R = np.max(radius)


class GPIS(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GPIS, self).__init__()
        self.custom_kernel = kernel
        self._X = None
        self._Y = None
        self._X_target = None

    @property
    def Alpha(self):
        return self.alpha_

    @property
    def Kernel(self):
        return self.custom_kernel

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
