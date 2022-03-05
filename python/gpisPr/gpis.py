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
from point2SDF import Point2SDF
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class GPISData:
    def __init__(self,surface_points) -> None:
        self._surface_points=surface_points
        self._surface_points_down=PointCloud()
        self._surface_value=None
        self._R=1
        self.voxel_size=0.0001
        self.out_lier=None
        self.out_lier_value=None
        self._X_source=None
        self._Y_source=None
        
    @property
    def X_source(self):
        return self._X_source
    @property
    def Y_source(self):
        return self._Y_source

    def __call__(self):
        self.create_outlier()
        self._X_source=np.vstack([self._surface_points_down.point,self.out_lier])
        surface_value=np.zeros(self._surface_points_down.size)
        self._Y_source=np.concatenate([surface_value,self.out_lier_value])
        self.compute_max_radius()
    
    def create_outlier(self):
        self._surface_points_down.estimate_normal(self.voxel_size, 30)
        point2sdf = Point2SDF(self._surface_points_down)
        query_points, sdf=point2sdf.sample_sdf_near_surface(number_of_points=1000)
        self.out_lier=query_points[sdf>0,:]
        self.out_lier_value=sdf[sdf>0]
    @property
    def surface_points_down(self):
        return self._surface_points_down
    
    def voxel_points(self,voxel_size):
        self.voxel_size=voxel_size
        self._surface_points_down.pcd=self.surface_points.voxel_down_sample(self.voxel_size)
    @property
    def surface_points(self):
        return self._surface_points
    @surface_points.setter
    def surface_points(self,v:PointCloud=None):
        self._surface_points=v
    @property
    def surface_value(self):
        return self._surface_value
    @surface_value.setter
    def surface_value(self,v):
        self._surface_value=v    
    @property
    def maxR(self):
        return self._R
    def compute_max_radius(self):
        radius = pdist(self._surface_points.point, metric="euclidean")
        self._R = np.max(radius)

class GPIS(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GPIS, self).__init__()
        self.custom_kernel = kernel
        self._X_source = None
        self._Y_source = None
        self._X_target = None
        
    def fit(self, X, y):
        self.X_source=X
        self.Y_source=y
        return super().fit(X, y)
    @property
    def Alpha(self):
        return self.alpha_

    @property
    def Kernel(self):
        return self.custom_kernel

    @property
    def X_source(self):
        return self._X_source

    @X_source.setter
    def X_source(self, v):
        self._X_source = v

    @property
    def Y_source(self):
        return self._Y_source

    @Y_source.setter
    def Y_source(self, v):
        self._Y_source = v

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
