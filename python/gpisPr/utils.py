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
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


def DEG2RAD(deg):
    return deg*np.pi/180


def RAD2DEG(rad):
    return rad*180/np.pi


def find_knn(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_correspondences(feats0, feats1, mutual_filter=True):
    # the feats1 to feats0
    nns01 = find_knn(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


class Transformation:
    def __init__(self) -> None:
        self._T = np.identity(4)

    @property
    def trans(self):
        return self._T[:3, 3]

    @trans.setter
    def trans(self, v):
        self._T[:3, 3] = v

    @property
    def rotation(self):
        return self._T[:3, :3]

    @rotation.setter
    def rotation(self, v):
        self._T[:3, :3] = v

    @property
    def Transform(self):
        return self._T

    @Transform.setter
    def T(self, v):
        self._T = v


def getRightHandCoordinate(v1, v2, v3):
    if np.dot(np.cross(v1, v2), v3) > 0:
        return np.stack([v1, v2, v3], axis=1)
    else:
        return np.stack([v1, v2, -v3], axis=1)


def getAllRightHandCoordinate(R):
    R_list = []
    R_list.append(getRightHandCoordinate(R[:,0], R[:,1], R[:,2]))
    R_list.append(getRightHandCoordinate(R[:,0], R[:,2], R[:,1]))

    R_list.append(getRightHandCoordinate(R[:,1], R[:,0], R[:,2]))
    R_list.append(getRightHandCoordinate(R[:,1], R[:,2], R[:,0]))

    R_list.append(getRightHandCoordinate(R[:,2], R[:,1], R[:,0]))
    R_list.append(getRightHandCoordinate(R[:,2], R[:,0], R[:,1]))
    return R_list



def get_PCA_eigen_vector(pointXYZ):
    pca = PCA(n_components=3)
    pca.fit(pointXYZ)
    pca_eigen_vector = pca.components_.T
    return pca_eigen_vector


if __name__ == "__main__":
    x, y = np.mgrid[0:5, 2:8]
    tree = cKDTree(np.c_[x.ravel(), y.ravel()])
    dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
    print(dd)
    print(ii)
