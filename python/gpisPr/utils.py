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
from scipy.spatial import cKDTree,KDTree
from sklearn.decomposition import PCA
import time 
import open3d as o3d
import transforms3d as t3d

def DEG2RAD(deg):
    return deg*np.pi/180


def RAD2DEG(rad):
    return rad*180/np.pi


def find_knn(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1,balanced_tree=False,compact_nodes=False)
    dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1) # find the nn_ids from the feat 1
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds
def find_knn_open3d(feat0, feat1,knn=1):
    feat1_fpfh=o3d.pipelines.registration.Feature()
    feat1_fpfh.data=feat0.T
    fpfh_tree_1=o3d.geometry.KDTreeFlann(feat1_fpfh)
    feat0_fpfh=o3d.pipelines.registration.Feature()
    feat0_fpfh.data=feat1.T
    _,indx,dis=fpfh_tree_1.search_knn_vector_xd(feat0_fpfh.data[:,0].reshape(33,1),knn=1)
    print(indx[0])

def find_correspondences(feats0, feats1, mutual_filter=True):
    # the feats1 to feats0
    #find_knn_open3d(feat0=feats0,feat1=feats1)
    start=time.time()
    nns0_to_1 = find_knn(feats0, feats1, knn=1, return_distance=False) # index of feats1 for each feats0
    corres01_idx0 = np.arange(len(nns0_to_1)) # The length of feats 0
    corres01_idx1 = nns0_to_1 # index of feats 1 for each feats0

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns1_to_0 = find_knn(feats1, feats0, knn=1, return_distance=False) # index of feats0 for each feats1
    corres10_idx1 = np.arange(len(nns1_to_0)) # the length of feats 1
    corres10_idx0 = nns1_to_0 # index of feats 0 for each feats 1

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]
    
    return corres_idx0, corres_idx1
def relative_position(x):
    rel_pose=[]
    print(x.shape[0])
    for i in range(x.shape[0]):
        rel_pose.append(x[i,:]-x[i+1:,:])
    return np.vstack(np.asarray(rel_pose,dtype="object"))

class Transformation:
    def __init__(self) -> None:
        self._T = np.identity(4)
    def setT(self,trans,rot_deg):
        self.rotation = t3d.euler.euler2mat(DEG2RAD(rot_deg[0]), DEG2RAD(rot_deg[1]), DEG2RAD(rot_deg[2]), 'sxyz')
        self.trans=trans
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
    def Transform(self, v):
        self._T = v



def getRightHandCoordinate(v1, v2, v3):
    if np.dot(np.cross(v1, v2), v3) > 0:
        return np.stack([v1, v2, v3], axis=1)
    else:
        return np.stack([v1, v2, -v3], axis=1)


def getAllRightHandCoordinate(R):
    R_list = []
    right_hand=getRightHandCoordinate(R[:,0], R[:,1], R[:,2])
    R_list.append(np.asarray([right_hand[:,0],right_hand[:,1],right_hand[:,2]]))
    R_list.append(np.asarray([-right_hand[:,0],-right_hand[:,1],right_hand[:,2]]))

    R_list.append(np.asarray([right_hand[:,0],-right_hand[:,1],-right_hand[:,2]]))
    R_list.append(np.asarray([-right_hand[:,0],right_hand[:,1],-right_hand[:,2]]))
    return R_list



def get_PCA_eigen_vector(pointXYZ):
    pca = PCA(n_components=3)
    pca.fit(pointXYZ)
    pca_eigen_vector = pca.components_.T
    #print("singular_values: ",pca.singular_values_)
    return pca_eigen_vector

def full_connected_graph(pointXYZ):
    pass

if __name__ == "__main__":
    x, y = np.mgrid[0:5, 2:8]
    tree = cKDTree(np.c_[x.ravel(), y.ravel()])
    dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
    print(dd)
    print(ii)
