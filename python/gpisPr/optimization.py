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


import copy

from matplotlib.pyplot import axis 
from liegroups import SE3
from scipy.linalg import cho_factor, cho_solve
from gpis import GPIS
from utils import *
import numpy as np
from registration import Registration
from pointCloud import PointCloud
import open3d as o3d
from sklearn.decomposition import PCA
import transforms3d as t3d
import os
import sys
from telnetlib import SE
from tkinter.messagebox import NO

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class Optimization:
    def __init__(self, voxel_size, gpis: GPIS = None):
        self.voxel_size = voxel_size
        self.gpisGR = gpis
        self.sumofDetla = 0
        self.T_update = Transformation()
        self.obj_value=10000

    def objective(self, target_points):
        return np.mean(self.gpisGR.predict(target_points) ** 2)

    def init(self, source: PointCloud = None, target: PointCloud = None):
        source_down, source_fpfh = source.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        target_down, target_fpfh = target.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        transform_es = Registration.execute_registration_fpfh_pca_init(source_down, source_fpfh, target_down, target_fpfh)
        # Registration.draw_registration_result(source.pcd,target.pcd,transform_es.Transform)
        return transform_es

    def step(self, target_points, T_last: Transformation = None):
        target_points_update = self.updateTarget_Point(target_points, T_last)
        se3_epsilon = self.calculateTransformationPerturbation()
        T_update = self.update_transformation(se3_epsilon, T_last)
        return target_points_update, T_update 

    def updateGaussNewtonBasedPerturabation(self, l):
        JTJ, JTr = self.calculateTransformationPerturbation()
        JTJ_Hat = np.zeros(JTJ.shape[0], JTr.shape[1])
        diagonalS = np.zeros(JTJ.shape[0], JTr.shape[1])
        diagonalS.diagonal = JTJ.diagonal
        JTJ_Hat = JTJ + l * diagonalS
        L_, low = cho_factor(JTJ_Hat, lower=True)
        se3_epsilon = cho_solve(L_, JTr)
        return se3_epsilon

    def update_transformation(self, se3_epsilon, T_last: Transformation):
        return np.matmul(SE3.exp(se3_epsilon), T_last.Transform)

    def updateTarget_Point(self, target_points: np.ndarray = None, transform: Transformation = None):
        if target_points.shape[0] == 4:
            return np.matmul(transform.Transform, target_points)
        else:
            raise ValueError(
                "the points should be represented in the way of 4\times N")

    def calculateTransformationPerturbation(self,target_points):
        N,_=target_points.shape
        BetaM = self.getBetaM(target_points)
        DeltaM = self.getDeltaM(target_points).reshape(N,-1,6)
        Alpha=self.gpisGR.Alpha.reshape(-1,1)
        betaalpha = np.sum(Alpha*BetaM,axis=0).reshape(-1,1)
        Alpha2=np.repeat(np.expand_dims(Alpha,axis=0),N,axis=0)
        DeltaMAlpha = np.sum(DeltaM*Alpha2,axis=1).reshape(N,6,1)
        JTJ = np.sum(np.matmul(DeltaMAlpha, DeltaMAlpha.reshape(N,1,6)),axis=0) # 6\times 6
        JTr = np.sum(np.matmul(DeltaMAlpha, np.expand_dims(betaalpha,axis=1)),axis=0) # 6 \times 1
        return JTJ, JTr

    def getBetaM(self,target_points):
        return self.gpisGR.Kernel(self.gpisGR.X_source, target_points)

    def getDeltaM(self,target_points):
        N,_=target_points.shape
        N_source,_=self.gpisGR.X_source.shape
        Ty_odot = SE3.odot(PointCloud.PointXYZ2homogeneous(target_points))  # R^(N \times 4 \times 6)
        Ty_odot=np.repeat(Ty_odot,N_source,axis=0)
        dk_dy = self.gpisGR.Kernel.gradient(self.gpisGR.X_source, target_points).T.reshape(-1,1)#*(target_points-self.gpisGR.X_source)
        target_points_copy=np.repeat(copy.deepcopy(target_points),N_source,axis=0)
        source_points_copy=np.repeat(copy.deepcopy(self.gpisGR.X_source),N_source,axis=0)
        target_source_diff=(target_points_copy-source_points_copy).reshape(N*N_source,3)
        dk=PointCloud.PointXYZ2homogeneous(dk_dy*target_source_diff).reshape(N*N_source,1,4)
        deltaM=np.matmul(dk,Ty_odot)
        return deltaM

    def execute(self):
        pass

if __name__ == '__main__':
    source = PointCloud(filename="../data/happy.pcd")
    source()
    target = PointCloud(filename="../data/happy.pcd")
    target()
    transinit = Transformation()
    transinit.trans = np.asarray([0.1, 0.2, 0.2])
    rot = t3d.euler.euler2mat(
        DEG2RAD(20.0), DEG2RAD(40.0), DEG2RAD(30.0), 'sxyz')
    transinit.rotation = rot
    target.transform(transinit=transinit.Transform)
    opt = Optimization(voxel_size=0.01)
    opt.init(source=source, target=target)

    # p1=np.matmul(transinit.rotation,source.point.T)
    # p2=target.point.T
    # print(p2-p1)
