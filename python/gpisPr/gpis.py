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

from tkinter.messagebox import NO
from pointCloud import PointCloud
import numpy as np
from skkernel import SKWilliamsMinusKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import pdist
from scipy.linalg.blas import sgemm
from point2SDF import Point2SDF
from liegroups import SE3
import transforms3d as t3d
from utils import *
from scipy.linalg import cho_factor, cho_solve
import copy
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class GPISData:
    def __init__(self,surface_points,num_out_lier=1000) -> None:
        self._surface_points=surface_points
        self._surface_points_down=surface_points
        self._surface_value=None
        self._R=1
        self.num_out_lier=num_out_lier
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
        query_points, sdf=point2sdf.sample_sdf_near_surface(number_of_points=self.num_out_lier)
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
        radius = pdist(self.X_source, metric="euclidean")
        self._R = np.max(radius)
        print("max_R: ",self._R)
class ConsitionPSDGPISModel:
    def __init__(self,kernel=None,random_state=0) -> None:
         self.Kernel =kernel
         self.alpha_=None
    def fit(self, X, y):
        self.X_source=X
        self.Y_source=y
        K=self.Kernel(X)
        self.Alpha=np.matmul(np.linalg.inv(K),y)
    def prediction(self, X):
        K_trans = self.Kernel(X, self.X_source)
        y_mean = K_trans@self.Alpha
        return y_mean
    @property
    def Alpha(self):
        return self.alpha_
    @Alpha.setter
    def Alpha(self,v):
        self.alpha_=v
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
    
class GPISModel(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=0.2,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GPISModel, self).__init__(kernel=kernel)
        self.Kernel =kernel
        self._X_source = None
        self._Y_source = None
        self._X_target = None

    def prediction(self, X):
        K_trans = self.kernel_(X, self.X_train_)
        print("K_trans: ",K_trans.shape)
        y_mean=K_trans@self.Alpha
        #y_mean = self._y_train_std * y_mean + self._y_train_mean
        return y_mean
    @property
    def Alpha(self):
        return self.alpha_
    @Alpha.setter
    def Alpha(self,v):
        self.alpha_=v
    @property
    def X_source(self):
        return self.X_train_

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

class GPISOpt:
    def __init__(self, voxel_size, gpisModel: GPISModel = None):
        self.voxel_size = voxel_size
        self.gpisModel = gpisModel
        self.sumofDetla = 0
        self.T_update = Transformation()
        self.obj_value=10000
        self.l=0.01

    def objective(self, target_points):
        return np.mean(self.gpisModel.predict(target_points) ** 2)

    def init(self, source: PointCloud = None, target: PointCloud = None):
        source_down, source_fpfh = source.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        target_down, target_fpfh = target.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        transform_es = self.execute_registration_fpfh_pca_init(source_down, source_fpfh, target_down, target_fpfh)
        # Registration.draw_registration_result(source.pcd,target.pcd,transform_es.Transform)
        return transform_es
    def execute(self):
        pass
    def step(self, target_points, T_last: Transformation = None):
        target_points_update = self.updateTarget_Point(target_points, T_last)
        se3_epsilon = self.calculateTransformationPerturbation(target_points=target_points_update,l=self.l)
        T_update = self.update_transformation(se3_epsilon, T_last)
        return target_points_update, T_update 
    
    def update_transformation(self, se3_epsilon, T_last: Transformation):
        return np.matmul(SE3.exp(se3_epsilon), T_last.Transform)

    def updateTarget_Point(self, target_points: np.ndarray = None, transform: Transformation = None):
        if target_points.shape[1] == 4:
            return np.matmul(transform.Transform, target_points.T)
        else:
            raise ValueError(
                "the points should be represented in the way of 4\times N")

    def updateGaussNewtonBasedPerturabation(self, targe_points,l=0):
        JTJ, JTr = self.calculateTransformationPerturbation(targe_points)
        print("JTJ: ",JTJ.shape)
        print("JTr: ",JTr.shape)
        JTJ_Hat = np.zeros_like(JTJ)
        print("JTJ_Hat: ",JTJ_Hat.shape)
        diagonalS = np.zeros_like(JTJ)
        print("diagonalS: ",diagonalS.shape)
        np.fill_diagonal(diagonalS,JTJ.diagonal())
        JTJ_Hat = JTJ + l * diagonalS
        L_= cho_factor(JTJ_Hat, lower=True)
        se3_epsilon = cho_solve(L_, JTr)
        return se3_epsilon


    def calculateTransformationPerturbation(self,target_points):
        N,_=target_points.shape
        BetaM = self.getBetaM(target_points)
        DeltaM = self.getDeltaM(target_points).reshape(N,-1,6)
        Alpha=self.gpisModel.Alpha.reshape(-1,1)
        betaalpha = np.sum(Alpha*BetaM,axis=0).reshape(-1,1)
        Alpha2=np.repeat(np.expand_dims(Alpha,axis=0),N,axis=0)
        DeltaMAlpha = np.sum(DeltaM*Alpha2,axis=1).reshape(N,6,1)
        JTJ = np.sum(np.matmul(DeltaMAlpha, DeltaMAlpha.reshape(N,1,6)),axis=0) # 6\times 6
        JTr = np.sum(np.matmul(DeltaMAlpha, np.expand_dims(betaalpha,axis=1)),axis=0) # 6 \times 1
        return JTJ, JTr

    def getBetaM(self,target_points):
        return self.gpisModel.kernel_(self.gpisModel.X_source, target_points)

    def getDeltaM(self,target_points):
        N,_=target_points.shape
        N_source,_=self.gpisModel.X_source.shape
        Ty_odot = SE3.odot(PointCloud.PointXYZ2homogeneous(target_points))  # R^(N \times 4 \times 6)
        Ty_odot=np.repeat(Ty_odot,N_source,axis=0)
        dk_dy = self.gpisModel.Kernel.gradient(self.gpisModel.X_source, target_points).T.reshape(-1,1)#*(target_points-self.gpisGR.X_source)
        target_points_copy=np.repeat(copy.deepcopy(target_points),N_source,axis=0)
        source_points_copy=np.repeat(copy.deepcopy(self.gpisModel.X_source),N,axis=0)
        target_source_diff=(target_points_copy-source_points_copy).reshape(N*N_source,3)
        dk=PointCloud.PointXYZ2homogeneous(dk_dy*target_source_diff).reshape(N*N_source,1,4)
        deltaM=np.matmul(dk,Ty_odot)
        return deltaM

    def execute_registration_fpfh_pca_init(self,source_down,source_fpfh,target_down, target_fpfh):
        corres_idx0, corres_idx1 = find_correspondences(source_fpfh, target_fpfh)
        source_down_fpfh = source_down[corres_idx0, :]
        target_down_fpfh = target_down[corres_idx1, :]

        source_pca_vectors=get_PCA_eigen_vector(source_down_fpfh)
        R_source=getRightHandCoordinate(source_pca_vectors[:,0],source_pca_vectors[:,1],source_pca_vectors[:,2])

        target_pca_vectors=get_PCA_eigen_vector(target_down_fpfh)
        R_target=getRightHandCoordinate(target_pca_vectors[:,0],target_pca_vectors[:,1],target_pca_vectors[:,2])

        R_es=np.matmul(R_target,R_source.T)

        source_down_fpfh_center=PointCloud.PointCenter(source_down_fpfh)
        target_down_fpfh_center=PointCloud.PointCenter(target_down_fpfh)
        trans=target_down_fpfh_center-np.matmul(R_es,source_down_fpfh_center)

        transform_es=Transformation()
        transform_es.rotation=R_es
        transform_es.trans=trans
        return transform_es


if __name__ == '__main__':
    X = np.random.rand(10, 3)
    y = np.random.rand(10, 1)
    kernel = SKWilliamsMinusKernel(3)
    gpis = GPISModel(kernel=SKWilliamsMinusKernel(3), random_state=0)
    gpis.fit(X, y)
