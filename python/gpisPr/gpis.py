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

from pydoc import doc
from tkinter.messagebox import NO
from pointCloud import PointCloud
import numpy as np
from skkernel import SKWilliamsMinusKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize
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
        self._surface_points=copy.deepcopy(surface_points)
        self._surface_points_down=copy.deepcopy(surface_points)
        self._surface_value=None
        self._R=1
        self.num_out_lier=num_out_lier
        self.voxel_size=0.0001
        self.out_lier=None
        self.out_lier_value=None
        self._X_source=None
        self._Y_source=None
    
    def __call__(self):
        self.create_outlier()
        self._X_source=np.vstack([self._surface_points_down.point,self.out_lier])
        surface_value=np.zeros(self._surface_points_down.size)
        self._Y_source=np.concatenate([surface_value,self.out_lier_value])
        self.compute_max_radius()

    @property
    def X_source(self):
        return self._X_source
    @property
    def Y_source(self):
        return self._Y_source

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
        y_mean=K_trans@self.Alpha
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
        return self.y_train_

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
    def __init__(self, voxel_size,gpisModel: GPISModel = None):
        self.gpisModel = gpisModel
        self.voxel_size = voxel_size
        self.sumofDetla = 0
        self.T_last = Transformation()
        self.obj_value=10000
        self.obj_opt_min_=0
        self.obj_last=1000000
        self.l=0.01
        self.stop_condition=0.0001
        self.max_iteration=100
    @property
    def obj_opt_min(self):
        return self.obj_opt_min_
    @obj_opt_min.setter
    def obj_opt_min(self,v):
        self.obj_opt_min_=v
    def objective(self, target_points):
        return np.mean(np.abs(self.gpisModel.prediction(target_points)))
    def check_objective_improvement(self, target_points_updated):
        new_objective=self.objective(target_points=target_points_updated)
        return new_objective<self.obj_last
        
    def init(self, source: PointCloud = None, target: PointCloud = None):
        source_down, source_fpfh = source.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        target_down, target_fpfh = target.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        transform_es = self.execute_registration_fpfh_pca_init(source_down, source_fpfh, target_down, target_fpfh)
        return transform_es
        
    def execute(self,source: PointCloud = None, target: PointCloud = None):
        iteration_step=0
        transinit=self.init(source=source,target=target)
        self.T_last=transinit
        while np.abs(self.obj_last-self.obj_opt_min_)>self.stop_condition:
            target_points_update, self.T_last=self.step(target_points=target,T_last=self.T_last)
            obj_update_value=self.objective(target_points=target_points_update)
            self.obj_last=obj_update_value
            iteration_step=iteration_step+1
            if iteration_step>self.max_iteration:
                break
        
    def step(self, target_points, T_last: Transformation = None):
        target_points_update = self.updateTarget_Point(target_points, T_last)
        se3_epsilon = self.updateGaussNewtonBasedPerturabation(target_points=target_points_update,l=self.l)
        T_update = self.update_transformation(se3_epsilon, T_last)
        return target_points_update, T_update 
    def updateGaussNewtonBasedPerturabation(self, target_points,l=0):
        JTJ, JTr = self.calculateTransformationPerturbation(target_points)
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

    def update_transformation(self, se3_epsilon, T_last: Transformation):
        se3_epsilon=se3_epsilon.reshape(-1)
        return np.matmul(SE3.exp(se3_epsilon).as_matrix(), T_last.Transform)

    def updateTarget_Point(self, target_points: np.ndarray = None, transform: Transformation = None):
        if target_points.shape[1] == 4:
            return PointCloud.Homogeneous2PointXYZ(np.matmul(transform.Transform, target_points.T))
        if target_points.shape[1] == 3:
            return PointCloud.Homogeneous2PointXYZ(np.matmul(transform.Transform, PointCloud.PointXYZ2homogeneous(target_points).T))


    def getBetaM(self,target_points):
        return self.gpisModel.kernel_(self.gpisModel.X_source, target_points)
    def getNormDerivative(self, point_source, point_target):
        N,_=point_target.shape
        N_source,_=point_source.shape
        target_points_copy=np.repeat(copy.deepcopy(point_target),N_source,axis=0)
        source_points_copy=np.repeat(copy.deepcopy(point_source),N,axis=0)
        target_source_diff=(target_points_copy-source_points_copy).reshape(N*N_source,3)
        target_source_diff_abs=np.abs(target_source_diff)
        target_source_diff_norm=normalize(target_source_diff_abs)
        norm_derivative=np.sign(target_source_diff)*target_source_diff_norm
        return norm_derivative
    def getDeltaM(self,target_points):
        N,_=target_points.shape
        N_source,_=self.gpisModel.X_source.shape
        Ty_odot = SE3.odot(PointCloud.PointXYZ2homogeneous(target_points))  # R^(N \times 4 \times 6)
        Ty_odot=np.repeat(Ty_odot,N_source,axis=0)
        dk_dr = self.gpisModel.Kernel.gradient(self.gpisModel.X_source, target_points).T.reshape(-1,1)
        dr_dy=self.getNormDerivative(self.gpisModel.X_source,target_points)
        """target_points_copy=np.repeat(copy.deepcopy(target_points),N_source,axis=0)
        #source_points_copy=np.repeat(copy.deepcopy(self.gpisModel.X_source),N,axis=0)
        #target_source_diff=(target_points_copy-source_points_copy).reshape(N*N_source,3)
        #dk=PointCloud.PointXYZ2homogeneous(dk_dy*target_source_diff).reshape(N*N_source,1,4)"""
        dk_dy=PointCloud.PointXYZ2homogeneous(dk_dr*dr_dy).reshape(N*N_source,1,4)
        deltaM=np.matmul(dk_dy,Ty_odot)
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
