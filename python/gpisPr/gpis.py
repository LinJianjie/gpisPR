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
from sklearn.preprocessing import normalize
from point2SDF import Point2SDF
from liegroups import SE3
import transforms3d as t3d
import tqdm
from utils import *
from scipy.linalg import cho_factor, cho_solve
import copy
import os
import sys
import time 
from registration import Registration
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class GPISData:
    def __init__(self,surface_points,num_in_out_lier=1000,has_in_lier=False) -> None:
        self._surface_points=copy.deepcopy(surface_points)
        self._surface_points_down=copy.deepcopy(surface_points)
        self._surface_value=None
        self._R=1
        self.num_out_lier=num_in_out_lier
        self.voxel_size=0.0001
        self.out_lier=None
        self.out_lier_value=None
        self.in_lier=None
        self.in_lier_value=None
        self.lier=None
        self.lier_value=None
        self.has_in_lier=has_in_lier
        self._X_source=None
        self._Y_source=None
        self.out_lier_PC=PointCloud()
        self.in_lier_PC=PointCloud()
    
    def __call__(self):
        self.create_outlier()
        self._X_source=np.vstack([self._surface_points_down.point,self.lier])
        surface_value= np.zeros(self._surface_points_down.size)
        print("surface_value: ",surface_value.shape)
        self._Y_source=np.concatenate([surface_value,self.lier_value])
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
        self.out_lier_PC.point=self.out_lier
        self.out_lier_value=sdf[sdf>0]
        if self.has_in_lier:
            self.in_lier=query_points[sdf<-0,:]
            self.in_lier_PC.point=self.in_lier
            self.in_lier_value=sdf[sdf<-0]
        if self.in_lier is not None:
            self.lier=np.vstack([self.out_lier,self.in_lier])
            self.lier_value=np.concatenate([self.out_lier_value,self.in_lier_value])
        else:
            self.lier=self.out_lier
            self.lier_value=self.out_lier_value
    @property
    def surface_points_down(self):
        return self._surface_points_down
    
    def voxel_points(self,voxel_size):
        self.voxel_size=voxel_size
        while self._surface_points_down.size>2000:
            self._surface_points_down.pcd=self.surface_points.voxel_down_sample(self.voxel_size)
            self.voxel_size=self.voxel_size*1.01
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
    @property
    def minR(self):
        radius = pdist(self.X_source, metric="euclidean")
        return np.min(radius)
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
        return np.mean(np.abs(y_mean))
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
    def __init__(self):
        self._gpisModel = None
        self.voxel_size = 0.0001
        self.sumofDetla = 0
        self.T_last = Transformation()
        self.obj_value=10000
        self.obj_opt_min_=0
        self.obj_last=1000000
        self.l=0.01
        self.stop_condition=0.0001
        self.max_iteration=30
    @property
    def gpisModel(self):
        return self._gpisModel
    @gpisModel.setter
    def gpisModel(self,v:GPISModel):
        self._gpisModel=v
    @property
    def obj_opt_min(self):
        return self.obj_opt_min_
    @obj_opt_min.setter
    def obj_opt_min(self,v):
        self.obj_opt_min_=v
    def objective(self, target_points):
        return np.mean(np.abs(self._gpisModel.prediction(target_points)))
    def check_objective_improvement(self, target_points_updated):
        new_objective=self.objective(target_points=target_points_updated)
        return new_objective<self.obj_last
        
    def init(self, source: PointCloud = None, target: PointCloud = None):
        source_down, source_fpfh = source.preprocess_point_cloud(self.voxel_size, toNumpy=True,kdhyper=True)
        target_down, target_fpfh = target.preprocess_point_cloud(self.voxel_size, toNumpy=True,kdhyper=True)
        source_down_fpfh=PointCloud()
        target_down_fpfh=PointCloud()
        transform_es,source_down_fpfh_,target_down_fpfh_ = self.execute_registration_fpfh_pca_init(source_down, source_fpfh, target_down, target_fpfh)
        source_down_fpfh.point=source_down_fpfh_
        target_down_fpfh.point=target_down_fpfh_
        transform_target2source_list=[]
        transform_target2source=Transformation()
        transform_target2source.Transform=np.linalg.inv(transform_es.Transform)
        transform_target2source_list.append(transform_target2source)
        return transform_target2source_list,source_down_fpfh,target_down_fpfh
    def init4(self, source: PointCloud = None, target: PointCloud = None):
        source_down, source_fpfh = source.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        target_down, target_fpfh = target.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        source_down_fpfh=PointCloud()
        target_down_fpfh=PointCloud()
        transform_es,source_down_fpfh_,target_down_fpfh_ = self.execute_registration_fpfh_pca_4init(source_down, source_fpfh, target_down, target_fpfh)
        source_down_fpfh.point=source_down_fpfh_
        target_down_fpfh.point=target_down_fpfh_
        transform_target2source_list=[]
        for i in range(4):
            transform_target2source=Transformation()
            transform_target2source.Transform=np.linalg.inv(transform_es[i].Transform)
            transform_target2source_list.append(transform_target2source)
        return transform_target2source_list,source_down_fpfh,target_down_fpfh
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
        print("T_last:\n ",T_last.Transform)
        T_check=copy.deepcopy(T_last)
        J_last=1000
        for i in tqdm.tqdm(range(self.max_iteration)):
            target_points_update = self.updateTarget_Point(target_points, T_check)
            if (i % 1)==0:
                J=np.mean(np.abs(self._gpisModel.prediction(target_points_update)))
                print("update J: ",J," J_last: ",J_last," opt: ",self.obj_opt_min_)
                if np.abs(J-J_last)<0.0000001:
                    print("J-J_last can not improve")
                    return target_points_update, T_check.Transform
                else:
                    J_last=J
                if (np.abs(J-self.obj_opt_min)<=0.00005 or J<self.obj_opt_min_ ):
                    print("arrive to the opt")
                    return target_points_update, T_check.Transform
            start = time.time()
            se3_epsilon = self.updateGaussNewtonBasedPerturabation(target_points=target_points_update,l=self.l)
            #print("-->compute se3_epsilon: ",time.time()-start)
            start = time.time()
            T_update = self.update_transformation(se3_epsilon, T_check)
            #print("-->compute T_update: ",time.time()-start)
            T_check.Transform=T_update 
            
        return target_points_update, T_update 
    def update_transformation(self, se3_epsilon, T_last: Transformation):
        se3_epsilon=se3_epsilon.reshape(-1)
        return np.matmul(SE3.exp(se3_epsilon).as_matrix(), T_last.Transform)

    def updateTarget_Point(self, target_points: np.ndarray = None, transform: Transformation = None):
        if target_points.shape[1] == 4:
           PointXYZW=target_points
        if target_points.shape[1] == 3:
            PointXYZW=PointCloud.PointXYZ2homogeneous(target_points)
        target_transform=PointCloud.Homogeneous2PointXYZ((np.matmul(transform.Transform, PointXYZW.T)).T)
        return target_transform
    def updateGaussNewtonBasedPerturabation(self, target_points,l=0):
        JTJ, JTr = self.calculateTransformationPerturbation(target_points)
        JTJ_Hat = np.zeros_like(JTJ)
        diagonalS = np.zeros_like(JTJ)
        np.fill_diagonal(diagonalS,JTJ.diagonal())
        JTJ_Hat = JTJ + l * diagonalS
        c,low= cho_factor(JTJ_Hat)
        se3_epsilon =-1*cho_solve((c, low), JTr)
        return se3_epsilon
    def calculateTransformationPerturbation(self,target_points):
        N,_=target_points.shape
        BetaM = self.getBetaM(target_points)
        DeltaM = self.getDeltaM(target_points).reshape(N,-1,6)
        Alpha=self._gpisModel.Alpha.reshape(-1,1)
        #betaalpha = np.sum(BetaM@Alpha,axis=0).reshape(-1,1)
        betaalpha=BetaM@Alpha
        Alpha2=np.tile(np.expand_dims(Alpha,axis=0),(N,1,1))
        print("DeltaM: ",DeltaM.shape)
        DeltaMAlpha = np.transpose(np.transpose(Alpha2,(0,2,1))@DeltaM,(0,2,1))
        print("DeltaMAlpha: ",DeltaMAlpha.shape)
        JTJ = np.sum(DeltaMAlpha@DeltaMAlpha.reshape(N,1,6),axis=0) # 6\times 6
        JTr = np.sum(DeltaMAlpha@np.expand_dims(betaalpha,axis=1),axis=0) # 6 \times 1
        #raise ValueError("stop")
        return JTJ, JTr

    def getBetaM(self,target_points):
        BetaM=self._gpisModel.kernel_(target_points,self._gpisModel.X_source)
        return BetaM
    def getNormDerivative(self, point_source, point_target):
        N,_=point_target.shape
        N_source,_=point_source.shape
        target_points_copy=np.repeat(copy.deepcopy(point_target),N_source,axis=0)
        source_points_copy=np.tile(copy.deepcopy(point_source),(N,1))
        target_source_diff=(target_points_copy-source_points_copy)
        #norm_derivative=target_source_diff
        target_source_diff_abs=np.abs(target_source_diff)
        #target_source_diff_norm=normalize(target_source_diff_abs)
        #print(np.sign(target_source_diff).shape)
        norm_derivative=np.sign(target_source_diff)*target_source_diff_abs
        return norm_derivative
    def getDeltaM(self,target_points):
        N,_=target_points.shape
        N_source,_=self._gpisModel.X_source.shape
        Ty_odot = SE3.odot(PointCloud.PointXYZ2homogeneous(target_points))  # R^(N \times 4 \times 6)
        Ty_odot=np.repeat(Ty_odot,N_source,axis=0)
        #Ty_odot=np.tile(Ty_odot,(N_source,1,1))
        #print("Ty_odot: ",Ty_odot.shape)
        #print("self.gpisModel.Kernel.gradient(self.gpisModel.X_source, target_points): ",self.gpisModel.Kernel.gradient(self.gpisModel.X_source, target_points).shape)
        dk_dr = self._gpisModel.Kernel.gradient(self._gpisModel.X_source, target_points).T.reshape(-1,1)
        dr_dy=self.getNormDerivative(self._gpisModel.X_source,target_points)
        dk_dy=PointCloud.PointXYZ2homogeneous(dk_dr*dr_dy).reshape(N*N_source,1,4)
        deltaM=dk_dy@Ty_odot
        return deltaM
    def execute_registration_fpfh_pca_4init(self,source_down,source_fpfh,target_down, target_fpfh):
        corres_idx0, corres_idx1 = find_correspondences(source_fpfh, target_fpfh)
        
        source_down_fpfh = source_down[corres_idx0, :]
        target_down_fpfh = target_down[corres_idx1, :]
        source_pca_vectors=get_PCA_eigen_vector(source_down_fpfh)
        R_source=getAllRightHandCoordinate(source_pca_vectors)
        target_pca_vectors=get_PCA_eigen_vector(target_down_fpfh)
        R_target=getRightHandCoordinate(target_pca_vectors[:,0],target_pca_vectors[:,1],target_pca_vectors[:,2])
        R_es=[np.matmul(R_target,R.T) for R in R_source]

        source_down_fpfh_center=PointCloud.PointCenter(source_down_fpfh)
        target_down_fpfh_center=PointCloud.PointCenter(target_down_fpfh)
        trans=[target_down_fpfh_center-np.matmul(R_es_i,source_down_fpfh_center) for R_es_i in R_es]
        trans_es_list=[]
        for i in range(4):
            transform_es=Transformation()
            transform_es.rotation=R_es[i]
            transform_es.trans=trans[i]
            trans_es_list.append(transform_es)
        return trans_es_list,source_down_fpfh,target_down_fpfh

    def execute_registration_fpfh_pca_init(self,source_down,source_fpfh,target_down, target_fpfh):
        corres_idx0, corres_idx1 = find_correspondences(source_fpfh, target_fpfh)
        source_down_fpfh = source_down[corres_idx0, :]
        target_down_fpfh = target_down[corres_idx1, :]
        print("corresponding source points: ",source_down_fpfh.shape)
        print("corresponding target points: ",source_down_fpfh.shape)
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
        return transform_es,source_down_fpfh,target_down_fpfh


if __name__ == '__main__':
    X = np.random.rand(10, 3)
    y = np.random.rand(10, 1)
    kernel = SKWilliamsMinusKernel(3)
    gpis = GPISModel(kernel=SKWilliamsMinusKernel(3), random_state=0)
    gpis.fit(X, y)
