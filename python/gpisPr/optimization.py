# Copyright (c) 2022 Jianjie Lin
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import sys
from telnetlib import SE
from tkinter.messagebox import NO

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import transforms3d as t3d
from sklearn.decomposition import PCA
import open3d as o3d
from pointCloud import PointCloud
from registration import Registration
import numpy as np
from utils import *
from gpis import GPIS
from scipy.linalg import cho_factor, cho_solve
from liegroups.liegroups import SE3

class Optimization:
    def __init__(self, voxel_size, gpis:GPIS=None):
        self.voxel_size = voxel_size
        self.gpisGR=gpis
        self.sumofDetla=0
        self.X_target_update=self.gpisGR.X_target
        self.T=Transformation()
    def init(self, source:PointCloud=None, target:PointCloud=None):
        source_down, source_fpfh = source.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        target_down, target_fpfh = target.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        transform_es=Registration.execute_registration_fpfh_pca_init(source_down,source_fpfh,target_down, target_fpfh)
        #Registration.draw_registration_result(source.pcd,target.pcd,transform_es.Transform)

    def updateGaussNewtonBasedPerturabation(self,l):
        JTJ,JTr=self.calculateTransformationPerturbation()
        JTJ_Hat=np.zeros(JTJ.shape[0],JTr.shape[1])
        diagonalS=np.zeros(JTJ.shape[0],JTr.shape[1])
        diagonalS.diagonal=JTJ.diagonal
        JTJ_Hat = JTJ + l * diagonalS
        c, low = cho_factor(JTJ_Hat)
        se3_epsilon=cho_solve((c,low),JTr)
        return se3_epsilon
    def update_espsilon(self,se3_epsion):
        pass
    def update_transformation(self):
        pass
    def updatePoint(self):
        pass
    def calculateTransformationPerturbation(self):
        self.updatePoint()
        BetaM = self.getBetaM()
        DeltaM = self.getDeltaM()
        betaalpha = np.matmul(BetaM.T,self.gpisGR.alpha_)
        DeltaMAlpha = np.matmul(self.gpisGR.alpha_, DeltaM)
        JTJ = np.matmul(DeltaMAlpha,DeltaMAlpha.T)
        JTr = np.matmul(DeltaMAlpha,betaalpha)
        return JTJ, JTr
    def getBetaM(self):
        return self.gpisGR.kernel_(self.gpisGR.X_train_,self.X_target_update)
    def getDeltaM(self):
        Ty_odot=SE3.odot(self.X_target_update)
        dk_dy=self.gpisGR.kernel.gradient(self.gpisGR.X_train_,self.X_target_update)*(self.X_target_update-self.gpisGR.X_train_)
        return Ty_odot*dk_dy
    def execute(self):
        pass


if __name__ == '__main__':
    source = PointCloud(filename="../data/happy.pcd")
    source()
    target = PointCloud(filename="../data/happy.pcd")
    target()
    transinit = Transformation()
    transinit.trans = np.asarray([0.1, 0.2, 0.2])
    rot = t3d.euler.euler2mat(DEG2RAD(20.0), DEG2RAD(40.0), DEG2RAD(30.0), 'sxyz')
    transinit.rotation = rot
    target.transform(transinit=transinit.Transform)
    opt = Optimization(voxel_size=0.01)
    opt.init(source=source, target=target)
    
    #p1=np.matmul(transinit.rotation,source.point.T)
    #p2=target.point.T
    #print(p2-p1)