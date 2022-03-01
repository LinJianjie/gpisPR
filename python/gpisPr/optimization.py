# Copyright (c) 2022 Jianjie Lin
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import transforms3d as t3d
from sklearn.decomposition import PCA
import open3d as o3d
from pointCloud import PointCloud
from registration import Registration
import numpy as np
from utils import *


class Optimization:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def init(self, source:PointCloud=None, target:PointCloud=None):
        source_down, source_fpfh = source.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        target_down, target_fpfh = target.preprocess_point_cloud(self.voxel_size, toNumpy=True)
        transform_es=Registration.execute_registration_fpfh_pca_init(source_down,source_fpfh,target_down, target_fpfh)
        Registration.draw_registration_result(source.pcd,target.pcd,transform_es.Transform)

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