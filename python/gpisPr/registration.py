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

import open3d as o3d
import copy
from pointCloud import PointCloud
from utils import *
import transforms3d as t3d

class Registration:
    @staticmethod
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    @staticmethod
    def ICP_init(source, target, max_distance_threshold, transinit):
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, max_distance_threshold,
                                                              transinit,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                  max_iteration=2000)
                                                              )
        return reg_p2p.transformation

    @staticmethod
    def execute_registration_fpfh_pca_init(source_down,source_fpfh,target_down, target_fpfh):
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
    @staticmethod
    def execute_registration_gpis():
        pass
