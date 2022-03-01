# Copyright (c) 2022 Jianjie Lin
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT



from operator import le
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
