
import copy
import sys
import os
import transforms3d as t3d
import open3d as o3d
import time 
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from gpisPr.pointCloud import PointCloud
from gpisPr.utils import *
from gpisPr.registration import Registration
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],width=1080,height=1000
                                      )

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 100
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 50
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    file_path=os.path.join(os.path.dirname(__file__),"../../","data/bunny_1420.pcd")
    source_surface = PointCloud(filename=file_path)
    source_surface()
    source_surface.scale(scale_=1)
    target_surface=copy.deepcopy(source_surface)
    target_surface.add_Gaussian_noise(0.001)
    source_surface.add_Gaussian_noise(0.001)
    transinit = Transformation()
    transinit.setT(trans=np.asarray([0.1, 0.01, 0.00]),rot_deg=[0,90,0])
    target_surface.transform(transinit.Transform)
    print(":: Load two point clouds and disturb initial pose.")
    
    source = source_surface.pcd
    target = target_surface.pcd
    
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 500
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

start = time.time()
voxel_size = 0.0001  # means 5cm for the dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size)
result_fast = execute_fast_global_registration(source_down, target_down,
                                               source_fpfh, target_fpfh,
                                               voxel_size)
print("Fast global registration took %.3f sec.\n" % (time.time() - start))
print(result_fast)
print("result_fast.transformation\n:",result_fast.transformation)
draw_registration_result(source_down, target_down, result_fast.transformation)

