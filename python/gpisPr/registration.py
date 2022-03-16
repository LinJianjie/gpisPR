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
from turtle import width

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import open3d as o3d
import copy
from pointCloud import PointCloud
from utils import *
import transforms3d as t3d

class Registration:
    @staticmethod
    def draw_registration_result(source:PointCloud,target:PointCloud, transformation:Transformation,source2target:True, window_name="Open3D",size=[1000,1000]):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.pcd.paint_uniform_color([1, 0.706, 0])
        target_temp.pcd.paint_uniform_color([0, 0.651, 0.929])
        if source2target:
            source_temp.transform(transformation.Transform)
        else:
            target_temp.transform(transformation.Transform)    
        o3d.visualization.draw_geometries([source_temp.pcd, target_temp.pcd],window_name =window_name,width=size[0],height=size[1])
    @staticmethod
    def draw_registraion_init(source:PointCloud,target:PointCloud,window_name="Open3D",size=[1000,1000]):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.pcd.paint_uniform_color([1, 0.706, 0])
        target_temp.pcd.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([source_temp.pcd, target_temp.pcd],window_name =window_name,width=size[0],height=size[1])
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
    def execute_registration_gpis():
        pass
    