import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import open3d as o3d
import copy
from gpisPr.pointCloud import PointCloud
from gpisPr.utils import *
import transforms3d as t3d
class Visualization:
    def __init__(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
    def animation(self,source,target,Transformation_list):

        self.vis.add_geometry(source)
        for update_T in Transformation_list:
            target_pcd=PointCloud()
            target_pcd.point=target
            self.vis.add_geometry(target_pcd.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
        self.vis.destroy_window()