# Copyright (c) 2022 Jianjie Lin
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from tkinter.messagebox import NO
import numpy as np
import open3d as o3d
import transforms3d as t3d
import copy 
from typing import List

class PointCloud:
    def __init__(self, filename=None):
        self._path = filename
        self._pcd = o3d.geometry.PointCloud()

    def __call__(self):
        self._pcd = o3d.io.read_point_cloud(self.path)

    def transform(self, transinit):
        self._pcd.transform(transinit)

    @property
    def point(self):
        return PointCloud.pca2xyz(self._pcd)

    @point.setter
    def point(self, v):
        self._pcd = PointCloud.xyz2pcd(v)

    @property
    def pcd(self):
        return self._pcd

    @pcd.setter
    def pcd(self, v):
        self._pcd = v

    @property
    def path(self):
        return self._path

    def preprocess_point_cloud(self, voxel_size, toNumpy=False):
        pcd_down = self._pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                                       max_nn=30))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                                                                                        max_nn=100))
        if toNumpy:
            return PointCloud.pca2xyz(pcd_down), np.array(pcd_fpfh.data).T
        else:
            return pcd_down, pcd_fpfh

    def visualize(self):
        o3d.visualization.draw_geometries([self._pcd])

    @staticmethod
    def vis(pcd_list):
        pcd_=[]
        for pointcoud in pcd_list:
            pcd_.append(pointcoud.pcd)
        o3d.visualization.draw_geometries(pcd_)

    @staticmethod
    def pca2xyz(pca):
        return np.array(pca.points)

    @staticmethod
    def xyz2pcd(pointXYZ):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointXYZ)
        return pcd

    @staticmethod
    def PointCenter(pointxyz):
        return PointCloud.PCDCenter(PointCloud.xyz2pcd(pointxyz))

    @staticmethod
    def PCDCenter(pcd):
        return pcd.get_center()


if __name__ == '__main__':
    source = PointCloud(filename="../data/bunny_1420.pcd")
    source()
    pcd_down, pcd_fpfh = source.preprocess_point_cloud(
        voxel_size=0.001, toNumpy=True)
    print(pcd_down.shape)
    print(pcd_fpfh.shape)
