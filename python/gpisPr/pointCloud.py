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
from tkinter.messagebox import NO

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import open3d as o3d
import transforms3d as t3d
import copy 
import trimesh

class PointCloud:
    def __init__(self, filename=None):
        self._path = filename
        self._pcd = o3d.geometry.PointCloud()
        self.normal=None
        self.voxel_size=0.00001

    def __call__(self):
        ending=self._path.split(".")[-1]
        if ending=="pcd":
            print("===> load pcd")
            self.load_pcd()
        if ending=="ply":
            print("===> load ply")
            self.load_mesh_and_sample()
        if ending=="h5":
            print("===> load h5")
            self.load_completion3D()

    def load_pcd(self):
        self._pcd = o3d.io.read_point_cloud(self.path)
    def load_mesh_and_sample(self,sample_point_count=2048):
        mesh=trimesh.load_mesh(self.path)
        points = mesh.sample(sample_point_count, return_index=False)
        self._pcd=PointCloud.xyz2pcd(points)

    def load_completion3D(self):
        pass
    def centralized(self):
        self._pcd.translate(self._pcd.get_center()*-1)
    def scale(self,scale_):
        self.centralized()
        self._pcd.scale(scale_,center=self._pcd.get_center())
    def transform(self, transinit):
        self._pcd.transform(transinit)
    def get_center(self):
        return self._pcd.get_center()
    def estimate_normal(self,radius=0.1,max_knn=30):
        self.pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius,
                                                                       max_nn=max_knn))
    @property
    def normal(self):
        return np.asarray(self.pcd.normals)
    @normal.setter
    def normal(self, value):
        self._normal=value
    @property
    def size(self):
        return self.point.shape[0]
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

    def voxel_down_sample(self,voxel_size,toNumpy=False):
        self.voxel_size=voxel_size
        pcd_down = self._pcd.voxel_down_sample(self.voxel_size)
        if toNumpy:
            return PointCloud.pca2xyz(pcd_down)
        return pcd_down

    def compute_fpfh(self,pcd_down,toNumpy=False):
        if not pcd_down.has_normals:
            pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2.0,
                                                                       max_nn=30))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5.0,
                                                                                                        max_nn=100))
        if toNumpy:
           return np.array(pcd_fpfh.data).T                                                                                            
        return pcd_fpfh                      
                                                                                   
    def preprocess_point_cloud(self, voxel_size, toNumpy=False):
        pcd_down = self.voxel_down_sample(voxel_size)
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
        self._pcd.paint_uniform_color([1, 0.706, 0])
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
    @staticmethod
    def PointXYZ2homogeneous(pointsXYZ):
        homo_points=np.ones([pointsXYZ.shape[0],4])
        homo_points[:,:3]=pointsXYZ
        return homo_points
    @staticmethod
    def Homogeneous2PointXYZ(home_points):
        return home_points[:,:3]
if __name__ == '__main__':
    source = PointCloud(filename="../data/bunny_1420.pcd")
    source()
    pcd_down, pcd_fpfh = source.preprocess_point_cloud(
        voxel_size=0.001, toNumpy=True)
    print(pcd_down.shape)
    print(pcd_fpfh.shape)
