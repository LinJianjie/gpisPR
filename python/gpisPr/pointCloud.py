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
import numpy as np
import open3d as o3d
import transforms3d as t3d
import copy 
from scipy.spatial.distance import pdist
import trimesh
import time
class PointCloud:
    def __init__(self, filename=None):
        self._path = filename
        self._pcd = o3d.geometry.PointCloud()
        self._pcd_down = o3d.geometry.PointCloud()
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
    def size_down(self):
        return self.point_down.shape[0]
    @property
    def point(self):
        return PointCloud.pca2xyz(self._pcd)
    @point.setter
    def point(self, v):
        self._pcd = PointCloud.xyz2pcd(v)
        if self.size_down==0:
            self._pcd_down=self._pcd
    @property
    def point_down(self):
        return PointCloud.pca2xyz(self._pcd_down)
    

    @property
    def pcd(self):
        return self._pcd

    @pcd.setter
    def pcd(self, v):
        self._pcd = v
    @property
    def pcd_down(self):
        return self._pcd_down
    @pcd_down.setter
    def pcd_down(self,v):
        self._pcd_down=v
    @property
    def path(self):
        return self._path

    def voxel_down_sample(self,voxel_size,toNumpy=False,inline=False):
        self.voxel_size=voxel_size
        pcd_down = self._pcd.voxel_down_sample(self.voxel_size)
        if inline:
            self.pcd_down=pcd_down
        else:
            if toNumpy:
                return PointCloud.pca2xyz(pcd_down)
            return pcd_down
    def fps(self, n_samples):
        """
        points: [N, 3] array containing the whole point cloud
        n_samples: samples you want in the sampled point cloud typically << N 
        """
        points=self.point
        points = np.array(points)
        
        # Represent the points by their indices in points
        points_left = np.arange(len(points)) # [P]

        # Initialise an array for the sampled indices
        sample_inds = np.zeros(n_samples, dtype='int') # [S]

        # Initialise distances to inf
        dists = np.ones_like(points_left) * float('inf') # [P]

        # Select a point from points by its index, save it
        selected = 0
        sample_inds[0] = points_left[selected]

        # Delete selected 
        points_left = np.delete(points_left, selected) # [P - 1]

        # Iteratively select points for a maximum of n_samples
        for i in range(1, n_samples):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]
            
            dist_to_last_added_point = (
                (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_last_added_point, 
                                            dists[points_left]) # [P - i]

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]

            # Update points_left
            points_left = np.delete(points_left, selected)

        return points[sample_inds]

    def compute_fpfh(self,pcd_down,toNumpy=False):
        if not pcd_down.has_normals:
            pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5.0,
                                                                       max_nn=100))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(
                                                                       radius=self.voxel_size * 5.0,
                                                                                                        max_nn=100))
        if toNumpy:
           return np.array(pcd_fpfh.data).T                                                                                            
        return pcd_fpfh                      
                                                                                   
    def preprocess_point_cloud(self, voxel_size, toNumpy=False,kdhyper=True,fps=False):
        self.pcd_down=self._pcd
        if fps:
            start=time.time()
            down_samppling=self.fps(n_samples=9000)
            print("it takes: ",time.time()-start)
            print("down_sampling: ",down_samppling.shape)
            raise ValueError("stop")

        else:
            if self.size>10000:
                i=0
                start=time.time()
                while self.size_down > 10000:
                    self.voxel_down_sample(voxel_size,inline=True)
                    voxel_size=voxel_size*1.1
                #print("it takes: ",time.time()-start)
                #print("down_sampling: ",self.size_down)
                #raise ValueError("stop")
            else:
                voxel_size=0.0001 # TODO how to get the propery voxel
        print("size of down: ",self.size_down)
        pcd_down=self.pcd_down
        if kdhyper:
            pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*50,max_nn=30))
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*50,max_nn=30))
            print(" finished")                                                   
        else:
            pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(knn=50))
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                                    o3d.geometry.KDTreeSearchParamKNN(knn=50))
        if toNumpy:
            return PointCloud.pca2xyz(pcd_down), np.array(pcd_fpfh.data).T
        else:
            return pcd_down, pcd_fpfh
    @property
    def minDis(self):
        radius = pdist(self.point, metric="euclidean")
        return np.min(radius)

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
