# Copyright (c) 2022 Jianjie Lin
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import math
from sklearn.neighbors import KDTree
import open3d as o3d
from pointCloud import PointCloud


def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(
        unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(
            amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]


class BadMeshException(Exception):
    pass


class Point2SDF:
    def __init__(self, pointCloud: PointCloud) -> None:
        self.points = pointCloud.point
        self.normals = pointCloud.normal
        self.kd_tree = KDTree(self.points)

    def get_random_surface_points(self, count, use_scans=True):
        if use_scans:
            indices = np.random.choice(self.points.shape[0], count)
            return self.points[indices, :]
        else:
            return self.mesh.sample(count)

    def get_sdf(self, query_points, use_depth_buffer=False, sample_count=11, return_gradients=False):
        if use_depth_buffer:
            distances, indices = self.kd_tree.query(query_points)
            distances = distances.astype(np.float32).reshape(-1)
            inside = ~self.is_outside(query_points)
            distances[inside] *= -1

            if return_gradients:
                gradients = query_points - self.points[indices[:, 0]]
                gradients[inside] *= -1

        else:
            distances, indices = self.kd_tree.query(
                query_points, k=sample_count)
            distances = distances.astype(np.float32)

            closest_points = self.points[indices]
            direction_from_surface = query_points[:,
                                                  np.newaxis, :] - closest_points
            inside = np.einsum(
                'ijk,ijk->ij', direction_from_surface, self.normals[indices]) < 0
            inside = np.sum(inside, axis=1) > sample_count * 0.5
            distances = distances[:, 0]
            distances[inside] *= -1

            if return_gradients:
                gradients = direction_from_surface[:, 0]
                gradients[inside] *= -1

        if return_gradients:
            near_surface = np.abs(distances) < math.sqrt(
                0.0025**2 * 3) * 3  # 3D 2-norm stdev * 3
            gradients = np.where(
                near_surface[:, np.newaxis], self.normals[indices[:, 0]], gradients)
            gradients /= np.linalg.norm(gradients, axis=1)[:, np.newaxis]
            return distances, gradients
        else:
            return distances

    def get_sdf_in_batches(self, query_points, use_depth_buffer=False, sample_count=11, batch_size=1000000, return_gradients=False):
        if query_points.shape[0] <= batch_size:
            return self.get_sdf(query_points, use_depth_buffer=use_depth_buffer, sample_count=sample_count, return_gradients=return_gradients)

        n_batches = int(math.ceil(query_points.shape[0] / batch_size))
        batches = [
            self.get_sdf(points, use_depth_buffer=use_depth_buffer,
                         sample_count=sample_count, return_gradients=return_gradients)
            for points in np.array_split(query_points, n_batches)
        ]
        if return_gradients:
            distances = np.concatenate([batch[0] for batch in batches])
            gradients = np.concatenate([batch[1] for batch in batches])
            return distances, gradients
        else:
            return np.concatenate(batches)  # distances

    def sample_sdf_near_surface(self, number_of_points=1000, use_scans=True, sign_method='normal', normal_sample_count=11, min_size=0, return_gradients=False):
        query_points = []
        surface_sample_count = int(number_of_points * 47 / 50) // 2
        surface_points = self.get_random_surface_points(
            surface_sample_count, use_scans=use_scans)
        query_points.append(
            surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
        query_points.append(
            surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))

        unit_sphere_sample_count = number_of_points - \
            surface_points.shape[0] * 2
        unit_sphere_points = sample_uniform_points_in_unit_sphere(
            unit_sphere_sample_count)
        query_points.append(unit_sphere_points)
        query_points = np.concatenate(query_points).astype(np.float32)

        if sign_method == 'normal':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=False,
                                          sample_count=normal_sample_count, return_gradients=return_gradients)
        elif sign_method == 'depth':
            sdf = self.get_sdf_in_batches(
                query_points, use_depth_buffer=True, return_gradients=return_gradients)
        else:
            raise ValueError(
                'Unknown sign determination method: {:s}'.format(sign_method))
        if return_gradients:
            sdf, gradients = sdf

        if min_size > 0:
            model_size = np.count_nonzero(
                sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
            if model_size < min_size:
                raise BadMeshException()

        if return_gradients:
            return query_points, sdf, gradients
        else:
            return query_points, sdf


if __name__ == "__main__":
    source = PointCloud(filename="../data/happy.pcd")
    source()
    source.estimate_normal(0.1, 30)
    point2sdf = Point2SDF(source)
    query_points, sdf=point2sdf.sample_sdf_near_surface()
    #source.visualize()
    outer=PointCloud.xyz2pcd(query_points[sdf>0,:])
    inline=PointCloud.xyz2pcd(query_points[sdf<0,:])
    source.pcd.paint_uniform_color([1, 0.706, 0])

    inline.paint_uniform_color([1, 0, 0])
    outer.paint_uniform_color([0, 0, 1])

    inlinePC=PointCloud()
    inlinePC.pcd=inline

    outlinePC=PointCloud()
    outlinePC.pcd=outer
    
    #PointCloud.vis([source,inlinePC,outlinePC])
    #print(outer)
    #print(inline)
    #print(query_points[sdf<0,:])
    print(sdf[:30])
