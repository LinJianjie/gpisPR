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


from  liegroups import SE3
from utils import *
import transforms3d as t3d
from gpis import GPIS,GPISData
from skkernel import SKWilliamsMinusKernel,SKWilliamsPlusKernel
from sklearn.gaussian_process.kernels import RBF
from optimization import Optimization
from pointCloud import PointCloud
from point2SDF import Point2SDF
def test_se3():
    X=Transformation()
    X.trans=np.asarray([2,3,4])
    X.rotation=t3d.euler.euler2mat(DEG2RAD(20.0), DEG2RAD(40.0), DEG2RAD(30.0), 'sxyz')
    print(X.Transform)
    p2 = np.asarray([2, 2, 3, 1])
    odot2 = SE3.odot(p2)
    print("odot2:",odot2)
    p22=np.matmul(X.Transform,p2)
    print("p22:",p22)
    p33=np.asarray([[1, 2, 3, 1],[2,3,4,1],[4,5,5,1]])
    print(p33.shape)
    odot33 = SE3.odot(p33)
    print(odot33.shape)
def test_gpis_kernel():
    X_source = np.random.rand(100, 3)
    y_source = np.random.rand(100, 1)
    X_target=np.random.rand(10, 3)
    kernel = SKWilliamsMinusKernel(3)
    gpis = GPIS(kernel=SKWilliamsMinusKernel(3), random_state=0)
    K_gradient=gpis.Kernel.gradient(X_source,X_target)
    print(K_gradient.shape)
def test_optimization():
    gpis = GPIS(kernel=SKWilliamsMinusKernel(3), random_state=0)
    X_source=np.random.rand(100,3)
    y_source=np.random.randint(2,size=100)
    target_points=np.random.rand(10,3)
    print("X_source:",X_source.shape)
    print("y_source:",y_source.shape)
    print("target_points:",target_points.shape)
    gpis.X_source=X_source
    gpis.Y_source_value=y_source
    gpis.fit(X_source,y_source)
    #print("alpha:",gpis.Alpha.shape)
    opt = Optimization(voxel_size=0.01,gpis=gpis)
    #BetaM=opt.getBetaM(target_points=target_points)
    #print("BetaM:", BetaM.shape)
    #DeltaM=opt.getDeltaM(target_points=target_points)
    #opt.calculateTransformationPerturbation(target_points=target_points)
    opt.updateGaussNewtonBasedPerturabation(targe_points=target_points,l=0.1)

def test_gpis():
    source = PointCloud(filename="../data/happy.pcd")
    source()
    source_down = PointCloud()
    source_down.pcd=source.voxel_down_sample(0.005)
    print("source_down:",source_down.size)
    source_down.estimate_normal(0.001, 30)
    point2sdf = Point2SDF(source_down)
    query_points, sdf=point2sdf.sample_sdf_near_surface(number_of_points=1000)
    #
    outer=query_points[sdf>0,:]
    outer_value=sdf[sdf>0]
    #outer=query_points
    #outer_value=sdf
    gpisData=GPISData()
    surface_points_outerlines_points=np.vstack([source_down.point,outer])
    surface_value=np.zeros(source_down.size)
    surface_points_outerlines_points_value=np.concatenate([surface_value,outer_value])
    print("surface_points_outerlines_points: ",surface_points_outerlines_points.shape)
    print("surface_points_outerlines_points_value: ",surface_points_outerlines_points_value.shape)
    gpisData.surface_points=surface_points_outerlines_points
    gpisData.surface_value=surface_points_outerlines_points_value
    gpisData.compute_max_radius()

    gpis = GPIS(kernel=SKWilliamsMinusKernel(gpisData.maxR), random_state=0)
    gpis.fit(gpisData.surface_points,gpisData.surface_value)
    ##
    index=5
    #print("point: ",outer[:index,:])
    gpis.X_source=gpisData.surface_points
    y_mean,y_std=gpis.predict(outer,return_std=True)
    
    #print("outer y_mean: ",np.power(y_mean-outer_value,2))
    #print("outer std_mean: ",y_std)
    #print("outer_value: ",outer_value)

    target = PointCloud(filename="../data/happy.pcd")
    target()
    transinit = Transformation()
    transinit.trans = np.asarray([0.1, 0.2, 0.2])
    rot = t3d.euler.euler2mat(
        DEG2RAD(0.0), DEG2RAD(180.0), DEG2RAD(0.0), 'sxyz')
    transinit.rotation = rot
    target.transform(transinit.Transform)
    PointCloud.vis([source,target])
    y_mean_1=gpis.predict(target.point,return_std=False)
    print("tar y_mean_1: ",np.sum(np.abs(y_mean_1)))
    y_mean_2=gpis.predict(source.point,return_std=False)
    print("sur y_mean_2: ",np.sum(np.abs(y_mean_2)))
    #print("surface std_mean: ",y_std)
    #print("surface_value: ",surface_value[:index])

if __name__=="__main__":
    #test_se3() # checked
    #test_gpis_kernel() # checked
    #test_optimization()
    test_gpis()