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


from numpy import source
from  liegroups import SE3
from utils import *
import transforms3d as t3d
from gpis import GPISModel,GPISData,GPISOpt,ConsitionPSDGPISModel
from skkernel import SKWilliamsMinusKernel,SKWilliamsPlusKernel,SKRBF,SKMatern
from sklearn.gaussian_process.kernels import RBF
from pointCloud import PointCloud
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
    gpis = GPISModel(kernel=SKWilliamsMinusKernel(3), random_state=0)
    K_gradient=gpis.Kernel.gradient(X_source,X_target)
    print(K_gradient.shape)
def test_optimization():
    gpisModel = GPISModel(kernel=SKWilliamsMinusKernel(3), random_state=0)
    X_source=np.random.rand(100,3)
    y_source=np.random.randint(2,size=100)
    target_points=np.random.rand(10,3)
    print("X_source:",X_source.shape)
    print("y_source:",y_source.shape)
    print("target_points:",target_points.shape)
    gpisModel.X_source=X_source
    gpisModel.Y_source=y_source
    gpisModel.fit(X_source,y_source)
    #print("alpha:",gpis.Alpha.shape)
    opt = GPISOpt(voxel_size=0.01,gpisModel=gpisModel)
    #BetaM=opt.getBetaM(target_points=target_points)
    #print("BetaM:", BetaM.shape)
    #DeltaM=opt.getDeltaM(target_points=target_points)
    #opt.calculateTransformationPerturbation(target_points=target_points)
    opt.updateGaussNewtonBasedPerturabation(targe_points=target_points,l=0.1)

def test_gpis():
    path="/home/lin/Workspace/Projetcs/github/gpisPR/python/data/bunny_1420.pcd"
    surface_points = PointCloud(filename=path)
    surface_points()
    gpisData=GPISData(surface_points=surface_points,num_out_lier=500)
    gpisData.voxel_points(0.003)
    gpisData()
    print("gpisData.X_source: ",gpisData.X_source.shape)
    #pisModel = ConsitionPSDGPISModel(kernel=SKWilliamsMinusKernel(R=gpisData.maxR), random_state=0)
    #gpisModel = GPISModel(kernel=SKWilliamsMinusKernel(R=gpisData.maxR), random_state=0)
    #gpisModel = GPISModel(kernel=SKRBF(length_scale=1,length_scale_bounds="fixed"), random_state=0)
    gpisModel = GPISModel(kernel=SKMatern(length_scale=1,length_scale_bounds="fixed",nu=1.5), random_state=0)
    #gpisModel.fit(gpisData.X_source,gpisData.Y_source)
    gpisModel.fit(gpisData.X_source,gpisData.Y_source)

    ##    
    target = PointCloud(filename=path)
    target()
    transinit = Transformation()
    transinit.trans = np.asarray([0.1, 0.2, 0.2])
    rot = t3d.euler.euler2mat(
        DEG2RAD(0.0), DEG2RAD(180.0), DEG2RAD(0.0), 'sxyz')
    transinit.rotation = rot
    target.transform(transinit.Transform)
    print("start to predict")
    #PointCloud.vis([source,target])
    import time
    start = time.time()
    y_mean_1=gpisModel.prediction(target.point)
    end = time.time()
    print("first: ",end - start)
    print("tar y_mean_1: ",np.sum(np.abs(y_mean_1)))
    K_gradient=gpisModel.Kernel.gradient(gpisData.X_source,target.point)
    print("K_gradient: ",K_gradient.shape)
    y_mean_2=gpisModel.prediction(surface_points.point)
    print("sur y_mean_2: ",np.sum(np.abs(y_mean_2)))
    #print("surface std_mean: ",y_std)
    #print("surface_value: ",surface_value[:index])

if __name__=="__main__":
    #test_se3() # checked
    #test_gpis_kernel() # checked
    #test_optimization()
    test_gpis()