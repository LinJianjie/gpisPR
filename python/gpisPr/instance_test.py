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


import copy
from matplotlib.transforms import Transform

from numpy import source
from  liegroups import SE3
from utils import *
import transforms3d as t3d
from gpis import GPISModel,GPISData,GPISOpt
from skkernel import SKWilliamsPlusKernel,SKWilliamsMinusKernel,SKRBF,SKMatern
from pointCloud import PointCloud
from registration import Registration
import sys
import os

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
    transinit = Transformation()
    transinit.trans = np.asarray([0.1, 0.2, 0.])
    rot = t3d.euler.euler2mat(DEG2RAD(50.0), DEG2RAD(180.0), DEG2RAD(60.0), 'sxyz')
    transinit.rotation = rot
    target_update=opt.updateTarget_Point(target_points=target_points,transform=transinit)
    print(target_update.shape)
    #BetaM=opt.getBetaM(target_points=target_points)
    #print("BetaM:", BetaM.shape)
    #DeltaM=opt.getDeltaM(target_points=target_points)
    #opt.calculateTransformationPerturbation(target_points=target_points)
    opt.updateGaussNewtonBasedPerturabation(targe_points=target_points,l=0.1)

def test_gpis():
    print(os.getcwd())
    file_path=os.path.join(os.path.dirname(__file__),"../","data/happy.pcd")
    surface_points = PointCloud(filename=file_path)
    surface_points()
    gpisData=GPISData(surface_points=surface_points,num_out_lier=500)
    gpisData.voxel_points(0.005)
    gpisData()
    print("gpisData.X_source: ",gpisData.X_source.shape)
    #pisModel = ConsitionPSDGPISModel(kernel=SKWilliamsMinusKernel(R=gpisData.maxR), random_state=0)
    gpisModel = GPISModel(kernel=SKWilliamsMinusKernel(R=gpisData.maxR), random_state=0)
    #gpisModel = GPISModel(kernel=SKRBF(length_scale=1,length_scale_bounds="fixed"), random_state=0)
    #gpisModel = GPISModel(kernel=SKMatern(length_scale=1,length_scale_bounds="fixed",nu=1.5), random_state=0)
    #gpisModel.fit(gpisData.X_source,gpisData.Y_source)
    gpisModel.fit(gpisData.X_source,gpisData.Y_source)
    target = PointCloud(filename=file_path)
    target()
    transinit = Transformation()
    transinit.trans = np.asarray([0.1, 0., 0.])
    rot = t3d.euler.euler2mat(
        DEG2RAD(90.0), DEG2RAD(0.0), DEG2RAD(0.0), 'sxyz')
    transinit.rotation = rot
    target.transform(transinit.Transform)
    print("start to predict")
    #PointCloud.vis([source,target])
    print("size of target: ",target.point.shape)
    import time
    start = time.time()
    y_mean_1=gpisModel.prediction(target.point)
    end = time.time()
    print("first: ",end - start)
    print("tar y_mean: ",np.mean(np.abs(y_mean_1)))
    #K_gradient=gpisModel.Kernel.gradient(gpisData.X_source,target.point)
    #print("K_gradient: ",K_gradient.shape)
    print("size of surface: ",surface_points.point.shape)
    y_mean_2=gpisModel.prediction(surface_points.point)
    print("sur y_mean: ",np.mean(np.abs(y_mean_2)))
    #print("surface std_mean: ",y_std)
    #print("surface_value: ",surface_value[:index])
def test_gpisOpt():
    vis=True
    print("=====> Prepare the Point Cloud Data")
    file_path=os.path.join(os.path.dirname(__file__),"../","data/happy.pcd")
    source_surface = PointCloud(filename=file_path)
    source_surface()
    # TODO, check the scale, und rescale, using the bounding box
    source_surface.scale(scale_=1)
    #raise ValueError("stop")
    target_surface=copy.deepcopy(source_surface)
    transinit = Transformation()
    transinit.trans = np.asarray([0, 0, 0.])
    rot = t3d.euler.euler2mat(DEG2RAD(90.0), DEG2RAD(180.0), DEG2RAD(0.0), 'sxyz')
    transinit.rotation = rot
    target_surface.transform(transinit.Transform)
    if True:
        Registration.draw_registraion_init(source=source_surface,target=target_surface)

    # prepera the GPIS Data
    print("=====> Prepare the GPIS Data")
    voxel_size=0.0001
    gpisData=GPISData(surface_points=source_surface,num_in_out_lier=1000,has_in_lier=False)
    gpisData.voxel_points(voxel_size)
    gpisData()
    #Registration.draw_registraion_init(source=gpisData.in_lier_PC,target=gpisData.out_lier_PC)
    #raise ValueError("Stop")
    print("gpisData.X_source: ",gpisData.X_source.shape)
    target_surface.voxel_down_sample(voxel_size=gpisData.voxel_size,inline=True)
    source_surface.voxel_down_sample(voxel_size=gpisData.voxel_size,inline=True)


    print("====> Prepare the GPIS Model")
    print("gpisData.maxR: ",gpisData.maxR)
    #gpisModel = GPISModel(kernel=SKWilliamsPlusKernel(R=gpisData.maxR,alpha=0.1), random_state=0)
    gpisModel = GPISModel(kernel=SKWilliamsMinusKernel(R=gpisData.maxR,alpha=0.01), random_state=0)
    #gpisModel=GPISModel(kernel=SKRBF(length_scale=1,length_scale_bounds="fixed"), random_state=0)
    #gpisModel = GPISModel(kernel=SKMatern(length_scale=10,length_scale_bounds="fixed",nu=1.5), random_state=0)
    gpisModel.fit(gpisData.X_source, gpisData.Y_source)
    y_mean_source=gpisModel.prediction(source_surface.point_down)
    print("source: ",y_mean_source)
    y_mean_1=gpisModel.prediction(target_surface.point_down)
    print("original target: ",y_mean_1)

    print("=====> start to optimiztion")

    opt = GPISOpt(voxel_size=voxel_size)
    opt.gpisModel=gpisModel

    #transform_target2source=opt.init(source_surface,target_surface)
    transform_target2source,_,_=opt.init4(source_surface,target_surface)
    #print("transform_target2source:\n ",transform_target2source.Transform)
    init_target_value=[]
    for i in range(4):
        target_points_init = opt.updateTarget_Point(target_surface.point_down, transform_target2source[i])
        y_mean_2=gpisModel.prediction(target_points_init)
        print("init target: ",y_mean_2)
        if vis:
            Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transform_target2source[i],source2target=False)
    raise ValueError("stop")
    # T*source_target
    opt.obj_opt_min_=y_mean_source

    print(" start with ICP init")
    icp_transformation=Registration.ICP_init(source=source_surface._pcd_down,target=target_surface._pcd_down,max_distance_threshold=1,transinit=transform_target2source.Transform)
    if vis:
        ICP_transformation=Transformation()
        ICP_transformation.Transform=icp_transformation
        Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=ICP_transformation,source2target=True)

    #raise ValueError("stop")

    target_points_update, T_update=opt.step(target_points=target_surface.point_down,T_last=Transformation())
    #target_points_update, T_update=opt.step(target_points=target_surface.point_down,T_last=transform_target2source)
    #print("T_update:\n",T_update)
    transformation_update=Transformation()
    transformation_update.Transform=T_update
    #target_points_update = opt.updateTarget_Point(target_surface.point, transformation_update)
    if True:
        Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transformation_update,source2target=False)

    if False:
        print(" conitinous with ICP init")
        icp_transformation_update=Registration.ICP_init(source=source_surface._pcd_down,target=target_surface._pcd_down,max_distance_threshold=10,transinit=np.linalg.inv(T_update))
        if vis:
            ICP_transformation=Transformation()
            ICP_transformation.Transform=icp_transformation_update
            Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=ICP_transformation,source2target=True)

    #target_PC=PointCloud()
    #target_PC.point=target_points_update
    #Registration.draw_registraion_init(source=source_surface,target=target_PC)
    #y_mean_3=gpisModel.prediction(target_points_update)
    #print("GT source 2 target transforma: \n", (transinit.Transform))
    #print("GT target 2 source transforma: \n", np.linalg.inv(transinit.Transform))
    #print("T_update:\n ",T_update)
    #print("update value: ",np.mean(np.abs(y_mean_3)))

def gpisOptDemo():
    vis=True
    print("=====> Prepare the Point Cloud Data")
    file_path=os.path.join(os.path.dirname(__file__),"../","data/happy.pcd")
    source_surface = PointCloud(filename=file_path)
    source_surface()
    # TODO, check the scale, und rescale, using the bounding box
    source_surface.scale(scale_=1)
    target_surface=copy.deepcopy(source_surface)
    transinit = Transformation()
    transinit.setT(trans=np.asarray([0., 0., 0.]),rot_deg=[0,90,0])
    target_surface.transform(transinit.Transform)
    if True:
        Registration.draw_registraion_init(source=source_surface,target=target_surface)

    print("=====> Set Up GPIS Opt")
    opt = GPISOpt()
    print("=====> Begin GPIS PCA init ")
    transform_target2source, source_down_fpfh, target_down_fpfh=opt.init4(source_surface,target_surface)

    print("====> Set Up GPIS model")
    gpisData=GPISData(surface_points=source_down_fpfh,num_in_out_lier=1000,has_in_lier=False)
    gpisData()
    print("gpisData.X_source: ",gpisData.X_source.shape)
    print("gpisData.maxR: ",gpisData.maxR)

    print("====> Prepare the GPIS Model")
    gpisModel = GPISModel(kernel=SKWilliamsMinusKernel(R=gpisData.maxR,alpha=0.01), random_state=0)
    gpisModel.fit(gpisData.X_source, gpisData.Y_source)
    y_mean_source=gpisModel.prediction(source_down_fpfh.point)
    print("source: ", y_mean_source)
    y_mean_1=gpisModel.prediction(target_down_fpfh.point)
    print("original target: ",y_mean_1)

    print("===>evaluate the init solution")
    init_predict=[]
    for transform_target2source_ in transform_target2source:
        target_points_init = opt.updateTarget_Point(target_down_fpfh.point, transform_target2source_)
        y_mean_init=gpisModel.prediction(target_points_init)
        init_predict.append(y_mean_init)
    init_predict=np.abs(np.asarray(init_predict)-y_mean_source)
    indx=np.argmin(init_predict)
    print("PCA init at ",indx," is chosen")
    if vis:
        Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transform_target2source[indx],source2target=False)
    visAll=True
    if visAll:
        for i in range(4):
            Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transform_target2source[i],source2target=False)

if __name__=="__main__":
    #test_se3() # checked
    #test_gpis_kernel() # checked
    #test_optimization()
    #test_gpis()
    #test_gpisOpt()
    gpisOptDemo()