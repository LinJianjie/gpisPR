import copy
import sys
import os
import transforms3d as t3d
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from gpisPr.liegroups import SE3
from gpisPr.utils import *

from gpisPr.gpis import GPISModel,GPISData,GPISOpt,GPISPrOptions
from gpisPr.skkernel import SKWilliamsPlusKernel,SKWilliamsMinusKernel,SKRBF,SKMatern
from gpisPr.pointCloud import PointCloud
from gpisPr.registration import Registration

def gpisOptDemo():
    gpisPrOptions=GPISPrOptions(gpis_alpha=0,voxel_size=0.0001,num_in_out_lier=50)
    vis=True
    print("=====> Prepare the Point Cloud Data")
    file_path=os.path.join(os.path.dirname(__file__),"../../","data/bunny_1420.pcd")
    source_surface = PointCloud(filename=file_path)
    source_surface()
    source_surface.scale(scale_=1)
    target_surface=copy.deepcopy(source_surface)
    transinit = Transformation()
    transinit.setT(trans=np.asarray([0.02, 0., 0.]),rot_deg=[90,90,0])
    target_surface.transform(transinit.Transform)
    if True:
        Registration.draw_registraion_init(source=source_surface,target=target_surface)

    print("=====> Set Up GPIS Opt")
    opt = GPISOpt(voxel_size=gpisPrOptions.voxel_size)
    print("=====> Begin GPIS PCA init ")
    transform_target2source, source_down_fpfh, target_down_fpfh=opt.init(source_surface,target_surface)

    print("====> Set Up GPIS model")
    gpisData=GPISData(surface_points=source_down_fpfh,num_in_out_lier=gpisPrOptions.num_in_out_lier,has_in_lier=False)
    gpisData()

    print("====> Prepare the GPIS Model")
    gpisModel = GPISModel(kernel=SKWilliamsMinusKernel(R=gpisData.maxR,alpha=gpisPrOptions.gpis_alpha), random_state=0)
    gpisModel.fit(gpisData.X_source, gpisData.Y_source)
    y_mean_source=gpisModel.prediction(source_down_fpfh.point)
    print("source: ", y_mean_source)
    y_mean_1=gpisModel.prediction(target_down_fpfh.point)
    print("original target: ",y_mean_1)

    print("===>evaluate the init solution") # affects by the num_in_out_lier
    init_predict=[]
    for transform_target2source_ in transform_target2source:
        target_points_init = opt.updateTarget_Point(target_down_fpfh.point, transform_target2source_)
        y_mean_init=gpisModel.prediction(target_points_init)
        init_predict.append(y_mean_init)
    init_predict=np.abs(np.asarray(init_predict)-y_mean_source)
    indx=np.argmin(init_predict)
    if vis:
        Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transform_target2source[indx],source2target=False)
    visAll=False
    if visAll:
        for transform_t2s in transform_target2source:
            Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transform_t2s,source2target=False)

    print("====> start to optimization")
    opt.gpisModel=gpisModel
    opt.obj_opt_min=y_mean_source
    #target_points_update, T_update=opt.step(target_points=target_down_fpfh.point,T_last=Transformation())
    _,T_update=opt.execute_gpis_point_registration(target_points=target_down_fpfh.point,T_last=transform_target2source[indx])
    transformation_update=Transformation()
    transformation_update.Transform=T_update
    #target_points_update = opt.updateTarget_Point(target_surface.point, transformation_update)
    if True:
        Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transformation_update,source2target=False)

if __name__=="__main__":
    gpisOptDemo()