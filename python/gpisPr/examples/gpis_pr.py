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
    gpisPrOptions=GPISPrOptions(gpis_alpha=0.0000,
    voxel_size=0.0001,
    num_in_out_lier=300,
    use_init4=False,
    use_batch=False,
    has_in_lier=True,
    use_pca_init=False,
    LM_factor=0.01)
    
    print("=====> Prepare the Point Cloud Data")
    file_path=os.path.join(os.path.dirname(__file__),"../../","data/bunny_1420.pcd")
    source_surface = PointCloud(filename=file_path)
    source_surface()
    source_surface.scale(scale_=1)
    target_surface=copy.deepcopy(source_surface)
    target_surface.add_Gaussian_noise(0.001)
    source_surface.add_Gaussian_noise(0.001)
    transinit = Transformation()
    transinit.setT(trans=np.asarray([0.01, 0.01, 0.0]),rot_deg=[50,0,40])
    target_surface.transform(transinit.Transform)
    vis_Init=False
    if vis_Init:
        Registration.draw_registraion_init(source=source_surface,target=target_surface)

    print("=====> Set Up GPIS Opt")
    gpisOpt = GPISOpt(voxel_size=gpisPrOptions.voxel_size,use_batch=gpisPrOptions.use_batch,LM_factor=gpisPrOptions.LM_factor)
    source_down_fpfh, target_down_fpfh=gpisOpt.preprocess_point_cloud(source_surface,target_surface)
    
    if gpisPrOptions.use_pca_init:
        print("=====> Begin GPIS PCA init ")
        if gpisPrOptions.use_init4:
            transform_target2source=gpisOpt.pca_init4(source_down_fpfh,target_down_fpfh)
        else:
            transform_target2source=gpisOpt.pca_init(source_down_fpfh,target_down_fpfh)
        print("====> Set Up GPIS model")

    print("=====>begin GPISData")
    gpisData=GPISData(surface_points=source_down_fpfh,num_in_out_lier=gpisPrOptions.num_in_out_lier,has_in_lier=gpisPrOptions.has_in_lier)
    gpisData()

    print("====> Prepare the GPIS Model")
    gpisModel = GPISModel(kernel=SKWilliamsMinusKernel(R=gpisData.maxR,alpha=gpisPrOptions.gpis_alpha), random_state=0)
    gpisModel.fit(gpisData.X_source, gpisData.Y_source)
    y_mean_source=gpisModel.prediction(source_down_fpfh.point)
    print("source: ", y_mean_source)
    y_mean_1=gpisModel.prediction(target_down_fpfh.point)
    print("original target: ",y_mean_1)

    print("===>evaluate the init solution") # affects by the num_in_out_lier
    if gpisPrOptions.use_pca_init:
        init_predict=[]
        for transform_target2source_ in transform_target2source:
            target_points_init = gpisOpt.updateTarget_Point(target_down_fpfh.point, transform_target2source_)
            y_mean_init=gpisModel.prediction(target_points_init)
            init_predict.append(y_mean_init)
        init_predict=np.abs(np.asarray(init_predict)-y_mean_source)
        indx=np.argmin(init_predict)
        vis=True
        if vis:
            Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transform_target2source[indx],source2target=False)
        visAll=False
        if visAll:
            for transform_t2s in transform_target2source:
                Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transform_t2s,source2target=False)

    print("====> start to optimization")
    gpisOpt.gpisModel=gpisModel
    gpisOpt.obj_opt_min=y_mean_source
    if gpisPrOptions.use_pca_init:
        _,T_update=gpisOpt.execute_gpis_point_registration(target_points=target_down_fpfh.point,T_last=transform_target2source[indx])
    else:
        T_init=Transformation()
        _,T_update=gpisOpt.execute_gpis_point_registration(target_points=target_down_fpfh.point,T_last=T_init)
    
    transformation_update=Transformation()
    transformation_update.Transform=T_update
    #target_points_update = opt.updateTarget_Point(target_surface.point, transformation_update)
    vis_result=False
    if vis_result:
        Registration.draw_registration_result(source=source_surface,target=target_surface,transformation=transformation_update,source2target=False)
    vis_animation=True
    if vis_animation:
        print(gpisOpt.update_transformation)
        Registration.animation_registration_results(source=source_surface,target=target_surface,Transformation_list=gpisOpt.update_gpis_Transormation)

if __name__=="__main__":
    gpisOptDemo()