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
from gpis import GPIS
from skkernel import SKWilliamsMinusKernel
from optimization import Optimization

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
if __name__=="__main__":
    #test_se3() # checked
    #test_gpis_kernel() # checked
    test_optimization()