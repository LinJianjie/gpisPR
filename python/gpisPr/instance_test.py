# Copyright (c) 2022 Jianjie Lin
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from  liegroups import SE3
from utils import *
import transforms3d as t3d
from gpis import GPIS
from skkernel import SKWilliamsMinusKernel

def test_se3():
    X=Transformation()
    X.trans=np.asarray([2,3,4])
    X.rotation=t3d.euler.euler2mat(DEG2RAD(20.0), DEG2RAD(40.0), DEG2RAD(30.0), 'sxyz')
    print(X.Transform)
    p2 = np.asarray([1, 2, 3, 1])
    p22=np.matmul(X.Transform,p2)
    print("p22:",p22)
    p33=np.asarray([[1, 2, 3, 1],[2,3,4,1],[4,5,5,1]])
    print(p33.shape)
    odot2 = SE3.odot(p33)
    print(odot2.shape)
def test_gpis_kernel():
    X_source = np.random.rand(100, 3)
    y_source = np.random.rand(100, 1)
    X_target=np.random.rand(10, 3)
    kernel = SKWilliamsMinusKernel(3)
    gpis = GPIS(kernel=SKWilliamsMinusKernel(3), random_state=0)
    K_gradient=kernel.gradient(X_source,X_target)
    print(K_gradient.shape)

if __name__=="__main__":
    #test_se3() # checked
    test_gpis_kernel()