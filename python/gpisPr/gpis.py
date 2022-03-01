# Copyright (c) 2022 Jianjie Lin
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from sklearn.gaussian_process import GaussianProcessRegressor
from skkernel import SKWilliamsMinusKernel
import numpy as np

class GPIS(GaussianProcessRegressor):
    def __init__(self,kernel=None, *, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GPIS, self).__init__()
        pass

if __name__ == '__main__':
    X=np.random.rand(10,3)
    y=np.random.rand(10,1)
    kernel=SKWilliamsMinusKernel(3)
    gpis=GPIS(kernel=SKWilliamsMinusKernel(3),random_state=0).fit(X,y)

