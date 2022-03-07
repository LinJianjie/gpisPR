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

"""Special Euclidean and Special Orthogonal Lie groups."""

from .numpy import SO2 as SO2
from .numpy import SE2 as SE2
from .numpy import SO3 as SO3
from .numpy import SE3 as SE3


try:
    from . import numpy
    from . import torch
except:
    pass

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"
