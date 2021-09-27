# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:31:46 2021

@author: zongsing.huang
"""

import time
import functools

import numpy as np

from GA import GA
import benchmark

M = 3
N = 3
P = 30
G = 500
D = M*N

makespan = functools.partial(benchmark.makespan_3, M=M, N=N)
optimizer = GA(fitness=makespan,
               D=D, P=P, G=G, M=M, N=N)
st = time.time()
optimizer.opt()
ed = time.time()

time = ed - st
gbest_X = optimizer.gbest_X
gbest_F = optimizer.gbest_F
loss_curves = optimizer.loss_curve