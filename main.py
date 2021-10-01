# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:31:46 2021

@author: zongsing.huang
"""

import time
import functools

from GA import GA
import benchmark

M = 10
N = 10
P = 60
G = 1000
D = M*N

makespan = functools.partial(benchmark.makespan_10, M=M, N=N)
optimizer = GA(fitness=makespan,
               D=D, P=P, G=G, M=M, N=N)
st = time.time()
optimizer.opt()
ed = time.time()

time = ed - st
gbest_X = optimizer.gbest_X
gbest_F = optimizer.gbest_F
loss_curves = optimizer.loss_curve

print(gbest_X)