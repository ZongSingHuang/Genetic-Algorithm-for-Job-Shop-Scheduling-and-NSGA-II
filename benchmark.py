# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:43:29 2021

@author: zongsing.huang
"""

import numpy as np
import pandas as pd

def makespan_3(X, M, N):
    X = X.astype(int)
    P = X.shape[0]
    F = np.zeros([P])
    
    Cost = np.array([[8.00, 10.0, 5.00],
                     [15.0, 9.00, 12.0],
                     [9.00, 7.00, 19.0]])
    Sequence = np.array([[2, 3, 1],
                         [1, 3, 2],
                         [3, 2, 1]], dtype=int) - 1

    
    for i in range(P):
        Machine = np.zeros([M])
        Job = np.zeros([N])
        Operation = np.zeros([N], dtype=int)
        
        for job in X[i]:
            # 1. 取得Job的Operation之工時與機台
            operation = Operation[job]
            sequence = Sequence[job, operation]
            cost = Cost[job, operation]
            
            # 2. 更新時間與次數
            Machine[sequence] += cost
            Job[job] += cost
            Operation[job] += 1
            
            # 3. 修正時間
            fixed_time = np.maximum(Machine[sequence], Job[job])
            Machine[sequence] = fixed_time
            Job[job] = fixed_time
            
            # 4. 更新甘特圖
        
        # makespan
        F[i] = Machine.max()
        
    return F

def makespan_10(X, M, N):
    X = X.astype(int)
    P = X.shape[0]
    F = np.zeros([P])
    
    Cost = np.array([[29.0, 78.0, 9.00, 36.0, 49.0, 11.0, 62.0, 56.0, 44.0, 21.0],
                     [43.0, 90.0, 75.0, 11.0, 69.0, 28.0, 46.0, 46.0, 72.0, 30.0],
                     [91.0, 85.0, 39.0, 74.0, 90.0, 10.0, 12.0, 89.0, 45.0, 33.0],
                     [81.0, 95.0, 71.0, 99.0, 9.00, 52.0, 85.0, 98.0, 22.0, 43.0],
                     [14.0, 6.00, 22.0, 61.0, 26.0, 69.0, 21.0, 49.0, 72.0, 53.0],
                     [84.0, 2.00, 52.0, 95.0, 48.0, 72.0, 47.0, 65.0, 6.00, 25.0],
                     [46.0, 37.0, 61.0, 13.0, 32.0, 21.0, 32.0, 89.0, 30.0, 55.0],
                     [31.0, 86.0, 46.0, 74.0, 32.0, 88.0, 19.0, 48.0, 36.0, 79.0],
                     [76.0, 69.0, 76.0, 51.0, 85.0, 11.0, 40.0, 89.0, 26.0, 74.0],
                     [85.0, 13.0, 61.0, 7.00, 64.0, 76.0, 47.0, 52.0, 90.0, 45.0]])
    Sequence = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         [1, 3, 5, 10, 4, 2, 7, 6, 8, 9],
                         [2, 1, 4, 3, 9, 6, 8, 7, 10, 5],
                         [2, 3, 1, 5, 7, 9, 8, 4, 10, 6],
                         [3, 1, 2, 6, 4, 5, 9, 8, 10, 7],
                         [3, 2, 6, 4, 9, 10, 1, 7, 5, 8],
                         [2, 1, 4, 3, 7, 6, 10, 9, 8, 5],
                         [3, 1, 2, 6, 5, 7, 9, 10, 8, 4],
                         [1, 2, 4, 6, 3, 10, 7, 8, 5, 9],
                         [2, 1, 3, 7, 9, 10, 6, 4, 5, 8]], dtype=int) - 1

    
    for i in range(P):
        Machine = np.zeros([M])
        Job = np.zeros([N])
        Operation = np.zeros([N], dtype=int)
        
        for job in X[i]:
            # 1. 取得Job的Operation之工時與機台
            operation = Operation[job]
            sequence = Sequence[job, operation]
            cost = Cost[job, operation]
            
            # 2. 更新時間與次數
            Machine[sequence] += cost
            Job[job] += cost
            Operation[job] += 1
            
            # 3. 修正時間
            fixed_time = np.maximum(Machine[sequence], Job[job])
            Machine[sequence] = fixed_time
            Job[job] = fixed_time
            
            # 4. 更新甘特圖
        
        # makespan
        F[i] = Machine.max()
        
    return F