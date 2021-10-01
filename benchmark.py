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
        Machine = pd.DataFrame(data=np.zeros([M]), columns=['Time'],
                                                   index=['M1', 'M2', 'M3'],)
        Job = pd.DataFrame(data=np.zeros([N]), columns=['Time'],
                                               index=['J1', 'J2', 'J3'])
        Operation = pd.DataFrame(data=np.zeros([N]), columns=['CT'],
                                                     index=['J1', 'J2', 'J3'], dtype=int)

        X[i] = np.array([1, 0, 2, 0, 0, 1, 2, 1, 2], dtype=int)
        
        for job in X[i]:
            # 1. 取得Job的Operation之工時與機台
            operation = Operation.iloc[job]
            sequence = Sequence[job, operation]
            cost = Cost[job, operation]
            
            # 2. 更新時間與次數
            Machine.iloc[sequence] += cost
            Job.iloc[job] += cost
            Operation.iloc[job] += 1
            
            # 3. 修正時間
            fixed_time = np.maximum(Machine.iloc[sequence].values, Job.iloc[job].values)
            Machine.iloc[sequence] = fixed_time
            Job.iloc[job] = fixed_time
            
            # 4. 更新甘特圖
        
        # makespan
        F[i] = np.max(Machine.values)
        
    return F

def makespan_10(X, M, N):
    Process_Time = np.array([[1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0],
                             [1.00, 3.00, 5.00, 10.0, 4.00, 2.00, 7.00, 6.00, 8.00, 9.00],
                             [2.00, 1.00, 4.00, 3.00, 9.00, 6.00, 8.00, 7.00, 10.0, 5.00],
                             [2.00, 3.00, 1.00, 5.00, 7.00, 9.00, 8.00, 4.00, 10.0, 6.00],
                             [3.00, 1.00, 2.00, 6.00, 4.00, 5.00, 9.00, 8.00, 10.0, 7.00],
                             [3.00, 2.00, 6.00, 4.00, 9.00, 10.0, 1.00, 7.00, 5.00, 8.00],
                             [2.00, 1.00, 4.00, 3.00, 7.00, 6.00, 10.0, 9.00, 8.00, 5.00],
                             [3.00, 1.00, 2.00, 6.00, 5.00, 7.00, 9.00, 10.0, 8.00, 4.00],
                             [1.00, 2.00, 4.00, 6.00, 3.00, 10.0, 7.00, 8.00, 5.00, 9.00],
                             [2.00, 1.00, 3.00, 7.00, 9.00, 10.0, 6.00, 4.00, 5.00, 8.00]])
    
    return 0