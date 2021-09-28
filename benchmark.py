# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:43:29 2021

@author: zongsing.huang
"""

import numpy as np
import pandas as pd

def makespan_3(X, M, N):
    X = X.astype(int)
    Process_Time = np.array([[8.00, 10.0, 5.00],
                             [15.0, 9.00, 12.0],
                             [9.00, 7.00, 19.0]])

    P = X.shape[0]
    
    for i in range(P):
        PH_remove = np.tile(np.arange(M, dtype=int), N).reshape(N, M).tolist() # 加工履歷 Processing history
        PH = pd.DataFrame(data=np.zeros([N, M])-1, columns=['operation 1', 'operation 2', 'operation 3'], index=['job 1', 'job 2', 'job 3'], dtype=int) # 加工履歷 Processing history
        CWH_machine = np.zeros([M], dtype=int) # 累計工時 Cumulative working hours
        CWH_job = pd.DataFrame(data=np.zeros([N]), columns=['time'], index=['job 1', 'job 2', 'job 3'], dtype=int) # 累計工時 Cumulative working hours
        
        O = np.zeros([N], dtype=int) # 加工次數
        print('初始化 累計工時、加工履歷')
        print('當前候選解為 {X}'.format(X=X[i]))
        X[i] = np.array([1, 0, 2, 0, 0, 1, 2, 1, 2], dtype=int)
        
        for job in X[i]:
            pre = CWH_machine + Process_Time[job, O[job]] # 預加
            sorted_idx = np.argsort(pre) # 依makespan由小到大做排序
            
            for machine in sorted_idx:
                if machine in PH_remove[job]:
                    PH_remove[job].remove(machine)
                    PH.iloc[job, O[job]] = machine
                    CWH_machine[machine] += Process_Time[job, O[job]]
                    O[job] += 1
                    break
        
        print(123)
        
        
    return 0

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