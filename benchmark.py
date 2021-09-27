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
    Process_Time = pd.DataFrame(data=Process_Time, columns=['Operation 1', 'Operation 2', 'Operation 3'], index=['Job 1', 'Job 2', 'Job 3'], dtype=int)

    P = X.shape[0]
    
    for i in range(P):
        CWH_machine = pd.DataFrame(data=np.zeros([N]), columns=['time'], index=['machine 1', 'machine 2', 'machine 3'], dtype=int) # 累計工時 Cumulative working hours
        CWH_job = pd.DataFrame(data=np.zeros([N]), columns=['time'], index=['job 1', 'job 2', 'job 3'], dtype=int) # 累計工時 Cumulative working hours
        PH = pd.DataFrame(data=np.zeros([N, M])-1, columns=['Operation 1', 'Operation 2', 'Operation 3'], index=['Job 1', 'Job 2', 'Job 3'], dtype=int) # 加工履歷 Processing history
        O = pd.DataFrame(np.zeros([N]), index=['job 1', 'job 2', 'job 3'], dtype=int) # 加工次數
        print('初始化 累計工時、加工履歷')
        print('當前候選解為 {X}'.format(X=X[i]))
        
        for job in X[i]:
            CWH_machine = CWH_machine.sort_values(by=['time'])
            
            for idx, machine in enumerate(CWH_machine.index):
                if idx not in PH.iloc[job, :]:
                    CWH_machine.iloc[idx] += Process_Time.iloc[job, O.iloc[job]].values
                    PH.iloc[job, O.iloc[job]] = int(machine[-1])-1
                    O.iloc[job] += 1
                    break
    
        PH = pd.DataFrame(data=PH.astype(int)+1, columns=['Operation 1', 'Operation 2', 'Operation 3'], index=['Job 1', 'Job 2', 'Job 3'])
        
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