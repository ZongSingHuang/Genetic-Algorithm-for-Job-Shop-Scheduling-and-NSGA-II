# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:43:29 2021

@author: zongsing.huang
"""

import numpy as np
import pandas as pd

def makespan_3(X, M, N):
    Process_Time = np.array([[8.00, 10.0, 5.00],
                             [15.0, 9.00, 12.0],
                             [9.00, 7.00, 19.0]])

    P = X.shape[0]
    
    for i in range(P):
        CWH = np.zeros([N]) # 累計工時 Cumulative working hours
        PH = np.zeros([N, M]) - 1 # 加工履歷 Processing history
        O = np.zeros([N]) # 加工次數
        print('初始化 累計工時、加工履歷')
        print('當前候選解為 {X}'.format(X=X[i]))
        
        for job in X[i]:
            sorted_CWH = CWH.argsort()
            print('目前各機台的累計工時，由小到大依序為')
            print('編號 {idx}'.format(idx=sorted_CWH))
            print('工時 {hr}'.format(hr=np.sort(CWH)))
            
            for machine in sorted_CWH:
                print('判定 工件 {job} 是否曾在 機台 {machine} 進行加工'.format(job=job, machine=machine))
                print('機台 {machine} 當前的履歷為 {PH}'.format(machine=machine, PH=PH[machine]))
                if machine not in PH[int(job)]:
                    CWH[int(job)] += Process_Time[int(job), int(O[int(job)])]
                    PH[int(job), int(O[int(job)])] = machine
                    O[int(job)] += 1
                    print('當前的加工履歷為 \n{PH}'.format(PH=PH))
                    print('當前的累計工時為 {CWH}'.format(CWH=CWH))
                    print('當前的加工次數為 {O}'.format(O=O))
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