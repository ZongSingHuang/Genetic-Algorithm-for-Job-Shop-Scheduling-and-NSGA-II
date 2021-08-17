# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:42:00 2021

@author: zongsing.huang
"""

# =============================================================================
# x in {0, 1}
# 最大化問題的最佳適應值為-D；最小化問題的最佳適應值為0
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fitness(X, MS, PT):
    P = X.shape[0]
    D = X.shape[1]
    N = MS.shape[0] # 工件數
    M = MS.shape[1] # 機台數
    F = np.zeros(P)
    X = X.astype(int)
    
    for i in range(P):
        cumulative_processing_counter = np.zeros(N, dtype=int) # 工件累積加工次數
        cumulative_running_time = np.zeros(N) # 機台累積運轉時間
        cumulative_processing_time = np.zeros(N) # 工件累積加工時間
        
        # X[i] = np.array([8, 6, 6, 2, 0, 4, 6, 7, 4, 7, 3, 2, 6, 2, 5, 4, 3, 1, 0, 4, 3, 8, 4, 2, 4, 3, 7, 9, 0, 9, 7, 2, 0, 9, 5, 9, 1, 3, 5, 2, 6, 1, 0, 2, 3, 8, 2, 1, 4, 9, 1, 0, 3, 7, 0, 3, 4, 6, 8, 9, 3, 5, 8, 6, 1, 7, 7, 8, 8, 9, 5, 1, 0, 7, 8, 5, 1, 6, 8, 9, 5, 6, 6, 0, 5, 9, 0, 1, 3, 4, 9, 5, 7, 8, 2, 7, 2, 5, 4, 1])
        for step, J_idx in enumerate(X[i]):
            O_idx = cumulative_processing_counter[J_idx] # 取得工件J_idx當前的加工程序別
            cost = PT.iloc[J_idx, O_idx] # 取得工件J_idx的加工時間
            M_idx = MS.iloc[J_idx, O_idx] # 取得工件J_idx的機台別
            # print('工單'+str(J_idx+1)+', 工序'+str(O_idx+1)+', 機台'+str(M_idx+1)+', 耗時'+str(cost))
            
            cumulative_processing_time[J_idx] = cumulative_processing_time[J_idx] + cost
            cumulative_running_time[M_idx] = cumulative_running_time[M_idx] + cost
            # print('所有機台的累積運轉時間 '+str(cumulative_running_time))
            # print('所有工件的累積加工時間 '+str(cumulative_processing_time))
        
            if cumulative_running_time[M_idx]<cumulative_processing_time[J_idx]:
                cumulative_running_time[M_idx]=cumulative_processing_time[J_idx]
            elif cumulative_running_time[M_idx]>cumulative_processing_time[J_idx]:
                cumulative_processing_time[J_idx]=cumulative_running_time[M_idx]
            # print('修正後 所有機台的累積運轉時間 '+str(cumulative_running_time))
            # print('修正後 所有工件的累積加工時間 '+str(cumulative_processing_time))
        
            cumulative_processing_counter[J_idx] = cumulative_processing_counter[J_idx] + 1
            # print('所有工件的累積加工次數 '+str(cumulative_processing_counter))

        makespan = cumulative_processing_time.max()
        F[i] = makespan
    
    return F

def initialization(MS, P):
    MS = MS.values.flatten()
    X = []
    
    for i in range(P):
        np.random.shuffle(MS)
        X.append(MS.copy())
    
    X = np.array(X)
    
    return X

def selection(X, F):
    P = X.shape[0]
    
    if F.min()<0:
        F = F + np.abs( F.min() )
    F_sum = np.sum(F)
    
    if F_sum==0:
        normalized_F = np.zeros(P)
    else:
        normalized_F = F/F_sum
    idx = np.argsort(normalized_F)[::-1]
    sorted_F = np.sort(normalized_F)[::-1]
    
    cumsum_F = np.cumsum(sorted_F)[::-1]
    cumsum_F = np.hstack([cumsum_F[1:], 0.0])
    
    new_idx = -1*np.zeros(P).astype(int)
    r = np.random.uniform(size=P)
    for i in range(len(r)):
        for j in range(len(cumsum_F)):
            if r[i]>cumsum_F[j]:
                new_idx[i] = idx[j]
                break
    
    p1 = X[new_idx][:int(P/2)]
    p2 = X[new_idx][int(P/2):]
    
    return p1, p2

def crossover(p1, p2, pc):
    P = p1.shape[0]
    D = p1.shape[1]
    new_p1 = np.zeros_like(p1)
    new_p2 = np.zeros_like(p2)
    c1 = np.zeros_like(p1)
    c2 = np.zeros_like(p2)
    
    for i in range(P):
        cut_point1, cut_point2 = np.sort( np.random.choice(range(1, D), size=2, replace=False) )
        new_p1[i] = np.hstack([ p1[i, :cut_point1], p2[i, cut_point1:cut_point2], p1[i, cut_point2:] ])
        new_p2[i] = np.hstack([ p2[i, :cut_point1], p1[i, cut_point1:cut_point2], p2[i, cut_point2:] ])
    
    for i in range(P):
        r1 = np.random.uniform()
        if r1<pc:
            c1[i] = new_p1[i]
        else:
            c1[i] = p1[i]
            
        r2 = np.random.uniform()
        if r2<pc:
            c2[i] = new_p2[i]
        else:
            c2[i] = p2[i]
    
    return c1, c2

def mutation(c1, pm):
    P = c1.shape[0]
    D = c1.shape[1]
    
    for i in range(P):
        basket = []
        
        for j in range(D):
            r = np.random.uniform()
            
            if r<=pm:
                basket.append(c1[i, j])
                c1[i, j] = -1
        
        if len(basket)>0:
            basket = np.array(basket)
            basket = np.roll(basket, 1)
            mask = c1[i]==-1
            c1[i, mask] = basket
        
    return c1

def elitism(X, F, new_X, new_F, er):
    M = X.shape[0]
    elitism_size = int(M*er)
    new_X2 = new_X.copy()
    new_F2 = new_F.copy()
    
    if elitism_size>0:
        idx = np.argsort(F)
        elitism_idx = idx[:elitism_size]
        new_X2[:elitism_size] = X[elitism_idx]
        new_F2[:elitism_size] = F[elitism_idx]
    
    return new_X2, new_F2

def fix_chromosome(X, N, M):
    # 每一個row中，每個工件(0~N-1)都要出現M次
    P = X.shape[0]
    D = X.shape[1]
    
    
    for i in range(P):
        # # 測試用
        # while True:
        #     X[i] = np.random.randint(low=0, high=10, size=[N*M])
        #     if len(np.bincount(X[i]))==10:
        #         break
            
        # step1. 紀錄每個工件的出現次數
        counter = np.bincount(X[i])
        # 建立一個空籃子
        basket = []
        
        # 逐一檢查每個工件的出現次數，是否等於機台數
        for j in range(N):
            if counter[j]!=M:
                # 若工件j的出現次數!=機台數M，則在籃子裡放入M個工件j
                basket.append(np.ones(M)*j)
                # 並且把X[i]裡的工件j全部清空(-1)
                mask = X[i]==j
                X[i, mask] = -1
        
        # 把籃子裡的東西打亂，然後放在X[i]中
        basket = np.array(basket).flatten()
        np.random.shuffle(basket)
        mask = X[i]==-1
        X[i, mask] = basket
        
        # 修復失敗就報警
        if np.bincount(X[i]).sum()!=D:
            print(X[i])
            print('修復失敗')
            print('='*20)
        
    return X

#%% 資料載入
MS = pd.read_excel('JSP_dataset.xlsx', sheet_name=0) - 1
PT = pd.read_excel('JSP_dataset.xlsx', sheet_name=1)

#%% 參數設定
N = MS.shape[0] # 工件數
M = MS.shape[1] # 機台數
P = 30
D = N*M
G = 2000
pc = 0.8
pm = 0.2
er = 0.1

#%% 初始化
# 若P不是偶數，則進行修正
if P%2!=0:
    P = 2 * (P//2)
    
X = initialization(MS, P)
gbest_X = np.zeros(D)
gbest_F = np.inf
loss_curve = np.zeros(G)

#%% 迭代
# 適應值計算
F = fitness(X, MS, PT)

for g in range(G):
    # 更新F
    if F.min()<gbest_F:
        best_idx = np.argmin(F)
        gbest_X = X[best_idx]
        gbest_F = F[best_idx]
    loss_curve[g] = gbest_F
    print(str(g) + '   ' +str(gbest_F))
    
    # 選擇
    p1, p2 = selection(X, F)
    
    # 交配
    c1, c2 = crossover(p1, p2, pc)
    c1 = fix_chromosome(c1, N, M)
    c2 = fix_chromosome(c2, N, M)
    
    # 突變
    c1 = mutation(c1, pm)
    c2 = mutation(c2, pm)
    
    # 更新X
    new_X = np.vstack([c1, c2])
    np.random.shuffle(new_X)
    
    # 適應值計算
    new_F = fitness(new_X, MS, PT)
    
    # 菁英
    new_X, new_F = elitism(X, F, new_X, new_F, er)
    
    X = new_X.copy()
    F = new_F.copy()
    
#%% 作畫
plt.figure()
plt.plot(loss_curve)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Fitness')