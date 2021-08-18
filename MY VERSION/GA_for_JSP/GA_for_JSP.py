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
plt.rcParams['font.family'] = "Times New Roman"

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
        counter = [sum(X[i]==val) for val in range(N)]
        counter = np.array(counter)
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

def gantt(gbest_X, MS, PT):
    P = X.shape[0]
    D = X.shape[1]
    N = MS.shape[0] # 工件數
    M = MS.shape[1] # 機台數
    F = np.zeros(P)
    gbest_X = gbest_X.astype(int)
    
    Machine = [[] for i in range(M)]
    Job = [[] for i in range(N)]

    cumulative_processing_counter = np.zeros(N, dtype=int) # 工件累積加工次數
    cumulative_running_time = np.zeros(N) # 機台累積運轉時間
    cumulative_processing_time = np.zeros(N) # 工件累積加工時間
    
    for step, J_idx in enumerate(gbest_X):
        O_idx = cumulative_processing_counter[J_idx] # 取得工件J_idx當前的加工程序別
        cost = PT.iloc[J_idx, O_idx] # 取得工件J_idx的加工時間
        M_idx = MS.iloc[J_idx, O_idx] # 取得工件J_idx的機台別
        
        cumulative_processing_time[J_idx] = cumulative_processing_time[J_idx] + cost
        cumulative_running_time[M_idx] = cumulative_running_time[M_idx] + cost
    
        if cumulative_running_time[M_idx]<cumulative_processing_time[J_idx]:
            cumulative_running_time[M_idx]=cumulative_processing_time[J_idx]
        elif cumulative_running_time[M_idx]>cumulative_processing_time[J_idx]:
            cumulative_processing_time[J_idx]=cumulative_running_time[M_idx]
            
        present = cumulative_processing_time[J_idx] - cost
        increment = cost
        
        Machine[M_idx].append((present, increment))
        Job[M_idx].append('Job ' +str(J_idx+1))
        print(Machine[0], Machine[1], Machine[2])
    
        cumulative_processing_counter[J_idx] = cumulative_processing_counter[J_idx] + 1
        # print('所有工件的累積加工次數 '+str(cumulative_processing_counter))
    
    plt.figure(dpi=100, facecolor='white')
    plt.title("Gantt Chart", pad=10, fontsize=16, fontweight='bold') # 標題
    # '#BC3C28', '#0972B5', '#E28726', '#21854D'
    plt.broken_barh(Machine[0], (30, 5), facecolors='#BC3C28', edgecolor='black', label='Machine 1', zorder=2)
    plt.broken_barh(Machine[1], (20, 5), facecolors='#0972B5', edgecolor='black', label='Machine 2', zorder=2)
    plt.broken_barh(Machine[2], (10, 5), facecolors='#E28726', edgecolor='black', label='Machine 3', zorder=2)
    for txt_set, loc_set, y_loc in zip(Job, Machine, [35-2.5, 25-2.5, 15-2.5]):
        for txt, loc in zip(txt_set, loc_set):
            plt.text(loc[0]+1, y_loc-.5, txt, fontsize='medium', alpha=1)
            # print(1)
    plt.legend(frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -.3)) # 每一row顯示4個圖例
    plt.xlabel('Time', fontsize=15) # X軸標題
    plt.ylabel('Machine', fontsize=15) # Y軸標題
    plt.yticks([35-2.5, 25-2.5, 15-2.5], ['M1', 'M2', 'M3'])
    plt.grid(linestyle="-", linewidth=.5, color="gray", alpha=.6) # 網格
    plt.tight_layout() # 自動校正

#%% 資料載入
MS = pd.read_excel('JSP_3x3.xlsx', sheet_name=0) - 1
PT = pd.read_excel('JSP_3x3.xlsx', sheet_name=1)
# MS = pd.read_excel('JSP_10x10.xlsx', sheet_name=0) - 1
# PT = pd.read_excel('JSP_10x10.xlsx', sheet_name=1)

#%% 參數設定
N = MS.shape[0] # 工件數
M = MS.shape[1] # 機台數
P = 30
D = N*M
G = 3
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
plt.tight_layout()

gantt(gbest_X, MS, PT)