# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:52:03 2021

@author: zongsing.huang
"""

import numpy as np
import matplotlib.pyplot as plt

class GA():
    def __init__(self, fitness, D=30, P=20, G=500, M=5, N=5,
                 pc=0.8, pm=0.1, er=0.1, ir=0.1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.M = M
        self.N = N
        self.pc = pc
        self.pm = pm
        self.er = er
        self.ir = ir
        
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
    def opt(self):
        # 初始化
        self.X = self.initialization()

        # 適應值計算
        F = self.fitness(self.X)
        
        # 迭代
        for g in range(self.G):

            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 選擇
            p1, p2 = self.selection(F)
            
            # 交配
            o1, o2 = self.crossover(p1, p2)
            
            # 修復
            o1 = self.fix_chromosome(o1)
            o2 = self.fix_chromosome(o2)
            
            # 突變
            o1 = self.mutation(o1)
            o2 = self.mutation(o2)
            
            # 合併
            X_new = np.vstack([o1, o2])
            
            # 適應值計算
            F_new = self.fitness(X_new)
            
            # 移民
            X_new, F_new = self.immigrant(X_new, F_new)
            
            # 菁英
            X_new, F_new = self.elitism(self.X, F, X_new, F_new)
        
            self.X = X_new.copy()
            F = F_new.copy()
    
    def initialization(self):
        X = np.zeros([self.P, self.D])
        job_set = np.arange(self.N)
        job_set = np.repeat(job_set, self.M)
        
        for i in range(self.P):
            np.random.shuffle(job_set)
            X[i] = job_set.copy()
        
        return X
    
    def selection(self, F):
        if F.min()<0:
            F = F + np.abs( F.min() )
        F_sum = np.sum(F)
        
        if F_sum==0:
            normalized_F = np.zeros([self.P])
        else:
            normalized_F = F/F_sum
        idx = np.argsort(normalized_F)[::-1]
        sorted_F = np.sort(normalized_F)[::-1]
        
        cumsum_F = np.cumsum(sorted_F)[::-1]
        cumsum_F = np.hstack([cumsum_F[1:], 0.0])
        
        new_idx = -1*np.zeros([self.P]).astype(int)
        r = np.random.uniform(size=[self.P])
        for i in range(len(r)):
            for j in range(len(cumsum_F)):
                if r[i]>cumsum_F[j]:
                    new_idx[i] = idx[j]
                    break
        
        p1 = self.X[new_idx][:int(self.P/2)]
        p2 = self.X[new_idx][int(self.P/2):]
        
        return p1, p2
    
    def crossover(self, p1, p2):
        o1 = p1.copy()
        o2 = p2.copy()
        P = p1.shape[0]
        
        for i in range(P):
            r = np.random.uniform()
            if r<=self.pc:
                idx = np.random.choice(self.D, size=2, replace=False)
                idx1, idx2 = np.sort(idx)
                o1[i, idx1:idx2] = p2[i, idx1:idx2].copy()
                o2[i, idx1:idx2] = p1[i, idx1:idx2].copy()
        
        return o1, o2
    
    def mutation(self, o):
        new_o = o.copy()
        P = o.shape[0]
        ct = int(self.D*self.pm)
        
        for i in range(P):
            idx = np.random.choice(self.D, size=ct, replace=False)
            
            temp = o[i, idx].copy()
            temp = np.roll(temp, 1)
            new_o[i, idx] = temp.copy()
        
        return new_o
    
    def fix_chromosome(self, o):
        P = o.shape[0]
        
        for i in range(P):
            basket = []
            
            for j in range(self.N):
                mask = o[i]==j
                
                if sum(mask)>self.M:
                    idx = np.where(o[i]==j)[0]
                    idx = idx[self.M:]
                    o[i, idx] = -1
                
                if sum(mask)<self.M:
                    temp = np.ones([self.M-sum(mask)])*j
                    basket = basket + temp.tolist()
            
            basket = np.array(basket)
            np.random.shuffle(basket)
            idx = np.where(o[i]==-1)[0]
            o[i, idx] = basket.copy()
        
        return o
    
    def elitism(self, X, F, X_new, F_new):
        P = X.shape[0]
        elitism_size = int(P*self.er)
        
        if elitism_size>0:
            idx = np.argsort(F)
            elitism_idx = idx[:elitism_size]
            elite_X = X[elitism_idx]
            elite_F = F[elitism_idx]
            
            for i in range(elitism_size):
                
                if elite_F[i]<F_new.mean():
                    idx = np.argsort(F_new)
                    worst_idx = idx[-1]
                    X_new[worst_idx] = elite_X[i]
                    F_new[worst_idx] = elite_F[i]
        
        return X_new, F_new
    
    def immigrant(self, X_new, F_new):
        P = X_new.shape[0]
        immigrant_size = int(P*self.ir)
        
        if immigrant_size>0:
            job_set = np.arange(self.N)
            immigrant_X = np.repeat(job_set, self.M)
            
            for i in range(immigrant_size):
                immigrant_F = self.fitness(immigrant_X)
                
                if immigrant_F<F_new.mean():
                    idx = np.argsort(F_new)
                    worst_idx = idx[-1]
                    X_new[worst_idx] = immigrant_X.copy()
                    F_new[worst_idx] = immigrant_F
        
        return X_new, F_new