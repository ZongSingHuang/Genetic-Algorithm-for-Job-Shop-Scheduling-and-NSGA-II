# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:52:03 2021

@author: zongsing.huang
"""

import numpy as np
import matplotlib.pyplot as plt
import time

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
        
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
        
    def opt(self):
        # 初始化
        self.X = self.initialization()
        
        # 迭代
        for g in range(self.G):
            # 適應值計算
            st = time.time()
            F = self.fitness(self.X)
            ed = time.time()
            print('適應值計算:' + str(ed-st))
            
            # 更新最佳解
            st = time.time()
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            ed = time.time()
            print('更新最佳解:' + str(ed-st))
            
            # 收斂曲線
            st = time.time()
            self.loss_curve[g] = self.gbest_F
            ed = time.time()
            print('收斂曲線:' + str(ed-st))
            
            # 更新
            st = time.time()
            p1, p2 = self.selection(F)
            ed = time.time()
            print('選擇:' + str(ed-st))
            
            st = time.time()
            o1, o2 = self.crossover(p1, p2)
            ed = time.time()
            print('交配:' + str(ed-st))
            
            st = time.time()
            o1 = self.fix_chromosome(o1)
            o2 = self.fix_chromosome(o2)
            ed = time.time()
            print('修復:' + str(ed-st))
            
            st = time.time()
            o1 = self.mutation(o1)
            o2 = self.mutation(o2)
            ed = time.time()
            print('突變:' + str(ed-st))
            
            self.X = np.vstack([o1, o2])
            
            new_X, new_F = self.elitism(X, F, new_X, new_F, er)
            
            new_X, new_F = self.immigrant(new_X, new_F, ir, M, pt)
            
            print('gbest_F:' + str(self.gbest_F))
            print('-'*20)
    
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
    
    def elitism(self, X, F, new_X, new_F, er):
        P = X.shape[0]
        elitism_size = int(P*er)
        
        if elitism_size>0:
            idx = np.argsort(F)
            elitism_idx = idx[:elitism_size]
            elite_X = X[elitism_idx]
            elite_F = F[elitism_idx]
            
            for i in range(elitism_size):
                
                if elite_F[i]<new_F.mean():
                    idx = np.argsort(new_F)
                    worst_idx = idx[-1]
                    new_X[worst_idx] = elite_X[i]
                    new_F[worst_idx] = elite_F[i]
        
        return new_X, new_F
    
    def immigrant(self, new_X, new_F, ir, M, pt):
        P = new_X.shape[0]
        D = new_X.shape[1]
        N = P
        immigrant_size = int(P*er)
        
        if immigrant_size>0:
            
            for i in range(immigrant_size):
                immigrant_X = np.random.choice(M, size=[1, D])
                immigrant_F = fitness(immigrant_X, pt, N, M)
                
                if immigrant_F<new_F.mean():
                    idx = np.argsort(new_F)
                    worst_idx = idx[-1]
                    new_X[worst_idx] = immigrant_X
                    new_F[worst_idx] = immigrant_F
        
        return new_X, new_F