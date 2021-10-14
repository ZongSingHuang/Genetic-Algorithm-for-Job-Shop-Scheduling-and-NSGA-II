Machine = np.zeros([M])
Job = np.zeros([N])
Operation = np.zeros([N], dtype=int)
Idle_len = [[] for i in range(M)]
Idle_st = [[] for i in range(M)]
Idle_ed = [[] for i in range(M)]

for job in V3[i]:
    # 初始化
    need_to_fixed = True
    
    # 取得加工次數operation、機台編號sequence、加工耗時cost
    operation = Operation[job] # 取得工件job當前的加工次數
    sequence = Sequence[job, operation] # 取得工件job當前的機台編號
    cost = Cost[job, operation] # 取得工件job當前的加工耗時
    
    if Idle_len[sequence]: # 若機台編號sequence有閒置時間Idle
		if any(Idle_len[sequence]>=cost): # 若機台編號sequence的閒置時間Idle_len 大於等於 當前的加工耗時cost 為存在
			condition1 = Idle_len[sequence]>=cost
			condition2 = Idle_ed[sequence]>=Job[job]+cost
			if operation==0: # 若當前的加工次數為0，則僅需考慮條件1
				mask = condition1
			else: # 若當前的加工次數不為0，則需考慮條件1及條件2
				mask = condition1*condition2
				
			if any(mask)==True: # 若遮罩存在True，代表至少有一個Idle可以被使用
				idx = np.where(mask==True)[0]
				if len(idx)>1: # 若有多個Idle可以被使用，則默認選擇第1個(接近時刻0)
					idx = idx[0]
				
				# 修正時間
				
				Job[job] += cost # 僅須更新Job
				need_to_fixed = False # 不需要修正機台時間Machine及工件時間Job
				
			else: # 若遮罩不存在True，代表沒有Idle可以被使用
				Machine[sequence] += cost
				Job[job] += cost
		else: # 若機台編號sequence的閒置時間Idle_len 大於等於 當前的加工耗時cost 為不存在
			Machine[sequence] += cost
			Job[job] += cost
    else: # 若機台編號sequence沒有閒置時間Idle
        Machine[sequence] += cost
        Job[job] += cost
    
    # 更新加工次數
    Operation[job] += 1
    
    # 新增機台編號sequence的閒置時間Idle
    if Job[job]>Machine[sequence]: # 只有在Job[job]>Machine[sequence]才會有新的閒置時間Idle
		Idle_new = Job[job] - Machine[sequence]
        Idle_len[sequence].append(Idle)
        Idle_st[sequence].append(Machine[sequence]-cost)
        Idle_ed[sequence].append(Job[job]-cost)
    
    # 修正機台時間Machine及工件時間Job
    if need_to_fixed==True:
        fixed_time = maximum(Machine[sequence], Job[job])
        Machine[sequence] = fixed_time
        Job[job] = fixed_time
    
    