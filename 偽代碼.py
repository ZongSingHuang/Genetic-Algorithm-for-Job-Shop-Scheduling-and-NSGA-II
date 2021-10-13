for job in V3[i]:
    # 初始化
    need_to_fixed = 0
    
    # 取得加工次數operation、機台編號sequence、加工耗時cost
    operation = Operation[job] # 取得工件job當前的加工次數
    sequence = Sequence[job, operation] # 取得工件job當前的機台編號
    cost = Cost[job, operation] # 取得工件job當前的加工耗時
    
    # 若機台編號sequence有閒置時間Idle
    if Idle_len[sequence]:
    
    # 若機台編號sequence沒有閒置時間Idle
    else:
        Machine[sequence] += cost
        Job[job] += cost
    
    # 更新加工次數
    Operation[job] += 1
    
    # 新增機台編號sequence的閒置時間Idle
    Idle_new = np.maximum(0, Job[job]-Machine[sequence]) # 只有在Job[job]>Machine[sequence]才會有新的閒置時間Idle
    if Idle_new>0:
        Idle_len[sequence].append(Idle)
        Idle_st[sequence].append(Machine[sequence]-cost)
        Idle_ed[sequence].append(Job[job]-cost)
    
    # 修正機台時間Machine及工件時間Job
    if need_to_fixed==True:
        fixed_time = maximum(Machine[sequence], Job[job])
        Machine[sequence] = fixed_time
        Job[job] = fixed_time
    
    