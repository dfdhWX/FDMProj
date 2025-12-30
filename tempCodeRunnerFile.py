#====== GA2 获取初始值=======
    q_init, all_X, all_F = optimizer.run_GA2(n_gen=200,pop_size=100)

    # 调用优化
    best_q = optimizer.run_GA1(
        q_init=q_init, 
        q_ratio_bounds=(0.7, 1.3), # 允许 30% 的微调
        weight=0.65,               # 65% 关注几何，35% 关注张力均匀
        u_limit=50.0,              # 依然保持 50mm 的硬性位移红线
        t_limits=(5.0, 100.0)      # 限制张力在安全区间，防止出现极端高/低值
    )
