import numpy as np

def tension_ratio(tensions):
    """计算张力比（最大张力/最小张力）"""
    t_min = np.min(tensions)
    t_min = max(t_min, 1e-8)  # 避免除零
    return np.max(tensions) / t_min

def micro_adjust_q(optimizer, solver, q_seed, surf_eids, 
                   rmse_limit=1.0, max_iter=500, alpha=0.02):
    """
    局部微调力密度 q，强约束 RMSE ≤ rmse_limit，降低张力比

    参数:
        optimizer: FDMOptimizer 实例，包含 elsets 信息
        solver: FDMSolver 实例
        q_seed: 初始力密度 numpy array (长度 = num_group 或 num_elemt)
        surf_eids: 表面索元素编号列表
        rmse_limit: 最大允许 RMSE (mm)
        max_iter: 最大迭代次数
        alpha: 步长
    """
    q = q_seed.copy()
    
    # 判断 q 是按组还是按元素
    is_group_q = len(q) == len(solver.elsets)
    
    for it in range(max_iter):
        coords, tensions = solver.solve(q)
        rmse = 1e3*np.sqrt(np.mean(np.sum((coords - solver.ncoord)**2, axis=1)))
        tr = tension_ratio(tensions[surf_eids])
        
        if rmse > rmse_limit:
            print(f"[Iter {it}] RMSE 超限 {rmse:.4f} mm，停止微调")
            break
        
        # 找出表面索张力最大和最小元素
        idx_max_eid = surf_eids[np.argmax(tensions[surf_eids])]
        idx_min_eid = surf_eids[np.argmin(tensions[surf_eids])]

        if is_group_q:
            # 找最大/最小元素所属组
            group_max = None
            group_min = None
            for i, idxs in enumerate(solver.elsets):
                if idx_max_eid in idxs:
                    group_max = i
                if idx_min_eid in idxs:
                    group_min = i
            if group_max is None or group_min is None:
                raise ValueError("无法映射表面索到力密度组，请检查 elsets")

            # 调整 q_group
            q[group_max] *= (1 - alpha)
            q[group_min] *= (1 + alpha)
        else:
            # 按元素调整
            q[idx_max_eid] *= (1 - alpha)
            q[idx_min_eid] *= (1 + alpha)
        
        # 可选：打印进度
        if it % 10 == 0:
            print(f"[Iter {it}] RMSE={rmse:.4f} mm, TR={tr:.4f}")
    
    # 最终解
    final_coords, final_tensions = solver.solve(q)
    final_rmse = 1e3*np.sqrt(np.mean(np.sum((final_coords - solver.ncoord)**2, axis=1)))
    final_tr = tension_ratio(final_tensions[surf_eids])
    print(f"\n[Micro-adjust Done] RMSE={final_rmse:.4f} mm, TR={final_tr:.4f}")
    
    return q, final_coords, final_tensions
