import matplotlib
matplotlib.use('Agg')  # 必须置于顶部，解决子进程导致的 RuntimeError

import numpy as np
import sys
import datetime
import os
import multiprocessing

# 导入自定义模块
from generate_HCA_mesh import HCA_Mesh_Generator
from fdm_adapter import FDMAdapter
from fdm_solver import FDMSolver
from fdm_optimizer_v8 import FDMOptimizer
from abaqus_expoter import AbaqusExporter

# ================= 日志记录器 =================
class Logger(object):
    def __init__(self, base_name="fdm_run"):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.filename = f"{base_name}_{timestamp}.log"
        self.terminal = sys.stdout
        self.log = open(self.filename, "a", encoding='utf-8')
        
        header = [
            "\n" + "╔" + "═" * 58 + "╗",
            "║" + " FDM 结构优化并行任务系统 (V8.1-Stable) ".center(51) + "║",
            "╠" + "═" * 58 + "╣",
            f"║  启动时刻: {now.strftime('%Y-%m-%d %H:%M:%S'):<41}║",
            f"║  硬件核心: {os.cpu_count():<41}║",
            "╚" + "═" * 58 + "╝\n"
        ]
        print("\n".join(header))

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        pass

sys.stdout = Logger()

def main():
    ## 并行化
    # 1. ------------------ 硬件与种群自动适配 ------------------
    cpu_count = os.cpu_count() or 1
    N_JOBS = max(1, cpu_count - 1) if cpu_count > 4 else cpu_count
    
    # 种群设为核心数的倍数
    POP_SIZE_GA2 = max(N_JOBS * 24, 120)
    POP_SIZE_GA1 = max(N_JOBS * 12, 64)
    
    MAX_LOOPS = 10       # 建议滚动 10 轮左右
    TARGET_RMSE = 1.0    # 理想目标 (mm)

    print(f"[System] 自动配置完毕: N_JOBS={N_JOBS}, GA2_POP={POP_SIZE_GA2}")

    # 2. ------------------ 模型与求解器准备 ------------------
    hca_model = HCA_Mesh_Generator(n_r=4, n_theta=36)
    hca_model.generate_mesh()
    adapter = FDMAdapter(hca_model)
    ncoord, conn, q_v, elsets, bcs = adapter.get_solver_inputs()
    surf_eids = adapter.get_surf_cable_eids()

    solver = FDMSolver(ncoord=ncoord, conn=conn, bcs=bcs, elsets=elsets)
    optimizer = FDMOptimizer(solver, only_surface=True, surf_elset=surf_eids)

    # 3. ================= Step 0: 灵活的种子初始化 =================
    # 方案 A: 从本地文件加载（推荐，适合断点续传）
    seed_path = "Optimized_V8_Stable_RMSE_13.81_TR_19.24.npy" 
    q_pre = optimizer.load_seeds(seed_path) if os.path.exists(seed_path) else None

    # 方案 B: 寻形预处理（适合全新开始）
    # print("\n[Step 0] 种子初始化...")
    # q_pre = optimizer.run_iteration(max_iter=100, rms_limit=15.0) 
    
    # 方案 C: 直接设为 None，由 GA2 内部处理初始化逻辑
    # q_pre = None 

    if q_pre is not None:
        print(f"✅ 成功获取初始种子，将作为 GA2 的演化起点。")
    else:
        print(f"⚠️ 未提供有效种子，GA2 将启动自生成/随机初始化模式。")

    # 4. ================= Step 1: GA2 全局滚动搜索 =================
    # 如果 q_pre 为 None，current_seeds 将被传入为 [None] 或 None
    # 确保 FDMOptimizer.run_GA2 内部能够处理这种情况
    current_seeds = [q_pre] if q_pre is not None else None
    global_best_q = q_pre
    global_best_rmse = 9999.0
    
    print(f"\n[Step 1] 开始 GA2 全局并行搜索...")

    for loop_idx in range(1, MAX_LOOPS + 1):
        print(f"\n" + ">>>" * 15)
        print(f" 开始第 {loop_idx} 轮演化...")
        
        dynamic_u_limit = max(40.0, 120.0 - (loop_idx-1)*20.0)
        
        res_q_selected, res_all_q, res_all_F = optimizer.run_GA2(
            q_seeds=current_seeds, 
            q_bounds=(0.05, 400.0), 
            u_limit=dynamic_u_limit, 
            t_limits=(5.0, 1500.0), 
            n_gen=250, 
            pop_size=POP_SIZE_GA2, 
            n_jobs=N_JOBS, 
            penalty=True,
            plot_pareto=False 
        )

        if res_all_F is not None and res_q_selected is not None:
            # 重新计算真实物理 RMSE
            c, t = solver.solve(res_q_selected)
            # 核心修复：计算 RMSE
            current_rmse = 1e3*np.sqrt(np.mean(np.sum((c - solver.ncoord)**2, axis=1)))
            
            print(f"--- 第 {loop_idx} 轮结果: 本轮优选 RMSE = {current_rmse:.4f} mm ---")
            
            if current_rmse < global_best_rmse:
                global_best_rmse = current_rmse
                global_best_q = res_q_selected
                print(f"🌟 检测到更优解！全局最佳更新为: {global_best_rmse:.4f} mm")
            
            current_seeds = [global_best_q] + list(res_all_q)
            
            if global_best_rmse < TARGET_RMSE:
                break
        else:
            current_seeds = [global_best_q]

    # 5. ------------------ Step 2: GA1 精修 ------------------
    print("\n[Step 2] 启动 GA1 并行精修 (基于历史最佳种子)...")
    
    # 强压 RMSE
    q_mid = optimizer.run_GA1(
        q_init=global_best_q,
        q_ratio_bounds=(0.7, 1.3),
        RMSE_weight=0.9, 
        n_gen=200,
        pop_size=POP_SIZE_GA1,
        n_jobs=N_JOBS
    )

    # 平滑张力
    final_best_q = optimizer.run_GA1(
        q_init=q_mid,
        q_ratio_bounds=(0.9, 1.1),
        RMSE_weight=0.4,
        n_gen=200,
        pop_size=POP_SIZE_GA1,
        n_jobs=N_JOBS
    )

    # 6. ------------------ Step 3: 最终成果 ------------------
    print("\n" + "═"*60)
    optimizer._print_status_report("最终优化成果报告", final_best_q)
    optimizer.save_seeds(final_best_q, "Optimized_V8_Stable")
    
    final_coords, final_tensions = solver.solve(final_best_q)
    exporter = AbaqusExporter(solver)
    exporter.write_inp("FDM_Result_Stable.inp", final_coords, final_tensions)
    print(f"任务圆满完成。")
    print("═"*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        import traceback
        traceback.print_exc()