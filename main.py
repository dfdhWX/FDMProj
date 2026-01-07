import matplotlib
matplotlib.use('Agg')  # 必须置于顶部，解决子进程导致的 GUI 冲突

import numpy as np
import sys
import datetime
import os
import multiprocessing

# 导入 pymoo 相关组件
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.parallelization.joblib import JoblibParallelization

# 导入你的自定义模块
from generate_HCA_mesh import HCA_Mesh_Generator
from fdm_adapter import FDMAdapter
from fdm_solver import FDMSolver
from fdm_optimizer_v8 import FDMOptimizer, FDMGA2Problem, FDMGA1Problem # 确保类名对应
# 导出 ABAQUS INP
from abaqus_expoter import AbaqusExporter # 确保拼写正确

# ================= 1. 日志记录器 =================
class Logger(object):
    def __init__(self, base_name="fdm_optimization"):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.filename = f"{base_name}_{timestamp}.log"
        self.terminal = sys.stdout
        self.log = open(self.filename, "a", encoding='utf-8')
        
        header = [
            "\n" + "╔" + "═" * 58 + "╗",
            "║" + " FDM 结构优化并行任务系统 (V8.5-Stable) ".center(51) + "║",
            "╠" + "═" * 58 + "╣",
            f"║  启动时刻: {now.strftime('%Y-%m-%d %H:%M:%S'):<41}║",
            f"║  计算精度目标: RMSE < 1.0 mm {' ':<23}║",
            "╚" + "═" * 58 + "╝\n"
        ]
        print("\n".join(header))

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        pass

# ================= 2. 主程序 =================
def main():
    sys.stdout = Logger()
    
    # --- 硬件与并行参数适配 ---
    cpu_count = os.cpu_count() or 1
    N_JOBS = max(1, cpu_count - 1) 
    POP_SIZE_GA2 = max(N_JOBS * 20, 128) # 确保种群足够覆盖搜索空间
    MAX_LOOPS = 10                        # GA2 滚动演化轮次
    TARGET_RMSE = 2.0                    # 必须死守的门槛 (mm)
    TARGET_TR = 20.0                    # 理想张力比目标

    print(f"[System] 并行配置: 使用 {N_JOBS} 核计算, 初始种群规模 {POP_SIZE_GA2}")

    # --- 模型与求解器初始化 ---
    # 结构参数
    D, F = 10.0, 6.0
    H = 6.206
    h = 0.5*H-D**2/(16.0*F)
    # 网格生成
    hca_model = HCA_Mesh_Generator(n_r=4, n_theta=36, D=D, F=F, H = H, h = h)
    hca_model.generate_mesh()
    hca_model.mesh_plot(show_labels=False)
    
    # ============== FDM =================
    adapter = FDMAdapter(hca_model)
    ncoord, conn, q_v, elsets, bcs = adapter.get_solver_inputs()
    surf_eids = adapter.get_surf_cable_eids()

    solver = FDMSolver(ncoord=ncoord, conn=conn, bcs=bcs, elsets=elsets)
    optimizer = FDMOptimizer(solver, only_surface=True, surf_elset=surf_eids)

    # ================= STEP 0: 初始种子生成 =================
    print("\n[Step 0] 正在通过迭代法生成初始可行种子...")
    # 这一步是为了拿到你说的 RMSE<1, TR=78 的解
    q_iter = optimizer.run_iteration(max_iter=2000, rms_limit=TARGET_RMSE)
    
    # 
    # seed_path = 'GA1V8_RMSE_1.00_TR_14.46.npy'
    # q_iter = optimizer.load_seeds(seed_path)
    
    # 物理评估初始状态
    c0, t0 = solver.solve(q_iter)
    current_best_rmse = 1e3 * np.sqrt(np.mean(np.sum((c0 - solver.ncoord)**2, axis=1)))
    current_best_tr = np.max(t0[surf_eids]) / (np.min(t0[surf_eids]) + 1e-8)
    global_best_q = q_iter

    print(f">>> 初始种子状态: RMSE = {current_best_rmse:.4f} mm, TR = {current_best_tr:.2f}")

    # ================= STEP 1: GA2 滚动演化 (TR 压降阶段) =================
    # 使用 NSGA2 在保持 RMSE < 1 的前提下压低 TR
    for loop in range(1, MAX_LOOPS + 1):
        print(f"\n" + "="*60)
        print(f" GA2 演化轮次 {loop}/{MAX_LOOPS} | 当前最优 TR: {current_best_tr:.2f}")
        print("="*60)

        # 动态调节：如果 RMSE 已经稳在 1.0 以内，提高变异 eta 以进行更精细的搜索
        dyn_eta = 20 if current_best_rmse > 0.95 else 40
        
        # 调用封装好的并行 GA2
        res_q, all_X, all_F = optimizer.run_GA2(
            q_seeds=global_best_q,
            q_bounds=(0.01, 600.0),
            pop_size=POP_SIZE_GA2,
            n_gen=200,
            rmse_limit=TARGET_RMSE, # 传入硬指标
            n_jobs=N_JOBS,
            mutation_eta=dyn_eta
        )

        # 验证本轮最优解
        c_loop, t_loop = solver.solve(res_q)
        rmse_loop = 1e3 * np.sqrt(np.mean(np.sum((c_loop - solver.ncoord)**2, axis=1)))
        tr_loop = np.max(t_loop[surf_eids]) / (np.min(t_loop[surf_eids]) + 1e-8)

        # 更新逻辑：优先保证 RMSE 达标，再看 TR 是否下降
        if rmse_loop <= TARGET_RMSE:
            if tr_loop < current_best_tr:
                print(f"✨ 成功优化! TR 从 {current_best_tr:.2f} 降至 {tr_loop:.2f} (RMSE: {rmse_loop:.4f})")
                current_best_tr = tr_loop
                current_best_rmse = rmse_loop
                global_best_q = res_q
            else:
                print(f"--- 本轮未发现更低 TR，保持当前种子 ---")
        else:
            # 如果不小心跳出了 RMSE<1 区域，取 RMSE 最小的解作为下轮寻找可行域的基准
            if rmse_loop < current_best_rmse:
                global_best_q = res_q
                current_best_rmse = rmse_loop
                print(f"📉 正在收敛 RMSE: {rmse_loop:.4f}")

        # 提前终止：如果 TR 已经非常理想
        if current_best_rmse < TARGET_RMSE and current_best_tr < TARGET_TR:
            print("✅ 已达到预设目标，提前结束演化。")
            break
    
    optimizer.save_seeds(global_best_q, "GA2V8")
    # ================= STEP 2: GA1 终极微调 (单目标压榨) =================
    print("\n[Step 2] 启动 GA1 单目标窄域精修...")
    # 只在当前最佳解的 ±8% 范围内变动 q
    final_best_q = optimizer.run_GA1(
        q_init=global_best_q,
        q_ratio_bounds=(0.92, 1.08),
        RMSE_weight=0.1,    # 给 TR 90% 的权重
        RMSE_tol=0.85,      # 只要精度好于 0.85mm，就不再为精度扣分
        n_gen=150,
        pop_size=max(N_JOBS * 12, 64),
        n_jobs=N_JOBS
    )

    # ================= STEP 3: 结果导出 =================
    print("\n" + "═"*60)
    optimizer._print_status_report("最终优化成果报告", final_best_q)
    

    filename = optimizer.save_seeds(final_best_q, f"GA1V8")
    
    
    final_coords, final_tensions = solver.solve(final_best_q)
    exporter = AbaqusExporter(solver)
    exporter.write_inp(f"Final_Optimized.inp", final_coords, final_tensions)
    
    print(f"所有任务已圆满完成，文件已保存。")
    print("═"*60)

if __name__ == "__main__":
    # 多进程并行必须在 main 保护下运行
    try:
        main()
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        import traceback
        traceback.print_exc()