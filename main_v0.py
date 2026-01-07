import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, os, datetime
from generate_HCA_mesh import HCA_Mesh_Generator
from fdm_adapter import FDMAdapter
from fdm_solver import FDMSolver
from optimizer_v1 import FDMOptimizer
from abaqus_expoter import AbaqusExporter

class Logger(object):
    def __init__(self, base_name="fdm_opt"):
        # 1. 定义文件夹名称
        log_dir = "Log"
        
        # 2. 如果文件夹不存在则创建
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.terminal = sys.stdout
        
        # 3. 将路径组合： Log/base_name_时间.log
        log_filename = f"{base_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # 4. 打开文件
        self.log = open(log_path, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
    

def main():
    sys.stdout = Logger()
    N_JOBS = max(1, os.cpu_count() - 1)
    
    # --- 模型初始化 ---
    D, F, H = 10.0, 6.0, 6.206
    h = 0.5*H - D**2/(16.0*F)
    hca_model = HCA_Mesh_Generator(n_r=4, n_theta=36, D=D, F=F, H=H, h=h)
    hca_model.generate_mesh()
    
    adapter = FDMAdapter(hca_model)
    ncoord, conn, q_v, elsets, bcs = adapter.get_solver_inputs()
    surf_eids = adapter.get_surf_cable_eids()
    solver = FDMSolver(ncoord=ncoord, conn=conn, bcs=bcs, elsets=elsets)
    optimizer = FDMOptimizer(solver, only_surface=True, surf_elset=surf_eids)

    # --- 加载初始解 ---
    seed_path = 'seeds/Opt_R1.85240_T16.93301.npy' # 确保此文件存在
    global_best_q = optimizer.load_seeds(seed_path)
    
    # ====== 分别计算 rmse=0 和 tr =1 时 q 的范围  
    # ======== 迭代法计算 RMSE 为 0  =====================
    qR = optimizer.run_iteration(global_best_q, max_iter=50000, rms_limit=1e-5)
    # 计算等张力时
    ten_eq = np.full(solver.num_elemt,1.0,dtype=float)
    qT = solver.compute_q(ten_eq)
    
    # --- 3. 向量归一化与对齐 ---
    # 消除绝对量级差异，只保留分布形态差异
    qR_norm = qR / np.mean(qR)
    qT_norm = qT / np.mean(qT)
    
    # 计算“精度”到“均匀”的映射向量
    q_target_vector = qT_norm / qR_norm
    
    # ======== 生成基准值 ========
    RMSE_MAX, TR_MAX = solver.get_RMSE_TR(global_best_q, surf_eids)
    # --- 核心阈值定义 ---
    # RMSE_MAX = 0.98   # 归一化基准：超过此值则淘汰
    # TR_MAX = 14.47    # 归一化基准：超过此值则淘汰
    
    # 初始评估
    c0, t0 = solver.solve(global_best_q)
    curr_rmse = 1e3 * np.sqrt(np.mean(np.sum((c0 - solver.ncoord)**2, axis=1)))
    curr_tr = np.max(t0[surf_eids]) / (np.min(t0[surf_eids]) + 1e-8)
    print(f"\n>>> 初始状态: RMSE={curr_rmse:.4f}, TR={curr_tr:.2f} (目标: <{RMSE_MAX}, <{TR_MAX})")

    # ================= STEP 1: GA2 滚动演化 =================
    # 1. 明确基准（由种子生成）
    RMSE_MAX, TR_MAX = solver.get_RMSE_TR(global_best_q, surf_eids)
    print(f"\n>>> 初始基准锁定: RMSE < {RMSE_MAX:.6f}, TR < {TR_MAX:.6f}")

    # ================= STEP 1: GA2 窄域搜索 (寻找更优前沿) =================
    # 如果 GA2 连续几轮没结果，说明需要更细腻的变异
    MAX_LOOPS = 0 
    for loop in range(1, MAX_LOOPS + 1):
        print(f"\n{'='*20} GA2 Refine Loop {loop} {'='*20}")
        
        # 将搜索范围从全域搜索 (0, 600) 切换为基于当前种子的 邻域搜索 (±10%)
        # 这能让算法在“狭缝”中生存的可能性提高 10 倍以上
        res_q, _, _ = optimizer.run_GA2(
            q_seeds=global_best_q,
            q_bounds=(global_best_q * 0.9, global_best_q * 1.1), # 关键：锁定邻域
            rmse_max=RMSE_MAX,
            tr_max=TR_MAX,
            pop_size=max(N_JOBS * 30, 200), # 增加种群密度
            n_gen=100,
            n_jobs=N_JOBS,
            mutation_eta=10 # 增加 eta，变异更加微小精细
        )

        c_l, t_l = solver.solve(res_q)
        rmse_l, tr_l = solver.get_RMSE_TR(res_q, surf_eids)

        # 只要有任何一项真正意义上的提升，就更新基准
        if (rmse_l < RMSE_MAX and tr_l < TR_MAX) or (tr_l < TR_MAX * 0.999):
            print(f"✨ 发现性能突破! TR: {TR_MAX:.4f}->{tr_l:.4f}")
            global_best_q, RMSE_MAX, TR_MAX = res_q, rmse_l, tr_l
        else:
            print("--- 邻域探测未发现显著改进 ---")


    # ================= STEP 2: GA1 动态滚动压榨 =================
    print("\n[Step 2] 进入 GA1 动态滚动压榨阶段...")
    
    GA1_LOOPS = 1
    current_q_init = global_best_q
    
    for g_loop in range(1, GA1_LOOPS + 1):
        print(f"\n{'='*20} GA1 导航轮次 {g_loop} {'='*20}")
        
        # 动态调整搜索步长：随着轮次增加，在物理走廊内越收越紧
        # alpha_ratio 控制向“等张力”方向移动的倾斜度
        shrink = 1.0 / g_loop
        
        # 为每个索组生成独立的数组边界 (Array Bounds)
        # xl: 该索组向等张力态移动的下限, xu: 上限
        xl = np.minimum(1.0, q_target_vector) * (1.0 - 0.05 * shrink)
        xu = np.maximum(1.0, q_target_vector) * (1.0 + 0.05 * shrink)
        
        # 执行 GA1
        res_q = optimizer.run_GA1(
            q_init=qR,                    # 始终以精度最准的点 qR 为参考基准
            q_ratio_bounds=(xl, xu),       # 传入量身定制的数组边界
            rmse_max=RMSE_MAX,
            tr_max=TR_MAX,
            weight_rms=0.01,              # 极度偏向 TR 优化
            pop_size=max(N_JOBS * 40, 240),# 增加种群密度以覆盖高维走廊
            n_gen=400,
            n_jobs=N_JOBS,
            alpha=80,                      # 适中的指数压力
            mutation_eta=30                # 较强的扰动，跳出局部最优
        )
        
        # 结果评估
        rmse_now, tr_now = solver.get_RMSE_TR(res_q, surf_eids)
        
        if tr_now < TR_MAX:
            _, tension = solver.solve(res_q)
            t_surf = tension[surf_eids]
            t_min, t_max = np.min(t_surf), np.max(t_surf)
            
            print(f"✨ 发现性能突破!")
            print(f"  [TR]   {TR_MAX:.5f} -> {tr_now:.5f} (改善: {(TR_MAX-tr_now)/TR_MAX*100:.2f}%)")
            print(f"  [RMSE] {RMSE_MAX:.5f} -> {rmse_now:.5f} mm")
            print(f"  [T-Range] {t_min:.2f}N ~ {t_max:.2f}N | CV: {np.std(t_surf)/np.mean(t_surf):.4f}")
            
            # 更新基准与参考点
            TR_MAX = tr_now
            RMSE_MAX = rmse_now
            current_q_init = res_q
            # 动态更新 qR 为当前找到的最优平衡点，以便下一轮在更窄范围内搜索
            qR = res_q 
        else:
            print(f"❌ 第{g_loop}轮未取得 TR 突破，保持当前基准。")

    final_best_q = current_q_init

    # ================= STEP 3: 导出 =================
    optimizer._print_status_report("Final Result", final_best_q)
    optimizer.save_seeds(final_best_q, "Opt")
    
    fc, ft = solver.solve(final_best_q)
    inp_name = f"R{RMSE_MAX:.5f}_T{tr_now:.5f}.inp"
    AbaqusExporter(solver).write_inp(f"Inp/{inp_name}", fc, ft)
    print("\n任务圆满完成。")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        import traceback
        traceback.print_exc()