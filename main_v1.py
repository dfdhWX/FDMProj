import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, os, datetime
from generate_HCA_mesh_v0 import HCA_Mesh_Generator
from fdm_adapter import FDMAdapter
from fdm_solver import FDMSolver
from optimizer_v1 import FDMOptimizer
from abaqus_expoter import AbaqusExporter
from post import save_to_matlab

class Logger(object):
    def __init__(self, base_name="fdm_opt"):
        log_dir = "Log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.terminal = sys.stdout
        log_filename = f"{base_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(log_dir, log_filename)
        self.log = open(log_path, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
    
    
def save_data(data, filename, header=None):
    """
    将给定数据存储在项目根目录下的 data 文件夹中。
    支持 .csv 带表头存储和 .npy 结构化存储。
    """
    target_dir = os.path.join(os.getcwd(), "data")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"[System] 已创建目录: {target_dir}")
    
    file_path = os.path.join(target_dir, filename)
    
    # --- 核心逻辑修改 ---
    
    if filename.endswith(".csv"):
        # 如果提供了 header（列表或逗号分隔字符串），则写入 CSV 第一行
        # comments="" 确保表头前没有 '#' 符号
        header_str = ",".join(header) if isinstance(header, list) else (header or "")
        np.savetxt(file_path, data, delimiter=",", header=header_str, comments="")
        
    elif filename.endswith(".npy"):
        # 如果是 numpy 数组且提供了 header，将其转换为结构化数组存储
        if header and isinstance(header, list) and isinstance(data, np.ndarray):
            # 自动构建数据类型：例如 [('gen', 'f8'), ('f_min', 'f8'), ...]
            dtype = [(name, data.dtype) for name in header]
            # 转换数据格式
            structured_data = np.empty(data.shape[0], dtype=dtype)
            for i, name in enumerate(header):
                structured_data[name] = data[:, i]
            np.save(file_path, structured_data)
        else:
            np.save(file_path, data)
            
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(data))
            
    print(f"✨ [IO] 数据已成功保存至: {file_path}")
    return file_path

 
def main():
    sys.stdout = Logger()
    N_JOBS = max(1, os.cpu_count() - 1)
    
    # --- 1. 模型初始化 ---
    D, F, H = 10.0, 6.0, 6.206
    h = 0.5*H - D**2/(16.0*F)
    hca_model = HCA_Mesh_Generator(n_r=4, n_theta=36, D=D, F=F, H=H, h=h)
    hca_model.generate_mesh()
    
    adapter = FDMAdapter(hca_model)
    ncoord, conn, q_v, elsets, bcs = adapter.get_solver_inputs()
    surf_eids = adapter.get_surf_cable_eids()
    solver = FDMSolver(ncoord=ncoord, conn=conn, bcs=bcs, elsets=elsets)
    optimizer = FDMOptimizer(solver, only_surface=True, surf_elset=surf_eids)
    

    # --- 2. 加载种子并计算物理两极 ---
    seed_path = 'seeds/Opt_R1.85240_T16.93301.npy' 
    global_best_q = optimizer.load_seeds(seed_path)
    
    print("\n[Prep] 计算精度极限点 qR...")
    qR_limit = optimizer.run_iteration(global_best_q, max_iter=50000, rms_limit=1e-5)
    
    print("[Prep] 计算等张力极限点 qT...")
    # 手动计算 T=1.0 时的理想 qT（避开 solver 索引问题）
    diffs_ideal = solver.C @ solver.ncoord
    lengths_ideal = np.linalg.norm(diffs_ideal, axis=1)
    qT_limit = np.zeros(solver.num_group)
    elsets_list = list(solver.elsets.values()) if isinstance(solver.elsets, dict) else solver.elsets
    for i, idx in enumerate(elsets_list):
        qT_limit[i] = np.mean(1.0 / (lengths_ideal[idx] + 1e-9))
    
    # 归一化极值向量
    qR_norm = qR_limit / np.mean(qR_limit)
    qT_norm = qT_limit / np.mean(qT_limit)
    
    # --- 3. 初始基准评估 ---
    RMSE_MAX, TR_MAX = solver.get_RMSE_TR(global_best_q, surf_eids)
    print(f"\n>>> 初始锁定基准: RMSE < {RMSE_MAX:.6f}, TR < {TR_MAX:.6f}")

    # ================= STEP 2: GA1 动态滚动导航 =================
    print("\n[Step 2] 进入 GA1 动态滚动压榨阶段...")
    
    GA1_LOOPS = 1 # 建议多跑几轮，动态基准效果更好
    current_q_init = global_best_q 
    
    for g_loop in range(1, GA1_LOOPS + 1):
        print(f"\n{'='*20} GA1 滚动导航轮次 {g_loop} {'='*20}")
        
        # --- 核心：动态向量对齐 ---
        # 计算当前点相对于等张力目标的偏离向量
        q_curr_norm = current_q_init / np.mean(current_q_init)
        q_target_vector = qT_norm / q_curr_norm 
        
        # --- 核心：动态走廊边界 ---
        shrink = 1.0 / g_loop
        # 允许搜索范围在“目标方向”上稍微放宽，在“反方向”上收紧
        xl = np.minimum(1.0, q_target_vector) * (1.0 - 0.05 * shrink)
        xu = np.maximum(1.0, q_target_vector) * (1.0 + 0.05 * shrink)
        
        # 执行 GA1，直接将 current_q_init 作为基准传入
        res_q, history_data = optimizer.run_GA1(
            q_init=current_q_init, 
            q_ratio_bounds=(xl, xu),
            rmse_max=RMSE_MAX,
            tr_max=TR_MAX,
            weight_rms=0.5,              # 极度偏向 TR 优化
            pop_size=max(N_JOBS * 40, 240),
            n_gen=200,
            n_jobs=N_JOBS,
            alpha=100,                     # 增强选择压力
            mutation_eta=30 + g_loop * 5,  # 随轮次增加变异精度
            use_penalty=False,
        )
        
        # 评估效果
        rmse_now, tr_now = solver.get_RMSE_TR(res_q, surf_eids)
        
        if tr_now < TR_MAX:
            _, tension = solver.solve(res_q)
            t_surf = tension[surf_eids]
            
            print(f"✨ 第 {g_loop} 轮发现突破!")
            print(f"  [TR]   {TR_MAX:.5f} -> {tr_now:.5f} (改善: {(TR_MAX-tr_now)/TR_MAX*100:.2f}%)")
            print(f"  [RMSE] {RMSE_MAX:.5f} -> {rmse_now:.5f} mm")
            print(f"  [张力] Min: {np.min(t_surf):.2f}N | Max: {np.max(t_surf):.2f}N | CV: {np.std(t_surf)/np.mean(t_surf):.4f}")
            
            # 更新滚动基准
            TR_MAX = tr_now
            RMSE_MAX = rmse_now
            current_q_init = res_q # 下一轮将以此解为中心搜索
        else:
            print(f" 第 {g_loop} 轮未取得突破，尝试在下一轮微调...")

    # ================= STEP 3: 最终导出 =================
    final_best_q = current_q_init
    optimizer._print_status_report("Final Optimized Result", final_best_q)
    
    # 获取最终物理状态
    fc, ft = solver.solve(final_best_q) # fc: coords, ft: tensions
    
    # =============== 保存结果 ====================
    tag = f"hca_R{RMSE_MAX:.4f}_T{TR_MAX:.2f}"
    
    # 1. 保存种子 (.npy)
    optimizer.save_seeds(final_best_q, tag)
    
    # 2. 保存优化历史 (.csv)
    # 建议带上表头方便读取
    history_header = ["gen", "f_min", "f_avg", "rmse", "tr", "f1", "f2"]
    save_data(history_data, f"{tag}_history.csv", header=history_header)
    
    # 3. 保存 Abaqus 校验文件 (.inp)
    AbaqusExporter(solver).write_inp(f"Inp/{tag}.inp", fc, ft)
    
    # 4. 新增：保存 MATLAB 绘图文件 (.mat)
    save_to_matlab(adapter, fc, ft, filename=f"{tag}_mesh.mat")

    print("\n[Done] 任务圆满完成。所有数据已分类存储至 data/, seeds/, Inp/ 目录。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()