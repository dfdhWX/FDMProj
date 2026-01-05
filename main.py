import numpy as np
from generate_HCA_mesh import HCA_Mesh_Generator
from fdm_adapter import FDMAdapter
from fdm_solver import FDMSolver
from fdm_optimizer_v7 import FDMOptimizer
from abaqus_expoter import AbaqusExporter
import sys
import datetime
import os

class Logger(object):
    """
    实验记录本模式日志：
    1. 每次运行生成独立文件，防止覆盖旧数据。
    2. 仅在文件头部打印运行快报，不干扰正文输出。
    """
    def __init__(self, base_name="optimization"):
        # 1. 生成带时间戳的文件名 (例如: fdm_20251230_2255.log)
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.filename = f"{base_name}_{timestamp}.log"
        
        self.terminal = sys.stdout
        self.log = open(self.filename, "w", encoding='utf-8')
        
        # 2. 构建运行表头 (Header)
        header = [
            "╔" + "═" * 58 + "╗",
            "║" + " FDM 结构优化任务运行快报 ".center(53) + "║",
            "╠" + "═" * 58 + "╣",
            f"║  启动时刻: {now.strftime('%Y-%m-%d %H:%M:%S'):<41}║",
            f"║  日志路径: {os.path.abspath(self.filename):<41}║",
            f"║  Python版本: {sys.version.split(' ')[0]:<39}║",
            "╚" + "═" * 58 + "╝\n"
        ]
        
        header_text = "\n".join(header)
        
        # 同时写入终端和文件
        self.terminal.write(header_text)
        self.log.write(header_text)

    def write(self, message):
        # 原样写入，不添加行前缀，保持控制台原貌
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        pass

# 在 main.py 最上方启动
sys.stdout = Logger("fdm_run")



def main():
    # =================创建HCA网格生成器实例=================
    hca_model = HCA_Mesh_Generator(n_r=4, n_theta=36)
    # 生成HCA网格
    hca_model.generate_mesh()
    # 绘制网格图
    # hca_model.mesh_plot(show_labels=False)


    # =================
    adapter = FDMAdapter(hca_model)
    ncoord, conn, q_v, elsets, bcs = adapter.get_solver_inputs()

    # =============== FDM Solver ====================
    solver = FDMSolver(ncoord=ncoord, conn=conn, bcs=bcs, elsets=elsets)

    ## ============== FDM 优化 =========================
    # 只考虑面索的张力均匀性
    only_surf = True
    surf_eids = adapter.get_surf_cable_eids()

    # 1. 实例化优化器
    optimizer = FDMOptimizer(
        solver, 
        only_surface=only_surf, 
        surf_elset=surf_eids
    )
    
    # =============== 加载种子 ==================
    # q_seed = optimizer.load_seeds('seed_RMSE_1.46_TR_12.38.npy')

    # ---------- GA2：全局多目标搜索 ----------
    # 目标：在大范围内找到 RMSE 和 Tension Ratio 的 Pareto 前沿
    q_base, all_q, all_F = optimizer.run_GA2(
        # q_seeds= q_seed,
        q_bounds=(0.1, 150.0), # 这里的范围根据你的工程经验设置
        u_limit=100.0,
        t_limits=(10.0, 500.0),
        n_gen=250,
        pop_size=200,
    )
    
    q_base, all_q, all_F = optimizer.run_GA2(
        q_seeds=all_q,
        q_bounds=(0.1, 150.0), # 这里的范围根据你的工程经验设置
        u_limit=100.0,
        t_limits=(10.0, 500.0),
        n_gen=200,
        pop_size=200,
    )

    # =================================================================
    # 核心策略：从“精度优先”平滑过渡到“张力均匀”
    # =================================================================

    # ---------- 第一轮：快速降准 (RMSE -> 1.0mm) ----------
    # 策略：高强度压低几何误差，不计张力比代价
    q_mid = optimizer.run_GA1(
        q_init=q_base,
        q_ratio_bounds=(0.85, 1.15),
        RMSE_weight=0.8,
        RMSE_max= 1.0,
        RMSE_tol = 0.8,
        u_limit=70.0,
        n_gen=200
    )

    # ---------- 第二轮：深度下探 (RMSE -> 0.1mm) ----------
    # 策略：进入 1mm 以内后，开始兼顾张力比
    best_q = optimizer.run_GA1(
        q_init=q_mid,
        q_ratio_bounds=(0.85, 1.15),
        RMSE_weight=0.3,
        RMSE_max= 1.0,
        RMSE_tol = 0.8,
        u_limit=70.0,
        n_gen=200
    )
    
    #====== 保存优秀的种子 ============
    optimizer.save_seeds(best_q)

    # 4. 最终校验
    final_coords, final_tensions = solver.solve(best_q)



    # =============== 写入 INP 文件 ====================
    exporter = AbaqusExporter(solver)
    exporter.write_inp("FDM_Result_Check.inp", final_coords, final_tensions)


if __name__ == "__main__":
    main()
