import numpy as np
from generate_HCA_mesh import HCA_Mesh_Generator
from fdm_adapter import FDMAdapter
from fdm_solver import FDMSolver
from fdm_optimizer_v3 import FDMOptimizer
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
    hca_model = HCA_Mesh_Generator(n_r=4, n_theta=24)
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
    # 1. 初始化优化器
    # 建议指定只针对表面索（surf_eids）进行张力比优化
    optimizer = FDMOptimizer(solver, only_surface=only_surf, surf_elset=surf_eids)

    # 1. 强制 GA2 找一个“极准”的底子
    q_seed = optimizer.run_GA2(rmse_target=0.4, n_gen=100) 

    # 2. 启动半径锁定精修
    q_final = optimizer.run_GA1(
        q_seed, 
        rmse_target=1.0, 
        anneal_range=(0.98, 1.02), # 极小步长，仅允许 2% 波动
        n_gen=500
    )
    # 4. 最终物理校验
    print("\n>>> 阶段 3: 执行最终物理一致性校验...")
    final_coords, final_tensions = solver.solve(q_final)


    # =============== 写入 INP 文件 ====================
    exporter = AbaqusExporter(solver)
    exporter.write_inp("FDM_Result_Check.inp", final_coords, final_tensions)


if __name__ == "__main__":
    main()
