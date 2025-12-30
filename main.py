import numpy as np
from generate_HCA_mesh import HCA_Mesh_Generator
from fdm_adapter import FDMAdapter
from fdm_solver import FDMSolver
from fdm_optimizer import FDMOptimizer
from abaqus_expoter import AbaqusExporter


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
    optimizer = FDMOptimizer(solver, only_surface=only_surf, surf_elset=surf_eids)
    ## 启动优化
    # best_q_groups, all_X, all_F = optimizer.run_GA2(n_gen=200,pop_size=100)
    
    
    ##--------
    # 1. 自动生成初始力密度 (等张力反推)
    #======= 通过迭代求解初始值======
    # q_init = optimizer.run_iteration()
    
    #====== GA2 获取初始值=======
    # 第一步：GA2 全局搜寻，只要位移在 100mm 以内且张力不松弛即可
    q_base, all_X, all_F = optimizer.run_GA2(
        u_limit=100.0, 
        t_limits=(10.0, 500.0), 
        n_gen=250,
        pop_size=200,
        plot_pareto=False,
    )

    # 第二步：GA1 局部精修，利用第一步的基准，全力优化张力比
    best_q = optimizer.run_GA1(
        q_init=q_base,
        weight=0.8,       # 给张力均匀性更多权重
        u_limit=50.0,      # 精度控制在 50mm 以内
        n_gen=500,
        pop_size=150,
    )

    # 利用最优力密度重新计算最终的物理状态
    final_coords, final_tensions = solver.solve(best_q)

    # =============== 写入 INP 文件 ====================
    exporter = AbaqusExporter(solver)
    exporter.write_inp("FDM_Result_Check.inp", final_coords, final_tensions)


if __name__ == "__main__":
    main()
