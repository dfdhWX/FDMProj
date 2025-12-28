import pygad
import numpy as np
from scipy.optimize import least_squares
from fdm_solver import FDMSolver
class FDMOptimizer:
    def __init__(self, solver:FDMSolver):
        self.solver = solver

    def run_lse(self, target_tensions, q_init):
        """
        最小二乘法寻优
        """
        # 定义函数句柄：变量 x 映射到残差向量
        def objective_handle(q_val):
            _, current_tensions = self.solver.solve(q_val)
            return current_tensions - target_tensions

        print("正在启动 LSE 优化...")
        res = least_squares(objective_handle, q_init, bounds=(1e-5, np.inf))
        return res.x

    def run_ga(self, target_tensions, q_bounds=(0.1, 100.0)):
        """
        遗传算法寻优 (PyGAD)
        """
        # 定义函数句柄：变量 solution 映射到适应度标量
        def fitness_handle(ga_instance, solution, solution_idx):
            _, current_tensions = self.solver.solve(solution)
            error = np.linalg.norm(current_tensions - target_tensions)
            return 1.0 / (error + 1e-8)

        ga_instance = pygad.GA(
            num_generations=50,
            num_parents_mating=5,
            fitness_func=fitness_handle, # 传入句柄
            sol_per_pop=20,
            num_genes=self.solver.num_elements,
            gene_space={'low': q_bounds[0], 'high': q_bounds[1]}
        )
        
        ga_instance.run()
        best_q, _, _ = ga_instance.best_solution()
        return best_q

    def run_dr(self, q_init):
        """
        动力松弛法寻优 (独立逻辑)
        """
        # DR 算法不需要传入 objective_handle，因为它内部就是物理时程积分
        # 它直接利用 self.solver.solve 获取不平衡力 R
        pass