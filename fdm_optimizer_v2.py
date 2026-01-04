import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.population import Population

# =================================================================
# 1. GA1-v2 问题定义：主动内力重分配 (突破张力比瓶颈的关键)
# =================================================================
class FDMGA1Problem(ElementwiseProblem):
    def __init__(self, solver, q_init, elset_indices, rmse_target=1.0, t_min_target=5.0):
        n_groups = solver.num_group
        # 核心改变：变量 x 是相对于 q_init 的独立系数，允许各组力密度 [0.5, 2.0] 倍异构变动
        super().__init__(n_var=n_groups, n_obj=1, xl=0.5, xu=2.0)
        self.solver = solver
        self.q_init = q_init
        self.cable_elset = elset_indices
        self.rmse_target = rmse_target
        self.t_min_target = t_min_target # 主动引导目标

    def _evaluate(self, x, out, *args, **kwargs):
        # 物理实现：q_new = q_old * x (x 向量化，各组独立)
        current_q = self.q_init * x
        new_coords, all_tensions = self.solver.solve(current_q)
        
        # 1. 基础指标计算
        rmse = np.sqrt(np.mean(np.sum((new_coords - self.solver.ncoord)**2, axis=1))) * 1000.0
        target_t = all_tensions[self.cable_elset]
        t_min, t_max = np.min(target_t), np.max(target_t)
        t_ratio = t_max / (t_min + 1e-8)

        # 2. 惩罚项：精度红线
        penalty = 0.0
        if rmse > self.rmse_target:
            penalty += 10000.0 * (rmse - self.rmse_target)**2
        
        # 3. 主动驱动力：将提升 Tmin 设为显式目标，而不仅仅是惩罚
        # 引导算法主动增强那些“弱索”，实现内力的自发均匀化
        t_min_drive = 0.0
        if t_min < self.t_min_target:
            t_min_drive = 200.0 * (self.t_min_target - t_min)

        # 4. 综合评价函数：张力比 + 弱索引导 + 精度残差 + 惩罚
        # 这种构造让算法意识到：调大力密度去救弱索是“获益”的
        out["F"] = t_ratio + t_min_drive + penalty + 0.1 * rmse

# =================================================================
# 2. GA2 问题定义：多目标余量搜索 (为精修阶段预留构形空间)
# =================================================================
class FDMGA2Problem(ElementwiseProblem):
    def __init__(self, solver, q_bounds, cable_eids, rmse_target=1.0, t_min_limit=2.0):
        n_groups = solver.num_group
        super().__init__(n_var=n_groups, n_obj=2, xl=q_bounds[0], xu=q_bounds[1])
        self.solver = solver
        self.cable_eids = cable_eids
        self.rmse_target = rmse_target
        self.t_min_limit = t_min_limit

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            new_coords, tensions = self.solver.solve(x)
            target_t = tensions[self.cable_eids]
            rmse = np.sqrt(np.mean(np.sum((new_coords - self.solver.ncoord)**2, axis=1))) * 1000.0
            t_min = np.min(target_t)
            t_ratio = np.max(target_t) / (t_min + 1e-8)
            
            penalty = 0.0
            if t_min < self.t_min_limit:
                penalty += 5000.0 * (self.t_min_limit - t_min)

            # 构形精度目标 f1, 张力平衡目标 f2
            f1 = rmse + penalty
            f2 = t_ratio + penalty
            
            # 精度断头台：若不达标，严重削弱张力比目标的竞争力
            if rmse > self.rmse_target:
                f2 += 10000.0 + (rmse - self.rmse_target) * 1000.0
            else:
                f1 = rmse * 0.1 # 达标后给精度一个极小的权重

            out["F"] = [f1, f2]
        except:
            out["F"] = [1e10, 1e10]

# =================================================================
# 3. FDMOptimizer 控制类
# =================================================================
class FDMOptimizer:
    def __init__(self, solver, only_surface=False, surf_elset=None):
        self.solver = solver
        self.cable_eids = surf_elset if only_surface else list(range(self.solver.num_elemt))

    def run_iteration(self, q_init=None, max_iter=500, rms_limit=0.5):
        curr_q = q_init if q_init is not None else np.full(self.solver.num_group, 5.0)
        for i in range(max_iter):
            ncoords, tension = self.solver.solve(curr_q)
            rmse = np.sqrt(np.mean(np.sum((ncoords - self.solver.ncoord)**2, axis=1))) * 1000.0
            if rmse < rms_limit: break
            curr_q = self.solver.compute_q(tension)
        return curr_q, rmse

    def run_GA2(self, q_init=None, q_bounds=(0.1, 200.0), pop_size=200, n_gen=100, rmse_target=1.0, show_plot=False):
        # 1. 获取种子并注入多样性
        if q_init is None:
            q_seed, seed_rmse = self.run_iteration(rms_limit=rmse_target/1.2)
        else:
            q_seed = q_init
            coords, _ = self.solver.solve(q_seed)
            seed_rmse = np.sqrt(np.mean(np.sum((coords - self.solver.ncoord)**2, axis=1))) * 1000.0

        X_init = np.random.uniform(q_bounds[0], q_bounds[1], (pop_size, self.solver.num_group))
        for i in range(int(pop_size * 0.6)):
            X_init[i, :] = q_seed * np.random.uniform(0.7, 1.3, self.solver.num_group)
        X_init[0, :] = q_seed

        # 2. 优化
        problem = FDMGA2Problem(self.solver, q_bounds, self.cable_eids, rmse_target=rmse_target)
        algorithm = NSGA2(pop_size=pop_size, sampling=Population.new("X", np.clip(X_init, q_bounds[0], q_bounds[1])))
        
        print(f"\n[Step: GA2 全局探索] 目标精度: {rmse_target}mm | 种子精度: {seed_rmse:.4f}mm")
        res = minimize(problem, algorithm, ("n_gen", n_gen), verbose=True)

        # 3. 结果筛选
        valid = res.F[:, 0] <= rmse_target
        idx = np.where(valid)[0][np.argmin(res.F[valid, 1])] if any(valid) else np.argmin(res.F[:, 0])
        
        self.plot_pareto_front(res.F, best_idx=idx, rmse_target=rmse_target, show_plot=show_plot)
        self._print_final_status("GA2 阶段性解", res.X[idx])
        return res.X[idx]

    def run_GA1(self, q_init, rmse_target=1.0, t_min_target=5.0, n_gen=300):
        # 注意：这里的变量范围 [0.5, 2.0] 赋予了算法重新分配内力的权力
        print(f"\n[Step: GA1 内力重构] 启动物理驱动优化 | 目标最小张力: {t_min_target}kN")
        problem = FDMGA1Problem(self.solver, q_init, self.cable_eids, rmse_target, t_min_target)
        
        pop_size = 120
        # 初始化：包含原始解及各种非对称扰动
        X_init = np.random.uniform(0.8, 1.2, (pop_size, self.solver.num_group))
        X_init[0, :] = 1.0 
        
        algorithm = GA(
            pop_size=pop_size, 
            sampling=Population.new("X", X_init),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.3, eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(problem, algorithm, ("n_gen", n_gen), verbose=True)
        final_q = q_init * res.X
        self._print_final_status("GA1 精修完成解", final_q)
        return final_q

    def plot_pareto_front(self, F_set, best_idx=None, rmse_target=1.0, show_plot=False):
        plt.figure(figsize=(10, 6))
        # 过滤掉惩罚后的超大值
        mask = F_set[:, 1] < 5000
        plt.scatter(F_set[mask, 0], F_set[mask, 1], c="royalblue", alpha=0.5, label="Candidates")
        plt.axvline(x=rmse_target, color="#ff4757", linestyle="--", label="RMSE Target")
        
        if best_idx is not None and F_set[best_idx, 1] < 5000:
            plt.scatter(F_set[best_idx, 0], F_set[best_idx, 1], c="#ffa502", marker="*", s=250, label="Selected Best")
        
        plt.title("Pareto Front: Accuracy vs. Tension Balance")
        plt.xlabel("Geometry RMSE (mm)"); plt.ylabel("Tension Ratio (Max/Min)")
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        if not os.path.exists("plots"): os.makedirs("plots")
        plt.savefig(f"plots/Optimization_PF_{int(time.time())}.png")
        if show_plot: plt.show()
        else: plt.close()

    def _print_final_status(self, label, q):
        coords, tensions = self.solver.solve(q)
        target_t = tensions[self.cable_eids]
        rmse = np.sqrt(np.mean(np.sum((coords - self.solver.ncoord) ** 2, axis=1))) * 1000.0
        t_max, t_min = np.max(target_t), np.min(target_t)
        print(f"\n" + "="*50)
        print(f" [{label}]")
        print(f"  - RMSE:  {rmse:.4f} mm")
        print(f"  - Ratio: {t_max/t_min:.4f}")
        print(f"  - Range: [{t_min:.2f} ~ {t_max:.2f}] kN")
        print("="*50 + "\n")