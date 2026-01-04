import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.population import Population

# =================================================================
# 1. GA1 问题定义：单目标优化 (侧重于在特定范围内精修张力均匀度)
# =================================================================
class FDMGA1Problem(ElementwiseProblem):
    def __init__(self, solver, q_init, q_range_ratio, elset_indices,
                 t_limits=None, u_limit=None, ratio_limit=2.0, use_penalty=True):

        n_groups = solver.num_group
        super().__init__(
            n_var=n_groups,
            n_obj=1,
            xl=q_range_ratio[0],
            xu=q_range_ratio[1]
        )

        self.solver = solver
        self.q_init = q_init
        self.cable_elset = elset_indices
        self.t_limits = t_limits
        self.u_limit = u_limit
        self.ratio_limit = ratio_limit
        self.use_penalty = use_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        current_q = self.q_init * x
        try:
            new_coords, all_tensions = self.solver.solve(current_q)
            cable_force = all_tensions[self.cable_elset]

            # 目标：张力极值比 (Max/Min)
            t_min = np.min(cable_force)
            t_max = np.max(cable_force)
            t_ratio = t_max / (t_min + 1e-8)

            f = t_ratio
            penalty = 0.0

            if self.use_penalty:
                # 约束 1: 位移硬约束
                node_dists = np.linalg.norm(new_coords - self.solver.ncoord, axis=1) * 1.0e3
                max_u = np.max(node_dists)
                if self.u_limit is not None and max_u > self.u_limit:
                    penalty += 1000.0 * (max_u / self.u_limit)**2

                # 约束 2: 张力下限（防松弛）
                if self.t_limits is not None and t_min < self.t_limits[0]:
                    penalty += 1000.0 * (self.t_limits[0] / (t_min + 1e-8))**2

            out["F"] = f + penalty
        except Exception:
            out["F"] = 1e10

# =================================================================
# 2. GA2 问题定义：多目标优化 (RMSE vs Tension Ratio)
# =================================================================
class FDMGA2Problem(ElementwiseProblem):
    def __init__(self, solver, q_bounds, cable_eids, u_limit=100.0, t_limits=(5.0, 500.0), use_penalty=True):
        n_groups = solver.num_group
        super().__init__(n_var=n_groups, n_obj=2, n_constr=0, 
                         xl=q_bounds[0], xu=q_bounds[1])
        self.solver = solver
        self.cable_eids = cable_eids
        self.u_limit = u_limit
        self.t_limits = t_limits
        self.use_penalty = use_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            new_coords, tensions = self.solver.solve(x)
            target_t = tensions[self.cable_eids]
            
            # 目标 1: 几何 RMSE (mm)
            dists = np.linalg.norm(new_coords - self.solver.ncoord, axis=1) * 1000
            f1 = np.sqrt(np.mean(dists**2))
            
            # 目标 2: 张力比 (Max/Min)
            t_min = np.min(target_t)
            t_max = np.max(target_t)
            f2 = t_max / (t_min + 1e-8)
            
            penalty = 0.0
            if self.use_penalty:
                # 位移超限惩罚
                max_u = np.max(dists)
                if max_u > self.u_limit:
                    penalty += 50.0 * (max_u - self.u_limit)**2
                
                # 张力下限惩罚
                if t_min < self.t_limits[0]:
                    penalty += 200.0 * (self.t_limits[0] - t_min)**2

            out["F"] = [f1 + penalty, f2 + penalty]
        except Exception:
            out["F"] = [1e10, 1e10]

# =================================================================
# 3. 优化器封装类
# =================================================================
class FDMOptimizer:
    def __init__(self, solver, only_surface=False, surf_elset=None):
        self.solver = solver
        self.cable_eids = list(range(self.solver.num_elemt))
        if only_surface:
            if surf_elset is None:
                raise ValueError("未传入面索单元索引集")
            self.cable_eids = surf_elset

    def run_iteration(self, q_init=None, max_iter=1000, rms_limit=1.0):
        """标准迭代法"""
        if q_init is None:
            q_init = np.full(self.solver.num_group, 1.0, dtype=float)
        new_q = q_init
        
        header = f"{'Iter':^8} | {'RMSE (mm)':^15} | {'Tension Ratio':^18}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for i in range(max_iter):
            ncoords, tension = self.solver.solve(new_q)
            new_q = self.solver.compute_q(tension)
            
            rmse = 1.0e3 * np.sqrt(np.mean(np.sum((ncoords - self.solver.ncoord) ** 2, axis=1)))
            target_t = tension[self.cable_eids]
            ten_ratio = np.max(target_t) / (np.min(target_t) + 1e-8)
            
            print(f"{i+1:^8} | {rmse:>15.4f} | {ten_ratio:>18.4f}")
            if rmse < rms_limit: break
        return new_q

    def run_GA1(self, q_init, q_ratio_bounds=(0.8, 1.2), t_limits=(10.0, 500.0), 
                u_limit=70.0, pop_size=100, n_gen=100, penalty=True):
        """单目标 GA 精修"""
        print("\n>> GA1：局部精修 (目标: 最小化 Tension Ratio)")
        
        problem = FDMGA1Problem(
            solver=self.solver, q_init=q_init, q_range_ratio=q_ratio_bounds,
            elset_indices=self.cable_eids, t_limits=t_limits, 
            u_limit=u_limit, use_penalty=penalty
        )

        pop_init = Population.new("X", np.ones((1, self.solver.num_group)))
        algorithm = GA(pop_size=pop_size, sampling=pop_init, 
                       crossover=SBX(prob=0.9, eta=20), mutation=PM(eta=30))

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)
        
        if res.X is None: return q_init
        final_q = q_init * res.X
        return final_q

    def run_GA2(self, q_bounds=(0.1, 150.0), pop_size=200, n_gen=250, 
                u_limit=100.0, t_limits=(10.0, 500.0), penalty=True, plot_pareto=True):
        """多目标 NSGA-II 搜索"""
        print("\n>> GA2：全局多目标搜索 (RMSE & Tension Ratio)")

        problem = FDMGA2Problem(
            self.solver, q_bounds, self.cable_eids,
            u_limit=u_limit, t_limits=t_limits, use_penalty=penalty
        )

        algorithm = NSGA2(
            pop_size=pop_size, sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)

        if res.X is None: raise RuntimeError("GA2 未找到可行解")

        # 筛选满足物理约束的解
        candidates = []
        for i in range(len(res.X)):
            coords, tensions = self.solver.solve(res.X[i])
            target_t = tensions[self.cable_eids]
            max_u = np.max(np.linalg.norm(coords - self.solver.ncoord, axis=1)) * 1e3
            if max_u < u_limit and np.min(target_t) > t_limits[0]:
                candidates.append(i)

        if candidates:
            # 在可行解中找 Tension Ratio (res.F[:,1]) 最小的
            idx_selected = min(candidates, key=lambda i: res.F[i, 1])
            print(f">> 已选定可行域内最优张力比解: #{idx_selected}")
        else:
            idx_selected = self._get_elbow_point_idx(res.F)
            print(">> 未发现严格可行解，选择 Pareto 折中解")

        if plot_pareto:
            self.plot_pareto_front(res.F, idx_selected)

        return res.X[idx_selected], res.X, res.F

    def _get_elbow_point_idx(self, F_set):
        f_min, f_max = F_set.min(axis=0), F_set.max(axis=0)
        F_norm = (F_set - f_min) / (f_max - f_min + 1e-8)
        return np.argmin(np.linalg.norm(F_norm, axis=1))

    def plot_pareto_front(self, F_set, best_idx=None):
        plt.figure(figsize=(8, 5))
        plt.scatter(F_set[:, 0], F_set[:, 1], c="royalblue", s=30, alpha=0.7, label="Pareto Front")
        if best_idx is not None:
            plt.scatter(F_set[best_idx, 0], F_set[best_idx, 1], c="red", marker="*", s=150, label="Selected Best")
        plt.xlabel("Geometry RMSE (mm)")
        plt.ylabel("Tension Ratio (Max/Min)")
        plt.title("GA2 Optimization: RMSE vs Tension Ratio")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.show()