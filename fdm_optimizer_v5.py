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

            # 1. 计算基础指标
            rmse = np.sqrt(np.mean(np.sum((new_coords - self.solver.ncoord)**2, axis=1))) * 1000.0
            t_ratio = np.max(cable_force) / (np.min(cable_force) + 1e-8)

            # 2. 核心逻辑：精度锁 (Accuracy Lock)
            # 如果 RMSE 超过 1.0mm，目标函数只关注精度，忽略张力比
            # 更加激进的评价逻辑
            if rmse > 1.0:
                f = 5000.0 + (rmse ** 2) * 100 # 加大超限惩罚
            else:
                # 缩小 Ratio 的影响力，或者提升精度权重，强制冲向 0.998
                f = t_ratio + 1.0 * rmse

            # 3. 物理约束（位移和张力下限）
            penalty = 0.0
            if self.use_penalty:
                # 位移硬约束（Umax）
                node_dists = np.linalg.norm(new_coords - self.solver.ncoord, axis=1) * 1.0e3
                max_u = np.max(node_dists)
                if self.u_limit is not None and max_u > self.u_limit:
                    penalty += 2000.0 * (max_u / self.u_limit)

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
# 3. 核心优化器：FDMOptimizer
# =================================================================
class FDMOptimizer:
    def __init__(self, solver, only_surface=False, surf_elset=None):
        self.solver = solver
        self.cable_eids = list(range(self.solver.num_elemt))
        if only_surface:
            if surf_elset is None:
                raise ValueError("未传入面索单元索引集")
            self.cable_eids = surf_elset

    def _print_status_report(self, label, q):
        """计算并打印当前力密度下的所有关键指标"""
        coords, tensions = self.solver.solve(q)
        target_t = tensions[self.cable_eids]
        
        # 1. 几何精度 RMSE
        dists = np.linalg.norm(coords - self.solver.ncoord, axis=1) * 1000.0
        rmse = np.sqrt(np.mean(dists**2))
        max_u = np.max(dists)
        
        # 2. 张力指标
        t_min = np.min(target_t)
        t_max = np.max(target_t)
        t_ratio = t_max / (t_min + 1e-8)

        print(f"\n" + "="*50)
        print(f"  [{label}] 运行快报")
        print("-" * 50)
        print(f"  - 几何精度 RMSE:   {rmse:>10.4f} mm")
        print(f"  - 张力极值比 Ratio: {t_ratio:>10.4f}")
        print(f"  - 张力范围 Range:   [{t_min:.2f} ~ {t_max:.2f}] kN")
        print(f"  - 最大位移 Umax:    {max_u:>10.4f} mm")
        print("="*50 + "\n")

    def run_iteration(self, q_init=None, max_iter=1000, rms_limit=1.0):
        """标准迭代法"""
        if q_init is None:
            q_init = np.full(self.solver.num_group, 1.0, dtype=float)
        new_q = q_init
        
        print(f"{'Iter':^8} | {'RMSE (mm)':^15} | {'Tension Ratio':^18}")
        for i in range(max_iter):
            ncoords, tension = self.solver.solve(new_q)
            new_q = self.solver.compute_q(tension)
            rmse = 1.0e3 * np.sqrt(np.mean(np.sum((ncoords - self.solver.ncoord) ** 2, axis=1)))
            ten_ratio = np.max(tension[self.cable_eids]) / (np.min(tension[self.cable_eids]) + 1e-8)
            print(f"{i+1:^8} | {rmse:>15.4f} | {ten_ratio:>18.4f}")
            if rmse < rms_limit: break
        return new_q

    def run_GA2(self, q_bounds=(0.1, 150.0), pop_size=200, n_gen=250, 
                u_limit=100.0, t_limits=(10.0, 500.0), penalty=True, plot_pareto=True):
        """多目标 NSGA-II 搜索 (优先选择均方根 RMSE 较小的解)"""
        print(f"\n>> GA2：全局多目标搜索开始 (种群: {pop_size}, 代数: {n_gen})")

        problem = FDMGA2Problem(
            self.solver, q_bounds, self.cable_eids,
            u_limit=u_limit, t_limits=t_limits, use_penalty=penalty
        )

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)

        if res.X is None: raise RuntimeError("GA2 未找到可行解集")

        # --- 种子筛选策略：优先选择均方根较小的解 ---
        F = res.F
        # 按照第一个目标 (RMSE) 升序排列
        sorted_indices = np.argsort(F[:, 0])
        # 筛选出 RMSE 小于 1.0mm 且满足基本张力约束的解
        rmse_strict_mask = (F[:, 0] < 1.0)
        
        if np.any(rmse_strict_mask):
            # 如果有精度达标的，选里面张力比 (F[:,1]) 最小的
            candidate_indices = np.where(rmse_strict_mask)[0]
            idx_selected = candidate_indices[np.argmin(F[candidate_indices, 1])]
            print(f">> 策略：在精度达标(1mm)解集中选择最优张力比解")
        else:
            # 如果精度都没达标，选 RMSE 最小的那个
            idx_selected = np.argmin(F[:, 0])
            print(f">> 策略：未发现精度达标解，已选择当前 RMSE 最小解")

        # 打印详细结果
        self._print_status_report("GA2 阶段性最优解", res.X[idx_selected])

        if plot_pareto:
            self.plot_pareto_front(F, idx_selected)

        return res.X[idx_selected], res.X, res.F

    def run_GA1(self, q_init, q_ratio_bounds=(0.8, 1.2), t_limits=(10.0, 500.0), 
                u_limit=70.0, pop_size=100, n_gen=100, penalty=True):
        """单目标 GA 精修 (并在完成后输出报告)"""
        print(f"\n>> GA1：局部精修开始 (搜索半径: {q_ratio_bounds})")
        
        problem = FDMGA1Problem(
            solver=self.solver, q_init=q_init, q_range_ratio=q_ratio_bounds,
            elset_indices=self.cable_eids, t_limits=t_limits, 
            u_limit=u_limit, use_penalty=penalty
        )

        # 以当前种子为中心进行采样
        pop_init = Population.new("X", np.ones((1, self.solver.num_group)))
        algorithm = GA(
            pop_size=pop_size, 
            sampling=pop_init, 
            crossover=SBX(prob=0.9, eta=25), 
            mutation=PM(eta=35)
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)
        
        if res.X is None:
            print(">> GA1 优化异常，保留原解")
            return q_init
            
        final_q = q_init * res.X
        
        # 打印详细结果
        self._print_status_report("GA1 精修完成解", final_q)
        
        return final_q

    def plot_pareto_front(self, F_set, best_idx=None):
        plt.figure(figsize=(8, 5))
        plt.scatter(F_set[:, 0], F_set[:, 1], c="royalblue", s=25, alpha=0.6, label="Pareto Solutions")
        if best_idx is not None:
            plt.scatter(F_set[best_idx, 0], F_set[best_idx, 1], c="crimson", marker="*", s=150, label="Selected Seed")
        plt.xlabel("Geometry RMSE (mm)")
        plt.ylabel("Tension Ratio (Max/Min)")
        plt.title("GA2 Pareto Front Analysis")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()
