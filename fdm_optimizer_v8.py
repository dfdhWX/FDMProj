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
from pymoo.parallelization.joblib import JoblibParallelization
import os

# =================================================================
# 1. GA2 问题定义：多目标优化 (RMSE vs TR)
# =================================================================
class FDMGA2Problem(ElementwiseProblem):
    def __init__(self, solver, q_bounds, cable_eids, rmse_limit=1.0, 
                 t_limits=(10.0, 1500.0), **kwargs):
        n_groups = solver.num_group
        # n_obj=2: [RMSE + Penalty, TensionRatio + Penalty]
        super().__init__(n_var=n_groups, n_obj=2, n_constr=0, 
                         xl=q_bounds[0], xu=q_bounds[1], **kwargs)
        self.solver = solver
        self.cable_eids = cable_eids
        self.rmse_limit = rmse_limit
        self.t_limits = t_limits

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            new_coords, tensions = self.solver.solve(x)
            target_t = tensions[self.cable_eids]

            # 1. 计算几何误差 (mm)
            dists = np.linalg.norm(new_coords - self.solver.ncoord, axis=1) * 1000
            rmse = np.sqrt(np.mean(dists**2)) 

            # 2. 计算张力比 (TR)
            t_min, t_max = np.min(target_t), np.max(target_t)
            tr = t_max / (t_min + 1e-8)

            # 3. 目标值初始化
            f1, f2 = rmse, tr

            # 【核心死守逻辑】：阶跃惩罚
            # 如果 RMSE 超过硬指标(1.0)，施加巨大惩罚，使其在帕累托排序中完全失效
            if rmse > self.rmse_limit:
                f1 += 5000.0 + (rmse - self.rmse_limit) * 1000.0
                f2 += 5000.0 
            
            # 张力下限惩罚 (防止索松弛)
            if t_min < self.t_limits[0]:
                f2 += 1000.0 * (self.t_limits[0] - t_min)**2

            out["F"] = [f1, f2]
            
        except Exception:
            out["F"] = [1e10, 1e10]

# =================================================================
# 2. GA1 问题定义：单目标精修
# =================================================================
class FDMGA1Problem(ElementwiseProblem):
    def __init__(self, solver, q_init, q_range_ratio, elset_indices, 
                 RMSE_weight=0.5, RMSE_tol=0.8, t_limits=(10.0, 1500.0), **kwargs):
        super().__init__(n_var=solver.num_group, n_obj=1, 
                         xl=q_range_ratio[0], xu=q_range_ratio[1], **kwargs)
        self.solver = solver
        self.q_init = q_init
        self.cable_elset = elset_indices
        self.RMSE_weight = RMSE_weight
        self.RMSE_tol = RMSE_tol
        self.t_limits = t_limits

    def _evaluate(self, x, out, *args, **kwargs):
        current_q = self.q_init * x
        try:
            new_coords, all_tensions = self.solver.solve(current_q)
            cable_force = all_tensions[self.cable_elset]
            
            rmse = np.sqrt(np.mean(np.sum((new_coords - self.solver.ncoord) ** 2, axis=1))) * 1000.0
            t_min, t_max = np.min(cable_force), np.max(cable_force)
            tr = t_max / (t_min + 1e-8)

            # ⚠️ 死守 RMSE<1: 超过直接巨大惩罚
            if rmse > 1.0:
                out["F"] = 1e10 + (rmse - 1.0)*1e3
                return

            # 正常目标函数: 尽量压低 TR
            score_tr = tr
            out["F"] = score_tr

            # 物理硬约束
            if t_min < self.t_limits[0]:
                out["F"] += 500.0 * (self.t_limits[0] - t_min)**2

        except:
            out["F"] = 1e10


# =================================================================
# 3. 核心优化器：FDMOptimizer
# =================================================================
class FDMOptimizer:
    def __init__(self, solver, only_surface=False, surf_elset=None):
        self.solver = solver
        self.cable_eids = list(range(self.solver.num_elemt))
        if only_surface:
            if surf_elset is None: raise ValueError("未传入面索单元索引集")
            self.cable_eids = surf_elset

    def _get_runner(self, n_jobs):
        if n_jobs <= 1: return None
        return JoblibParallelization(n_jobs=n_jobs)

    def run_GA2(self, q_seeds=None, q_bounds=(0.05, 500.0), pop_size=200, n_gen=250, 
                rmse_limit=1.0, t_limits=(10.0, 1500.0), n_jobs=4, mutation_eta=20):
        
        runner = self._get_runner(n_jobs)
        problem = FDMGA2Problem(self.solver, q_bounds, self.cable_eids, 
                                rmse_limit=rmse_limit, t_limits=t_limits, 
                                elementwise_runner=runner)

        # 初始种群处理
        if q_seeds is not None:
            seeds = np.atleast_2d(q_seeds)
            n_fill = pop_size - len(seeds)
            # 生成围绕种子的微扰个体
            clones = seeds[0] * (1 + 0.1 * np.random.randn(n_fill, self.solver.num_group))
            sampling = Population.new("X", np.clip(np.vstack([seeds, clones]), q_bounds[0], q_bounds[1]))
        else:
            sampling = FloatRandomSampling()

        algorithm = NSGA2(
            pop_size=pop_size, 
            sampling=sampling, 
            crossover=SBX(prob=0.9, eta=15), 
            mutation=PM(prob=0.1, eta=mutation_eta)
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)
        
        # --- 结果优选筛选 ---
        F, X = res.F, res.X
        mask = F[:, 0] <= rmse_limit # 筛选出 RMSE 真正达标的解
        
        if np.any(mask):
            # 在达标解中选 TR 最小的
            best_idx = np.where(mask)[0][np.argmin(F[mask, 1])]
        else:
            # 若没达标，取 RMSE 最小的解作为下步基础
            best_idx = np.argmin(F[:, 0])

        return X[best_idx], X, F

    def run_GA1(self, q_init, q_ratio_bounds=(0.9, 1.1), pop_size=100, n_gen=100, 
                RMSE_weight=0.5, RMSE_tol=0.8, n_jobs=4):
        
        runner = self._get_runner(n_jobs)
        problem = FDMGA1Problem(self.solver, q_init, q_ratio_bounds, self.cable_eids,
                                RMSE_weight, RMSE_tol, elementwise_runner=runner)

        algorithm = GA(
            pop_size=pop_size, 
            sampling=Population.new("X", np.ones((1, self.solver.num_group))),
            crossover=SBX(prob=0.9, eta=25), 
            mutation=PM(eta=45) # GA1 采用更细微的变异步长
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)
        final_q = q_init * res.X if res.X is not None else q_init
        return final_q

    def run_iteration(self, q_init=None, max_iter=1000, rms_limit=1.0):
        """经典 FDM 迭代法，用于生成高质量初始种子"""
        if q_init is None:
            q_init = np.full(self.solver.num_group, 1.0, dtype=float)
        new_q = q_init
        for i in range(max_iter):
            ncoords, tension = self.solver.solve(new_q)
            new_q = self.solver.compute_q(tension)
            rmse = 1.0e3 * np.sqrt(np.mean(np.sum((ncoords - self.solver.ncoord) ** 2, axis=1)))
            if rmse < rms_limit: break
        return new_q

    def _print_status_report(self, label, q):
        coords, tensions = self.solver.solve(q)
        target_t = tensions[self.cable_eids]
        dists = np.linalg.norm(coords - self.solver.ncoord, axis=1) * 1000.0
        rmse = np.sqrt(np.mean(dists**2))
        t_min, t_max = np.min(target_t), np.max(target_t)
        
        print(f"\n" + "=" * 50)
        print(f"   [{label}] 最终物理指标检测")
        print("-" * 50)
        print(f"   - RMSE: {rmse:>10.4f} mm")
        print(f"   - TR:   {t_max/(t_min+1e-8):>10.4f}")
        print(f"   - 极值: [{t_min:.2f} ~ {t_max:.2f}] kN")
        print("=" * 50 + "\n")

 
    def save_seeds(self, q, filename_prefix="seed"):
        """
        计算当前 q 的指标并保存，文件名包含 RMSE 和 TR
        """
        # 1. 计算当前指标
        coords, tensions = self.solver.solve(q)
        target_t = tensions[self.cable_eids]
        
        # RMSE (mm)
        dists = np.linalg.norm(coords - self.solver.ncoord, axis=1) * 1000.0
        rmse = np.sqrt(np.mean(dists**2))
        
        # Tension Ratio
        tr = np.max(target_t) / (np.min(target_t) + 1e-8)
        
        # 2. 构造文件名 (保留两位小数，方便查看)
        # 例如: seed_RMSE_0.85_TR_1.24.npy
        filename = f"{filename_prefix}_RMSE_{rmse:.2f}_TR_{tr:.2f}.npy"
        
        np.save(filename, q)
        print(f"[IO] 种子已保存: {filename}")
        return filename