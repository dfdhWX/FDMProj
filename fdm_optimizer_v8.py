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
from pymoo.core.problem import Problem
# 导入并行化组件
from pymoo.parallelization.joblib import JoblibParallelization
import os

# =================================================================
# 1. GA1 问题定义：单目标优化 (已继承 ElementwiseProblem)
# =================================================================
class FDMGA1Problem(ElementwiseProblem):
    def __init__(self, solver, q_init, q_range_ratio, elset_indices, RMSE_weight, 
                 RMSE_max, TR_max, RMSE_tol=0.8, TR_tol=10, RMSE_penalty=10.0, 
                 TR_penalty=10, t_limits=None, u_limit=None, use_penalty=True, **kwargs):
        
        n_groups = solver.num_group
        super().__init__(n_var=n_groups, n_obj=1, xl=q_range_ratio[0], xu=q_range_ratio[1], **kwargs)

        self.solver = solver
        self.q_init = q_init
        self.cable_elset = elset_indices
        self.RMSE_weight = RMSE_weight
        self.RMSE_max = RMSE_max
        self.TR_max = TR_max
        self.RMSE_tol = RMSE_tol
        self.TR_tol = TR_tol
        self.RMSE_penalty = RMSE_penalty
        self.TR_penalty = TR_penalty
        self.t_limits = t_limits
        self.u_limit = u_limit
        self.use_penalty = use_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        current_q = self.q_init * x
        try:
            new_coords, all_tensions = self.solver.solve(current_q)
            cable_force = all_tensions[self.cable_elset]

            rmse = np.sqrt(np.mean(np.sum((new_coords - self.solver.ncoord) ** 2, axis=1))) * 1000.0
            t_min = np.min(cable_force)
            t_ratio = np.max(cable_force) / (t_min + 1e-8)

            f1 = 0.0 if rmse < self.RMSE_tol else np.clip(rmse / (self.RMSE_max + 1e-8), 0, 1)
            f2 = 0.0 if t_ratio < self.TR_tol else np.clip((t_ratio - 1.0) / (self.TR_max - 1.0 + 1e-8), 0.0, 1.0)

            f = self.RMSE_weight * f1 + (1.0 - self.RMSE_weight) * f2
            penalty = 0.0
            if self.use_penalty:
                if rmse > self.RMSE_max: penalty += self.RMSE_penalty * (rmse - self.RMSE_max) ** 2
                if t_ratio > self.TR_max: penalty += self.TR_penalty * (t_ratio - self.TR_max) ** 2
                if self.t_limits is not None and t_min < self.t_limits[0]:
                    penalty += 5.0 * (self.t_limits[0] - t_min) ** 2

            out["F"] = f + penalty
        except:
            out["F"] = 1e10

# =================================================================
# 2. GA2 问题定义：多目标优化
# =================================================================
class FDMGA2Problem(ElementwiseProblem):
    def __init__(self, solver, q_bounds, cable_eids, u_limit=100.0, 
                 t_limits=(5.0, 500.0), use_penalty=True, **kwargs):
        n_groups = solver.num_group
        super().__init__(n_var=n_groups, n_obj=2, n_constr=0, xl=q_bounds[0], xu=q_bounds[1], **kwargs)
        self.solver = solver
        self.cable_eids = cable_eids
        self.u_limit = u_limit
        self.t_limits = t_limits
        self.use_penalty = use_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            new_coords, tensions = self.solver.solve(x)
            target_t = tensions[self.cable_eids]

            # 1. 计算几何误差 (单位: mm)
            dists = np.linalg.norm(new_coords - self.solver.ncoord, axis=1) * 1000
            f1 = np.sqrt(np.mean(dists**2)) 

            # 2. 计算张力比 (TR)
            t_min, t_max = np.min(target_t), np.max(target_t)
            # 基础目标：张力比
            f2 = t_max / (t_min + 1e-8)

            # 3. 惩罚项分流
            penalty_u = 0.0
            penalty_t = 0.0
            
            if self.use_penalty:
                # 位移超限惩罚 (影响精度目标)
                max_u = np.max(dists)
                if max_u > self.u_limit: 
                    penalty_u += 50.0 * (max_u - self.u_limit) ** 2
                
                # 张力过小惩罚 (只影响张力目标，保住 f1 的竞争力)
                # if t_min < self.t_limits[0]: 
                #     penalty_t += 100.0 * (self.t_limits[0] - t_min) ** 2

            # 【关键修改】：去耦输出
            # F1 (RMSE): 只受位移惩罚影响。即便张力极小，只要几何对得准，它依然是个“准解”
            # F2 (TR): 叠加张力惩罚。引导算法在不破坏形状的前提下通过演化找好张力
            out["F"] = [f1 + penalty_u, f2 + penalty_t]
            
        except Exception:
            out["F"] = [1e10, 1e10]

# =================================================================
# 3. 核心优化器：FDMOptimizer (并行增强版)
# =================================================================
class FDMOptimizer:
    def __init__(self, solver, only_surface=False, surf_elset=None):
        self.solver = solver
        self.cable_eids = list(range(self.solver.num_elemt))
        if only_surface:
            if surf_elset is None: raise ValueError("未传入面索单元索引集")
            self.cable_eids = surf_elset

    def _get_runner(self, n_jobs):
        """初始化并行运行器"""
        if n_jobs <= 1:
            return None
        print(f"[Parallel] 正在启动 {n_jobs} 核并行计算资源...")
        return JoblibParallelization(n_jobs=n_jobs)

    def _print_status_report(self, label, q):
        coords, tensions = self.solver.solve(q)
        target_t = tensions[self.cable_eids]
        dists = np.linalg.norm(coords - self.solver.ncoord, axis=1) * 1000.0
        rmse, max_u = np.sqrt(np.mean(dists**2)), np.max(dists)
        t_min, t_max = np.min(target_t), np.max(target_t)
        
        print(f"\n" + "=" * 50)
        print(f"  [{label}] 运行快报")
        print("-" * 50)
        print(f"  - 几何精度 RMSE:   {rmse:>10.4f} mm")
        print(f"  - 张力极值比 Ratio: {t_max/(t_min+1e-8):>10.4f}")
        print(f"  - 张力范围 Range:   [{t_min:.2f} ~ {t_max:.2f}] kN")
        print(f"  - 最大位移 Umax:    {max_u:>10.4f} mm")
        print("=" * 50 + "\n")

    def run_GA2(self, q_seeds=None, q_bounds=(0.1, 150.0), pop_size=200, n_gen=250, 
                u_limit=100.0, t_limits=(10.0, 500.0), penalty=True, 
                n_jobs=4, plot_pareto=False): # 新增 n_jobs 参数
        
        print(f"\n>> GA2：多目标搜索开始 (核心数: {n_jobs}, 种群: {pop_size})")

        runner = self._get_runner(n_jobs)
        problem = FDMGA2Problem(
            self.solver, q_bounds, self.cable_eids,
            u_limit=u_limit, t_limits=t_limits, use_penalty=penalty,
            elementwise_runner=runner # 注入并行执行器
        )

        if q_seeds is not None:
            sampling = Population.new("X", np.clip(np.atleast_2d(q_seeds), q_bounds[0], q_bounds[1]))
            print(f"   [Status] 已注入 {len(sampling)} 个种子个体")
        else:
            sampling = FloatRandomSampling()

        algorithm = NSGA2(pop_size=pop_size, sampling=sampling, crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20))

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)
        if res.X is None: raise RuntimeError("GA2 未找到可行解集")

        F, X = res.F, res.X
        mask = F[:, 0] < 1.0 # 精度达标筛选
        best_idx = np.where(mask)[0][np.argmin(F[mask, 1])] if np.any(mask) else np.argmin(F[:, 0])

        self._print_status_report("GA2 优选解", X[best_idx])
        if plot_pareto: self.plot_pareto_front(F, best_idx)
        return X[best_idx], X, F

    def run_GA1(self, q_init, q_ratio_bounds=(0.8, 1.2), t_limits=(10.0, 500.0), 
                u_limit=70.0, pop_size=100, n_gen=100, RMSE_weight=0.5, 
                RMSE_max=2.0, RMSE_tol=0.8, TR_max=10.0, TR_tol=1.0, 
                n_jobs=4, penalty=True): # 新增 n_jobs 参数

        print(f"\n>> GA1 精修开始 (核心数: {n_jobs}) | 权重 (RMSE:{RMSE_weight}, Ratio:{1.0-RMSE_weight})")

        runner = self._get_runner(n_jobs)
        problem = FDMGA1Problem(
            self.solver, q_init, q_ratio_bounds, self.cable_eids,
            RMSE_weight, RMSE_max, TR_max, RMSE_tol, TR_tol,
            t_limits=t_limits, u_limit=u_limit, use_penalty=penalty,
            elementwise_runner=runner # 注入并行执行器
        )

        algorithm = GA(pop_size=pop_size, sampling=Population.new("X", np.ones((1, self.solver.num_group))),
                       crossover=SBX(prob=0.9, eta=25), mutation=PM(eta=35))

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)
        final_q = q_init * res.X if res.X is not None else q_init
        self._print_status_report("GA1 阶段完成", final_q)
        return final_q

    def run_iteration(self, q_init=None, max_iter=1000, rms_limit=1.0):
        """标准迭代法"""
        if q_init is None:
            q_init = np.full(self.solver.num_group, 1.0, dtype=float)
        new_q = q_init

        print(f"{'Iter':^8} | {'RMSE (mm)':^15} | {'Tension Ratio':^18}")
        for i in range(max_iter):
            ncoords, tension = self.solver.solve(new_q)
            new_q = self.solver.compute_q(tension)
            rmse = 1.0e3 * np.sqrt(
                np.mean(np.sum((ncoords - self.solver.ncoord) ** 2, axis=1))
            )
            ten_ratio = np.max(tension[self.cable_eids]) / (
                np.min(tension[self.cable_eids]) + 1e-8
            )
            print(f"{i+1:^8} | {rmse:>15.4f} | {ten_ratio:>18.4f}")
            if rmse < rms_limit:
                break
        return new_q
    
    
    def plot_pareto_front(self, F_set, best_idx=None):
        plt.figure(figsize=(8, 5))
        plt.scatter(F_set[:, 0], F_set[:, 1], c="royalblue", s=25, alpha=0.6, label="Pareto Solutions")
        if best_idx is not None:
            plt.scatter(F_set[best_idx, 0], F_set[best_idx, 1], c="crimson", marker="*", s=150, label="Selected Seed")
        plt.xlabel("Obj 1: Geometry RMSE (mm) + Penalty")
        plt.ylabel("Obj 2: Tension Ratio + Penalty")
        plt.title("Optimization Pareto Front Analysis (Penalty Method)")
        plt.grid(True, linestyle="--", alpha=0.5); plt.legend(); plt.show()

    def save_seeds(self, q, filename_prefix="seed"):
        coords, tensions = self.solver.solve(q)
        rmse = np.sqrt(np.mean(np.sum((coords - self.solver.ncoord)**2, axis=1))) * 1000.0
        tr = np.max(tensions[self.cable_eids]) / (np.min(tensions[self.cable_eids]) + 1e-8)
        filename = f"{filename_prefix}_RMSE_{rmse:.2f}_TR_{tr:.2f}.npy"
        np.save(filename, q)
        print(f"[IO] 种子已保存: {filename}")
        return filename
    
    def load_seeds(self, filename="fdm_seeds.npy"):
        """从本地文件加载种子解"""
        if os.path.exists(filename):
            q_seeds = np.load(filename)
            print(f"[IO] 成功从 {filename} 加载种子解，形状为: {q_seeds.shape}")
            return q_seeds
        else:
            print(f"[Warning] 文件 {filename} 不存在，将返回 None")
            return None
