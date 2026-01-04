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
# 1. GA1 问题定义：工程级内力均衡 (引入软约束与群体 CV 指标)
# =================================================================
class FDMGA1Problem(ElementwiseProblem):
    def __init__(self, solver, q_init, cable_eids, rmse_target=1.0, 
                 t_min_target=5.0, anneal_range=(0.85, 1.15)):
        n_groups = solver.num_group
        # 自由度退火：限制力密度变动半径，保护结构刚度稳定性
        super().__init__(n_var=n_groups, n_obj=1, xl=anneal_range[0], xu=anneal_range[1])
        self.solver = solver
        self.q_init = q_init
        self.cable_eids = cable_eids
        self.rmse_target = rmse_target
        self.t_min_target = t_min_target

    def _evaluate(self, x, out, *args, **kwargs):
        current_q = self.q_init * x
        new_coords, all_tensions = self.solver.solve(current_q)
        
        # 1. 性能指标
        rmse = np.sqrt(np.mean(np.sum((new_coords - self.solver.ncoord)**2, axis=1))) * 1000.0
        target_t = all_tensions[self.cable_eids]
        t_min, t_max = np.min(target_t), np.max(target_t)
        t_cv = np.std(target_t) / (np.mean(target_t) + 1e-8)
        
        # 2. 卫星天线：极高压精度惩罚 (核心修正)
        # 只要 RMSE > 1.0mm，我们就让 penalty 占据统治地位
        penalty = 0.0
        if rmse > self.rmse_target:
            # 使用平方增长的巨大惩罚，迫使 GA 优先回归 1.0mm 以内
            penalty = 1e6 + (rmse - self.rmse_target)**2 * 50000.0
        
        # 3. 驱动目标
        # 只有在精度达标 (penalty == 0) 的个体之间，才去竞争 CV 和 Ratio
        # 增加 5.0 * rmse 的引导权重，让它在 1mm 内部继续向更准的方向卷
        out["F"] = t_cv * 100.0 + penalty + 5.0 * rmse

    

# =================================================================
# 2. GA2 问题定义：多目标余量探索 (寻找高潜力种子)
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
                penalty += 1e5 * (self.t_min_limit - t_min)

            f1 = rmse + penalty
            f2 = t_ratio + penalty
            
            if rmse > self.rmse_target:
                f2 += 1e7
            else:
                f1 = rmse * 0.1 # 达标区内鼓励高精度

            out["F"] = [f1, f2]
        except:
            out["F"] = [1e10, 1e10]

# =================================================================
# 3. FDMOptimizer 控制类
# =================================================================
class FDMOptimizer:
    def __init__(self, solver, only_surface=True, surf_elset=None):
        self.solver = solver
        self.cable_eids = surf_elset if only_surface else list(range(self.solver.num_elemt))

    def run_GA2(self, q_init=None, q_bounds=(0.1, 200.0), pop_size=200, n_gen=100, rmse_target=1.0, show_plot=False):
        X_init = np.random.uniform(q_bounds[0], q_bounds[1], (pop_size, self.solver.num_group))
        if q_init is not None: X_init[0, :] = q_init

        problem = FDMGA2Problem(self.solver, q_bounds, self.cable_eids, rmse_target=rmse_target)
        algorithm = NSGA2(pop_size=pop_size, sampling=Population.new("X", X_init))
        
        print(f"\n[GA2] 寻找全局构型种子 (目标精度: {rmse_target}mm)...")
        res = minimize(problem, algorithm, ("n_gen", n_gen), verbose=True)

        valid = res.F[:, 0] <= rmse_target
        idx = np.where(valid)[0][np.argmin(res.F[valid, 1])] if any(valid) else np.argmin(res.F[:, 0])
        
        self.plot_pareto_front(res.F, best_idx=idx, rmse_target=rmse_target, show_plot=show_plot)
        self._print_final_status("GA2 种子选定", res.X[idx])
        return res.X[idx]

    def run_GA1(self, q_init, rmse_target=1.0, t_min_target=5.0, n_gen=400, anneal_range=(0.95, 1.05)):
        # 关键修正：将退火半径收窄到 (0.95, 1.05)
        # 只有限制了力密度的波动范围，坐标才不会“乱跳”
        print(f"\n[GA1 卫星精修] 搜索半径收缩至 5% | 目标精度: {rmse_target}mm")
        
        problem = FDMGA1Problem(self.solver, q_init, self.cable_eids, rmse_target, t_min_target, anneal_range)
        
        algorithm = GA(
            pop_size=150, 
            sampling=Population.new("X", np.random.uniform(anneal_range[0], anneal_range[1], (150, self.solver.num_group))),
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob=0.2, eta=25), # 提高 eta 值，使变异更倾向于“微调”
            eliminate_duplicates=True
        )
        
        res = minimize(problem, algorithm, ("n_gen", n_gen), verbose=True)
        final_q = q_init * res.X
        self._print_final_status("GA1 最终解", final_q)
        return final_q

    def plot_pareto_front(self, F_set, best_idx=None, rmse_target=1.0, show_plot=False):
        plt.figure(figsize=(8, 5))
        mask = F_set[:, 1] < 1e6
        plt.scatter(F_set[mask, 0], F_set[mask, 1], c="royalblue", alpha=0.5, label="Candidates")
        plt.axvline(x=rmse_target, color="red", linestyle="--", label="Target RMSE")
        if best_idx is not None and F_set[best_idx, 1] < 1e6:
            plt.scatter(F_set[best_idx, 0], F_set[best_idx, 1], c="gold", marker="*", s=200, label="Selected")
        plt.xlabel("RMSE (mm)"); plt.ylabel("Tension Ratio"); plt.legend(); plt.grid(alpha=0.3)
        if not os.path.exists("plots"): os.makedirs("plots")
        plt.savefig(f"plots/Satellite_PF_{int(time.time())}.png")
        if show_plot: plt.show()
        else: plt.close()

    def _print_final_status(self, label, q):
        coords, tensions = self.solver.solve(q)
        target_t = tensions[self.cable_eids]
        rmse = np.sqrt(np.mean(np.sum((coords - self.solver.ncoord) ** 2, axis=1))) * 1000.0
        t_max, t_min = np.max(target_t), np.min(target_t)
        t_cv = np.std(target_t) / (np.mean(target_t) + 1e-8)
        print(f"\n >>> {label}:")
        print(f"     RMSE: {rmse:.4f}mm | Ratio: {t_max/t_min:.4f} | CV: {t_cv:.4f}")
        print(f"     Range: [{t_min:.2f} ~ {t_max:.2f}] kN")