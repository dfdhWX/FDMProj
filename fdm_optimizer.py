import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

class FDMGA1Problem(ElementwiseProblem):
    def __init__(self, solver, q_init, q_range_ratio, elset_indices, weight, 
                 f1_range, f2_range, t_limits=None, u_limit=None, use_penalty=True):
        n_groups = solver.num_group
        super().__init__(n_var=n_groups, n_obj=1, xl=q_range_ratio[0], xu=q_range_ratio[1])

        self.solver = solver
        self.q_init = q_init
        self.cable_elset = elset_indices
        self.alpha = weight
        self.f1_max = f1_range[1]
        self.f2_max = f2_range[1]
        self.t_limits = t_limits
        self.u_limit = u_limit
        self.use_penalty = use_penalty # 新增开关

    def _evaluate(self, x, out, *args, **kwargs):
        current_q = self.q_init * x

        # 1. 物理求解
        new_coords, all_tensions = self.solver.solve(current_q)
        cable_force = all_tensions[self.cable_elset]

        # 2. 几何偏差计算 (mm)
        node_dists = np.linalg.norm(new_coords - self.solver.ncoord, axis=1) * 1.0e3
        rmse_raw = np.sqrt(np.mean(node_dists**2))
        max_u = np.max(node_dists)

        # 3. 指标归一化
        f1_norm = np.clip(rmse_raw / min(self.f1_max, 200.0), 0, 1)
        t_cv_raw = np.std(cable_force) / (np.mean(cable_force) + 1e-8)
        f2_norm = np.clip(t_cv_raw / (self.f2_max + 1e-8), 0, 1)

        # 4. 可选惩罚项
        penalty = 0.0
        if self.use_penalty:
            # 位移惩罚
            if self.u_limit is not None:
                u_threshold = self.u_limit[1] if isinstance(self.u_limit, (list, tuple)) else self.u_limit
                if max_u > u_threshold:
                    penalty += 100.0 * (max_u / u_threshold) ** 2

            # 张力惩罚
            if self.t_limits is not None:
                t_min, _ = self.t_limits
                actual_min = np.min(cable_force)
                if actual_min < t_min:
                    penalty += 50.0 * (t_min / (actual_min + 1e-8)) ** 2

        out["F"] = (self.alpha * f1_norm + (1.0 - self.alpha) * f2_norm) + penalty
        
        

class FDMGA2Problem(ElementwiseProblem):
    def __init__(self, solver, q_bounds, cable_eids, u_limit=100.0, t_limits=(5.0, 500.0), use_penalty=True):
        """
        GA2 多目标问题定义
        :param n_obj: 2 (目标1: RMS误差, 目标2: 张力均匀度/CV)
        """
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
            # 1. 物理求解
            new_coords, tensions = self.solver.solve(x)
            target_t = tensions[self.cable_eids]
            
            # 2. 计算目标 1: 几何 RMSE (mm)
            dists = np.linalg.norm(new_coords - self.solver.ncoord, axis=1) * 1000
            f1 = np.sqrt(np.mean(dists**2))
            
            # 3. 计算目标 2: 张力变异系数 (CV) - 比极值比更易于收敛
            f2 = np.std(target_t) / (np.mean(target_t) + 1e-8)
            
            # 4. 惩罚逻辑
            penalty = 0.0
            if self.use_penalty:
                # 位移超限惩罚
                max_u = np.max(dists)
                if max_u > self.u_limit:
                    penalty += 50.0 * (max_u - self.u_limit)**2
                
                # 张力下限惩罚 (防松弛)
                t_min = np.min(target_t)
                if t_min < self.t_limits[0]:
                    penalty += 200.0 * (self.t_limits[0] - t_min)**2

            # 将惩罚直接累加到两个目标函数上，迫使解向可行域移动
            out["F"] = [f1 + penalty, f2 + penalty]
            
        except Exception:
            # 针对奇异矩阵等数值灾难的极端惩罚
            out["F"] = [1e10, 1e10]
            
            

class FDMOptimizer:
    def __init__(self, solver, only_surface=False, surf_elset=None):
        self.solver = solver
        ## ====判断是否需要考虑支撑索=============
        self.cable_eids = list(range(self.solver.num_elemt))
        if only_surface:
            if surf_elset is None:
                raise ValueError("未传入面索单元索引集")
            self.cable_eids = surf_elset

    def run_iteration(self, q_init = None, max_iter = 1000, rms_limit = 1.0):
        """迭代法求解"""
        if q_init is None:
            q_init = np.full(self.solver.num_group, 1, dtype=float)
            
        new_q = q_init
        
        # 打印表头
        header = f"{'Iter':^8} | {'RMSE (mm)':^15} | {'Tension Ratio':^18}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        ##========迭代求解================
        for iter in range(max_iter):
            #==========求解======
            ncoords, tension = self.solver.solve(new_q)
            # 计算新力密度
            new_q = self.solver.compute_q(tension, ncoord=None)
            
            #=======计算型面精度和张力均匀性==========
            rmse = 1.0e3*np.sqrt(np.mean(np.sum((ncoords - self.solver.ncoord) ** 2, axis=1)))
            ten_ratio = np.max(tension)/(np.min(tension)+1e-8)
            
            #====== 输出求解信息========
            # 输出当前行
            print(f"{iter+1:^8} | {rmse:>15.4f} | {ten_ratio:>18.4f}")
            
            #======= 判断===========
            if rmse < rms_limit:
                break
        
        # =========输出最终值========
        return new_q
    
    
    def run_GA1(self, q_init=None, q_ratio_bounds=(0.7, 1.3), weight=0.8,
                t_limits=(5.0, 500.0), u_limit=50.0, pop_size=100, n_gen=200, penalty=True):
        """
        GA1: 在初始力密度基础上通过比例因子进行精细寻优
        """
        if q_init is None:
            print(">> 警告: 未提供 q_init，使用默认全 1.0。")
            q_init = np.ones(self.solver.num_group)

        # --- 阶段 1: 边界探测 (获取归一化基准) ---
        print("\n>> 阶段 1: 探测局部范围内的指标范围...")
        sample_size = 100
        random_ratios = np.random.uniform(q_ratio_bounds[0], q_ratio_bounds[1], (sample_size, self.solver.num_group))
        f1_samples, f2_samples = [], []
        
        for r in random_ratios:
            try:
                c, t = self.solver.solve(q_init * r)
                target_t = t[self.cable_eids]
                f1_samples.append(np.sqrt(np.mean(np.sum((c - self.solver.ncoord)**2, axis=1))) * 1e3)
                f2_samples.append(np.std(target_t) / (np.mean(target_t) + 1e-8))
            except: continue

        f1_range = (min(f1_samples), max(f1_samples))
        f2_range = (min(f2_samples), max(f2_samples))
        print(f"   探测完成: RMSE峰值 {f1_range[1]:.1f}mm, CV峰值 {f2_range[1]:.3f}")

        # --- 阶段 2: 启动优化 ---
        print(f">> 阶段 2: 执行 GA1 (Alpha={weight}, Penalty={penalty})...")
        problem = FDMGA1Problem(self.solver, q_init, q_ratio_bounds, self.cable_eids,
                                weight, f1_range, f2_range, t_limits, u_limit, use_penalty=penalty)

        # 种子注入：确保初始解就在第一代种群中
        from pymoo.core.population import Population
        pop_init = Population.new("X", np.ones((1, self.solver.num_group)))

        algorithm = GA(
            pop_size=pop_size,
            sampling=pop_init,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)

        # --- 结果解析与报告 ---
        if res.X is None: return q_init
        
        final_q = q_init * res.X
        final_coords, final_t = self.solver.solve(final_q)
        target_t = final_t[self.cable_eids]
        
        print("\n" + "=" * 45)
        print("           GA1 优化最终报告")
        print("-" * 45)
        print(f" 几何精度 (RMS):     {np.sqrt(np.mean(np.sum((final_coords-self.solver.ncoord)**2, axis=1)))*1e3:>10.4f} mm")
        print(f" 张力极值比:         {np.max(target_t)/(np.min(target_t)+1e-8):>10.4f}")
        print(f" 张力范围 (kN):      [{np.min(target_t):.2f}, {np.max(target_t):.2f}]")
        print("=" * 45 + "\n")

        return final_q
    

    def run_GA2(self, q_bounds=(0.1, 150.0), pop_size=100, n_gen=300, 
                u_limit=100.0, t_limits=(5.0, 500.0), penalty=True, plot_pareto=True):
        """
        GA2: 基于 NSGA-II 的多目标优化，自动识别前沿边界解
        """
        print(f">> 启动 GA2 优化 (NSGA-II, Penalty={penalty})...")
        
        problem = FDMGA2Problem(
            self.solver, 
            q_bounds, 
            self.cable_eids, 
            u_limit=u_limit, 
            t_limits=t_limits, 
            use_penalty=penalty
        )

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)

        if res.X is None or len(res.F) == 0:
            raise ValueError("GA2 未能找到有效解，请检查约束设置。")

        # --- 帕累托前沿分析 ---
        # F[:, 0] 是 RMSE, F[:, 1] 是 Tension CV (或 Ratio)
        idx_best_rmse = np.argmin(res.F[:, 0])
        idx_best_tension = np.argmin(res.F[:, 1])
        idx_selected = self._get_elbow_point_idx(res.F)

        def get_detailed_info(idx):
            q_vals = res.X[idx]
            coords, tensions = self.solver.solve(q_vals)
            target_t = tensions[self.cable_eids]
            max_u = np.max(np.linalg.norm(coords - self.solver.ncoord, axis=1)) * 1e3
            t_min, t_max = np.min(target_t), np.max(target_t)
            t_ratio = t_max / (t_min + 1e-8)
            return res.F[idx, 0], max_u, t_ratio, t_min, t_max

        # 获取三个关键解的数据
        data_rmse = get_detailed_info(idx_best_rmse)
        data_tension = get_detailed_info(idx_best_tension)
        data_selected = get_detailed_info(idx_selected)

        # =========打印对比报告====================
        print("\n" + "="*60)
        print(f"{'GA2 多目标优化对比报告':^60}")
        print("-"*60)
        print(f"{'指标':<15} | {'最佳精度解':^12} | {'最佳张力解':^12} | {'所选折中解':^12}")
        print("-"*60)
        print(f"{'RMSE (mm)':<15} | {data_rmse[0]:>12.4f} | {data_tension[0]:>12.4f} | {data_selected[0]:>12.4f}")
        print(f"{'Max Dist (mm)':<15} | {data_rmse[1]:>12.4f} | {data_tension[1]:>12.4f} | {data_selected[1]:>12.4f}")
        print(f"{'Tension Ratio':<15} | {data_rmse[2]:>12.4f} | {data_tension[2]:>12.4f} | {data_selected[2]:>12.4f}")
        print(f"{'T Range (kN)':<15} | {f'{data_rmse[3]:.1f}-{data_rmse[4]:.1f}':^12} | {f'{data_tension[3]:.1f}-{data_tension[4]:.1f}':^12} | {f'{data_selected[3]:.1f}-{data_selected[4]:.1f}':^12}")
        print("="*60 + "\n")

        if plot_pareto:
            self.plot_pareto_front(res.F, idx_selected)

        return res.X[idx_selected], res.X, res.F
    
    

    def _get_elbow_point_idx(self, F_set):
        f_min, f_max = F_set.min(axis=0), F_set.max(axis=0)
        F_norm = (F_set - f_min) / (f_max - f_min + 1e-8)
        dist = np.linalg.norm(F_norm, axis=1)
        return np.argmin(dist)

    def plot_pareto_front(self, F_set, best_idx=None):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            F_set[:, 0], F_set[:, 1], c="magenta", s=35, alpha=0.7, label="Pareto Front"
        )
        if best_idx is not None:
            plt.scatter(
                F_set[best_idx, 0],
                F_set[best_idx, 1],
                c="black",
                marker="*",
                s=200,
                label="Selected Best",
            )

        plt.title("Pareto Front: Global RMS vs. Tension Ratio", fontsize=14)
        plt.xlabel("Global Geometry RMS Error", fontsize=12)
        plt.ylabel("Tension Max/Min Ratio", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()
