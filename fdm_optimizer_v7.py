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
import os


# =================================================================
# 1. GA1 问题定义：单目标优化 (侧重于在特定范围内精修张力均匀度)
# =================================================================
class FDMGA1Problem(ElementwiseProblem):
    def __init__(
        self,
        solver,
        q_init,
        q_range_ratio,
        elset_indices,
        RMSE_weight,  # (w_rmse, w_ratio)
        RMSE_max,  # 用于归一化的 RMSE 范围 [min, max]
        TR_max,  # TR 最大范围
        RMSE_tol=0.8,  # RMSE 小于这个值认为可以接受
        TR_tol=10,
        RMSE_penalty=10.0,
        TR_penalty=10,
        t_limits=None,
        u_limit=None,
        use_penalty=True,
    ):

        n_groups = solver.num_group
        super().__init__(
            n_var=n_groups, n_obj=1, xl=q_range_ratio[0], xu=q_range_ratio[1]
        )

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

            # 1. 原始指标计算
            rmse = (
                np.sqrt(np.mean(np.sum((new_coords - self.solver.ncoord) ** 2, axis=1)))
                * 1000.0
            )
            t_min = np.min(cable_force)
            t_ratio = np.max(cable_force) / (t_min + 1e-8)

            # ========= RMSE ==========
            if rmse < self.RMSE_tol:
                f1 = 0.0
            else:
                # 2. 归一化评价 (你的公式实现)
                # f1: RMSE 归一化
                f1 = rmse / (self.RMSE_max + 1e-8)
                f1 = np.clip(f1, 0, 1)  # 限制在 0-1 之间

            # ======== TR ============
            if t_ratio < self.TR_tol:
                f2 = 0.0
            else:
                # f2: Tension Ratio 归一化 (1 - 1/ratio)
                f2 = (t_ratio - 1.0) / (self.TR_max - 1.0 + 1e-8)
                f2 = np.clip(f2, 0.0, 1.0)

            # 总目标：加权求和
            f = self.RMSE_weight * f1 + (1.0 - self.RMSE_weight) * f2

            # 3. 惩罚项 (保持物理边界)
            penalty = 0.0
            if self.use_penalty:
                # 依然保留 RMSE > 阈值时的强力惩罚，确保不会跑偏太远
                if rmse > self.RMSE_max:
                    penalty += self.RMSE_penalty * (rmse - self.RMSE_max) ** 2

                if t_ratio > self.TR_max:
                    penalty += self.TR_penalty * (t_ratio - self.TR_max) ** 2

                # 张力下限惩罚
                if self.t_limits is not None and t_min < self.t_limits[0]:
                    penalty += 5.0 * (self.t_limits[0] - t_min) ** 2

            out["F"] = f + penalty
        except Exception:
            out["F"] = 1e10


# =================================================================
# 2. GA2 问题定义：多目标优化 (RMSE vs Tension Ratio)
# =================================================================
class FDMGA2Problem(ElementwiseProblem):
    def __init__(
        self,
        solver,
        q_bounds,
        cable_eids,
        u_limit=100.0,
        t_limits=(5.0, 500.0),
        use_penalty=True,
    ):
        n_groups = solver.num_group
        super().__init__(
            n_var=n_groups, n_obj=2, n_constr=0, xl=q_bounds[0], xu=q_bounds[1]
        )
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
                    penalty += 50.0 * (max_u - self.u_limit) ** 2

                # 张力下限惩罚
                if t_min < self.t_limits[0]:
                    penalty += 200.0 * (self.t_limits[0] - t_min) ** 2

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

        print(f"\n" + "=" * 50)
        print(f"  [{label}] 运行快报")
        print("-" * 50)
        print(f"  - 几何精度 RMSE:   {rmse:>10.4f} mm")
        print(f"  - 张力极值比 Ratio: {t_ratio:>10.4f}")
        print(f"  - 张力范围 Range:   [{t_min:.2f} ~ {t_max:.2f}] kN")
        print(f"  - 最大位移 Umax:    {max_u:>10.4f} mm")
        print("=" * 50 + "\n")

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

    def run_GA2(
        self,
        q_seeds=None,
        q_bounds=(0.1, 150.0),
        pop_size=200,
        n_gen=250,
        u_limit=100.0,
        t_limits=(10.0, 500.0),
        penalty=True,
        plot_pareto=True,
    ):
        """
        多目标 NSGA-II 搜索 (带种子解的热启动版本)

        参数:
        q_seeds: 初始力密度分布。可以是单组解 (n_var,) 或多组解 (n_seeds, n_var)。
        """
        print(f"\n>> GA2：多目标搜索开始 (种群: {pop_size}, 代数: {n_gen})")

        # 1. 定义问题
        problem = FDMGA2Problem(
            self.solver,
            q_bounds,
            self.cable_eids,
            u_limit=u_limit,
            t_limits=t_limits,
            use_penalty=penalty,
        )

        # 2. 处理初始采样策略 (Warm Start)
        if q_seeds is not None:
            q_seeds = np.atleast_2d(q_seeds)
            # 检查种子解是否超出边界，若超出则强行截断到边界内，防止 Pymoo 报错
            q_seeds = np.clip(q_seeds, q_bounds[0], q_bounds[1])

            # 使用种子解初始化种群，不足 pop_size 的部分会自动随机补全
            sampling = Population.new("X", q_seeds)
            print(f"   [Status] 已注入 {len(q_seeds)} 个种子个体进入初始种群")
        else:
            sampling = FloatRandomSampling()

        # 3. 配置 NSGA-II 算法
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        # 4. 执行优化
        res = minimize(
            problem,
            algorithm,
            ("n_gen", n_gen),
            seed=42,
            verbose=True,
            save_history=False,
        )

        if res.X is None:
            raise RuntimeError("GA2 未找到可行解集，请检查边界条件或约束设置")

        # 5. 结果筛选策略：优先选择精度达标 (RMSE < 1.0mm) 且张力比最小的解
        F = res.F
        X = res.X

        # 筛选 RMSE < 1.0mm 的索引
        rmse_threshold = 1.0
        mask = F[:, 0] < rmse_threshold

        if np.any(mask):
            # 在满足精度的解中找张力比 (F[:,1]) 最小的
            valid_indices = np.where(mask)[0]
            best_idx = valid_indices[np.argmin(F[valid_indices, 1])]
            print(
                f">> 决策策略：从精度达标 ({rmse_threshold}mm) 的解集中选择了张力最均匀的解"
            )
        else:
            # 如果都没有达标，则选择 RMSE 最小的解
            best_idx = np.argmin(F[:, 0])
            print(f">> 决策策略：未发现精度达标解，已选择当前 RMSE 最小解")

        # 打印最终报告
        self._print_status_report("GA2 优选解", X[best_idx])

        # 6. 绘制帕累托前沿
        if plot_pareto:
            self.plot_pareto_front(F, best_idx)

        return X[best_idx], X, F

    def run_GA1(
        self,
        q_init,
        q_ratio_bounds=(0.8, 1.2),
        t_limits=(10.0, 500.0),
        u_limit=70.0,
        pop_size=100,
        n_gen=100,
        RMSE_weight=0.5,
        RMSE_max=2.0,
        RMSE_tol=0.8,
        TR_max=10.0,
        TR_tol=1.0,
        penalty=True,
    ):

        print(f"\n>> GA1 精修开始 | 权重 (RMSE:{RMSE_weight}, Ratio:{1.0-RMSE_weight})")

        # Fixed the argument names to match FDMGA1Problem.__init__
        problem = FDMGA1Problem(
            solver=self.solver,
            q_init=q_init,
            q_range_ratio=q_ratio_bounds,
            elset_indices=self.cable_eids,
            RMSE_weight=RMSE_weight,
            RMSE_max=RMSE_max,
            RMSE_tol=RMSE_tol,
            TR_tol=TR_tol,
            TR_max=TR_max,
            t_limits=t_limits,
            u_limit=u_limit,
            use_penalty=penalty,
        )

        # Sampling with the current best (1.0 ratio) to help GA converge faster
        pop_init = Population.new("X", np.ones((1, self.solver.num_group)))

        algorithm = GA(
            pop_size=pop_size,
            sampling=pop_init,
            crossover=SBX(prob=0.9, eta=25),
            mutation=PM(eta=35),
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)

        if res.X is None:
            return q_init

        final_q = q_init * res.X
        self._print_status_report("GA1 阶段完成", final_q)

        return final_q

    def plot_pareto_front(self, F_set, best_idx=None):
        plt.figure(figsize=(8, 5))
        plt.scatter(
            F_set[:, 0],
            F_set[:, 1],
            c="royalblue",
            s=25,
            alpha=0.6,
            label="Pareto Solutions",
        )
        if best_idx is not None:
            plt.scatter(
                F_set[best_idx, 0],
                F_set[best_idx, 1],
                c="crimson",
                marker="*",
                s=150,
                label="Selected Seed",
            )
        plt.xlabel("Geometry RMSE (mm)")
        plt.ylabel("Tension Ratio (Max/Min)")
        plt.title("GA2 Pareto Front Analysis")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()
        
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


    def load_seeds(self, filename="fdm_seeds.npy"):
        """从本地文件加载种子解"""
        if os.path.exists(filename):
            q_seeds = np.load(filename)
            print(f"[IO] 成功从 {filename} 加载种子解，形状为: {q_seeds.shape}")
            return q_seeds
        else:
            print(f"[Warning] 文件 {filename} 不存在，将返回 None")
            return None
