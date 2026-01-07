import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.parallelization.joblib import JoblibParallelization


class FDMGA2Problem(ElementwiseProblem):
    def __init__(self, solver, q_bounds, cable_eids, rmse_max, tr_max, **kwargs):
        # 确保边界是有意义的，防止 xl == xu
        xl = np.array(q_bounds[0])
        xu = np.array(q_bounds[1])
        if np.any(xu - xl < 1e-6):
            xu = xl + 1e-4  # 强制维持一个微小的搜索空间

        super().__init__(
            n_var=solver.num_group, n_obj=2, n_constr=0, xl=xl, xu=xu, **kwargs
        )
        self.solver, self.cable_eids = solver, cable_eids
        self.rmse_max, self.tr_max = rmse_max, tr_max

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            rmse, tr = self.solver.get_RMSE_TR(x, self.cable_eids)
            f1, f2 = rmse / self.rmse_max, (tr - 1.0) / (self.tr_max - 1.0 + 1e-8)

            # 淘汰逻辑：超过基准即重罚
            if f1 > 1.0 or f2 > 1.0:
                f1 += 100.0
                f2 += 100.0
            out["F"] = [f1, f2]
        except:
            out["F"] = [1e5, 1e5]


class FDMGA1Problem(ElementwiseProblem):
    def __init__(
        self,
        solver,
        q_init,
        q_ratio_bounds,
        cable_eids,
        rmse_max,
        tr_max,
        weight_rms,
        alpha,
        use_penalty=True,  # 新增：控制开关
        **kwargs,
    ):
        super().__init__(
            n_var=solver.num_group,
            n_obj=1,
            xl=q_ratio_bounds[0],
            xu=q_ratio_bounds[1],
            **kwargs,
        )
        self.solver, self.q_init, self.cable_eids = solver, q_init, cable_eids
        self.rmse_max, self.tr_max, self.weight_rms = rmse_max, tr_max, weight_rms
        self.alpha = alpha
        self.use_penalty = use_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            current_q = self.q_init * x
            rmse, tr = self.solver.get_RMSE_TR(current_q, self.cable_eids)
            
            # 1. 线性归一化计算
            f1 = rmse / self.rmse_max
            # 修改点：将指数函数换为普通线性比例
            f2 = (tr -1.0) / (self.tr_max - 1.0)

            # 2. 核心逻辑分支
            if self.use_penalty:
                # 开启罚函数模式：设定明确的达标线
                if f1 > 1.0 or f2 > 1.0:
                    # 惩罚项：基础重罚 + 距离惩罚
                    out["F"] = 1e5 + (f1 - 1.0)**2 + (f2 - 1.0)**2
                else:
                    # 达标区：加权求和
                    out["F"] = self.weight_rms * f1 + (1.0 - self.weight_rms) * f2
            else:
                # 关闭罚函数模式：纯多目标加权搜索
                # 此时算法会为了降低 TR 而允许 RMSE 超过限制
                out["F"] = self.weight_rms * f1 + (1.0 - self.weight_rms) * f2

        except:
            # 针对矩阵奇异等数学崩溃的极端个体给予“死刑”
            out["F"] = 1e10


class FDMOptimizer:
    def __init__(self, solver, only_surface=False, surf_elset=None):
        self.solver = solver
        self.cable_eids = surf_elset if only_surface else None

    def run_GA2(
        self,
        q_seeds,
        q_bounds,
        rmse_max,
        tr_max,
        pop_size=160,
        n_gen=100,
        n_jobs=4,
        mutation_eta=30,
    ):
        runner = JoblibParallelization(n_jobs=n_jobs) if n_jobs > 1 else None
        problem = FDMGA2Problem(
            self.solver,
            q_bounds,
            self.cable_eids,
            rmse_max,
            tr_max,
            elementwise_runner=runner,
        )

        seeds = np.atleast_2d(q_seeds)
        # 初始种群生成：在种子周围进行极细微抖动，防止初始解超标太多
        pop_X = np.tile(seeds, (pop_size, 1))
        pop_X[1:] *= 1 + 0.005 * np.random.randn(pop_size - 1, self.solver.num_group)
        sampling = Population.new("X", pop_X)

        # 调整 SBX eta 为 15，PM eta 为 mutation_eta，增加数值稳定性
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.1, eta=mutation_eta),
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=False)

        mask = (res.F[:, 0] <= 1.0) & (res.F[:, 1] <= 1.0)
        if np.any(mask):
            idx = np.argmin(res.F[mask, 1])
            return res.X[mask][idx], res.X, res.F
        return seeds[0], res.X, res.F

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
    
    
    def run_GA1(
        self,
        q_init,
        q_ratio_bounds,
        rmse_max,
        tr_max,
        weight_rms=0.1,
        mutation_eta=50,
        pop_size=128,
        n_gen=200,
        n_jobs=4,
        alpha=25.0,
        use_penalty=False,
    ):
        runner = JoblibParallelization(n_jobs=n_jobs) if n_jobs > 1 else None
        problem = FDMGA1Problem(
            self.solver,
            q_init,
            q_ratio_bounds,
            self.cable_eids,
            rmse_max,
            tr_max,
            weight_rms,
            alpha=alpha,
            elementwise_runner=runner,
            use_penalty=use_penalty,
        )

        sampling = Population.new(
            "X",
            np.ones((pop_size, self.solver.num_group))
            * (1 + 0.001 * np.random.randn(pop_size, self.solver.num_group)),
        )
        # 修改 run_GA1 中的 algorithm 定义
        algorithm = GA(
            pop_size=pop_size,
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=20),
            # 强制 prob 略微调高，eta 不要加得太快
            mutation=PM(prob=0.25, eta=mutation_eta), 
            eliminate_duplicates=True # 必须开启：强行剔除重复个体
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=True)
        return q_init * res.X

    def load_seeds(self, path):
        return np.load(path)

    def save_seeds(self, q, name, root_dir="seeds"):
        rmse, tr = self.solver.get_RMSE_TR(q, self.cable_eids)
        fname = f"{root_dir}/{name}_R{rmse:.5f}_T{tr:.5f}.npy"
        np.save(fname, q)
        print(f"[IO] 种子已保存: {fname}")
        return fname

    def _print_status_report(self, label, q):
        rmse, tr = self.solver.get_RMSE_TR(q, self.cable_eids)
        print(f"[{label}] RMSE: {rmse:.5f} mm | TR: {tr:.5f}")
