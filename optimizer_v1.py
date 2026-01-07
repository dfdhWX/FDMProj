import numpy as np
import os
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.parallelization.joblib import JoblibParallelization
from pymoo.core.callback import Callback

class TrainingHistory(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data = []

    def notify(self, algorithm):
        # 1. 获取目标函数值 F
        f_vals = algorithm.pop.get("F")
        if f_vals is None or len(f_vals) == 0:
            return
            
        # 确保 f_vals 是一维的以进行 argmin
        f_flat = f_vals.flatten()
        idx_best = np.argmin(f_flat)
        
        # 2. 定义安全提取辅助函数
        def get_safe(key):
            arr = algorithm.pop.get(key)
            if arr is not None and len(arr) > idx_best:
                val = arr[idx_best]
                # 处理可能返回的 [value] 数组或直接的标量
                return val[0] if isinstance(val, (np.ndarray, list)) else val
            return 1e10 # 缺失数据的默认惩罚值

        # 3. 提取指标
        rmse_best = get_safe("RMSE")
        tr_best = get_safe("TR")
        f1_best = get_safe("f1")
        f2_best = get_safe("f2")
        
        # 4. 记录：[代数, F_min, F_avg, RMSE, TR, f1, f2]
        self.data.append([
            algorithm.n_gen, 
            f_flat.min(), 
            f_flat.mean(), 
            rmse_best, 
            tr_best,
            f1_best,
            f2_best
        ])
        

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
        q_ratio_bounds, # 现在接收 tuple (xl_array, xu_array)
        cable_eids,
        rmse_max,
        tr_max,
        weight_rms,
        alpha,
        use_penalty,
        **kwargs,
    ):
        # 允许 xl, xu 是数组，实现非对称物理走廊搜索
        xl = np.array(q_ratio_bounds[0])
        xu = np.array(q_ratio_bounds[1])
        
        super().__init__(
            n_var=solver.num_group,
            n_obj=1,
            xl=xl,
            xu=xu,
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
            
            # 1. 归一化计算
            f1 = rmse / self.rmse_max
            f2 = (tr - 1.0) / (self.tr_max - 1.0 + 1e-8)
            
            # 将原始物理指标存入 Population
            out["RMSE"] = rmse
            out["TR"] = tr
            out["f1"] = f1
            out["f2"] = f2

            # 2. 核心逻辑分支
            if self.use_penalty:
                # 开启罚函数模式
                if f1 > 1.0 or f2 > 1.0:
                    out["F"] = 1e5 + (f1 - 1.0)**2 + (f2 - 1.0)**2
                else:
                    # 采用加权逻辑，也可在此处切换回指数奖励
                    # out["F"] = self.weight_rms * f1 + (1.0 - self.weight_rms) * f2
                    # 保持指数型凹函数归一化 (方案 3) 的选择权利
                    val = self.weight_rms * f1 + (1.0 - self.weight_rms) * f2
                    out["F"] = (np.exp(self.alpha * val) - 1.0) / (np.exp(self.alpha) - 1.0)
            else:
                out["F"] = self.weight_rms * f1 + (1.0 - self.weight_rms) * f2

        except:
            out["F"] = 1e10
            out["RMSE"] = 1e10
            out["TR"] = 1e10
            out["f1"] = 1e10
            out["f2"] = 1e10


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
        pop_X = np.tile(seeds, (pop_size, 1))
        pop_X[1:] *= 1 + 0.005 * np.random.randn(pop_size - 1, self.solver.num_group)
        sampling = Population.new("X", pop_X)

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
            # 检查 NaN
            if np.isnan(rmse) or np.isnan(ten_ratio):
                raise ValueError("Numerical Instability")
        return new_q
    
    def run_GA1(
        self,
        q_init,
        q_ratio_bounds, # 支持数组 (xl_array, xu_array)
        rmse_max,
        tr_max,
        weight_rms=0.1,
        mutation_eta=50,
        pop_size=128,
        n_gen=200,
        n_jobs=4,
        alpha=25.0,
        use_penalty=False,
        initial_seeds=None, # 新增：支持传入插值精英种子
        **kwargs,
    ):
        runner = JoblibParallelization(n_jobs=n_jobs) if n_jobs > 1 else None
        
        # 确保 xl, xu 是数组
        xl = np.array(q_ratio_bounds[0])
        xu = np.array(q_ratio_bounds[1])

        problem = FDMGA1Problem(
            self.solver,
            q_init,
            (xl, xu),
            self.cable_eids,
            rmse_max,
            tr_max,
            weight_rms,
            alpha=alpha,
            elementwise_runner=runner,
            use_penalty=use_penalty,
        )

        # 核心修改：两极导航采样逻辑
        if initial_seeds is not None:
            # 种子注入（例如 qR 到 qT 的线性插值解）
            X_init = np.array(initial_seeds)
            n_remain = pop_size - len(X_init)
            if n_remain > 0:
                # 剩余部分在物理走廊内均匀随机分布
                X_remain = xl + (xu - xl) * np.random.rand(n_remain, self.solver.num_group)
                X_init = np.vstack([X_init, X_remain])
        else:
            # 默认：保留基准 1.0，并辅以扰动和走廊采样
            X_init = np.ones((pop_size, self.solver.num_group))
            # 80% 走廊随机搜索
            random_idx = np.arange(int(pop_size * 0.2), pop_size)
            X_init[random_idx] = xl + (xu - xl) * np.random.rand(len(random_idx), self.solver.num_group)
            # 其余保持 1.0 近邻微调
            X_init[:int(pop_size * 0.2)] *= (1 + 0.001 * np.random.randn(int(pop_size * 0.2), self.solver.num_group))

        sampling = Population.new("X", X_init)

        algorithm = GA(
            pop_size=pop_size,
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob=0.25, eta=mutation_eta), 
            eliminate_duplicates=True,
            save_history=False,
        )
        # 1. 初始化回调对象
        hist_callback = TrainingHistory()

        # 2. 执行优化 (关闭 save_history，增加 callback)
        res = minimize(
            problem, 
            algorithm, 
            ("n_gen", kwargs.get('n_gen', n_gen)), 
            seed=42, 
            verbose=True,
            callback=hist_callback, # 使用回调
            save_history=False      # 关闭报错根源
        )

        raw_data = np.array(hist_callback.data)

        # 定义表头和数据类型
        dt = np.dtype([
            ('gen', 'i4'), 
            ('f_min', 'f8'), 
            ('f_avg', 'f8'), 
            ('rmse', 'f8'), 
            ('tr', 'f8'), 
            ('f1', 'f8'), 
            ('f2', 'f8')
        ])

        # 创建结构化数组
        history_structured = np.empty(len(raw_data), dtype=dt)
        history_structured['gen'] = raw_data[:, 0]
        history_structured['f_min'] = raw_data[:, 1]
        history_structured['f_avg'] = raw_data[:, 2]
        history_structured['rmse'] = raw_data[:, 3]
        history_structured['tr'] = raw_data[:, 4]
        history_structured['f1'] = raw_data[:, 5]
        history_structured['f2'] = raw_data[:, 6]


        # 安全检查
        if res.X is None:
            return q_init, history_structured
            
        return q_init * res.X, history_structured

    def load_seeds(self, path):
        return np.load(path)

    def save_seeds(self, q, name, root_dir="seeds"):
        if not os.path.exists(root_dir): os.makedirs(root_dir)
        rmse, tr = self.solver.get_RMSE_TR(q, self.cable_eids)
        fname = f"{root_dir}/{name}_R{rmse:.5f}_T{tr:.5f}.npy"
        np.save(fname, q)
        print(f"[IO] 种子已保存: {fname}")
        return fname

    def _print_status_report(self, label, q):
        rmse, tr = self.solver.get_RMSE_TR(q, self.cable_eids)
        print(f"[{label}] RMSE: {rmse:.5f} mm | TR: {tr:.5f}")