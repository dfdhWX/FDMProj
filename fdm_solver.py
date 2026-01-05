import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class FDMSolver:
    def __init__(self, ncoord, conn, bcs, elsets):
        self.ncoord = ncoord
        self.conn = conn
        self.bcs = bcs
        self.elsets = elsets
        self.num_elemt = self.conn.shape[0]
        self.num_node = self.ncoord.shape[0]
        self.num_group = len(elsets)
        
        self.__init_matrix()

    def __init_matrix(self):
        # 使用稀疏矩阵构造拓扑矩阵 C
        rows = np.repeat(np.arange(self.num_elemt), 2)
        cols = self.conn.flatten()
        data = np.tile([1.0, -1.0], self.num_elemt)
        self.C = sparse.csr_matrix((data, (rows, cols)), shape=(self.num_elemt, self.num_node))

        # 边界条件处理 (保持不变)
        all_nodes = set(range(self.num_node))
        self.nID_c = sorted(list(set(sum(self.bcs.values(), []))))
        self.nID_f = sorted(list(all_nodes - set(self.nID_c)))

    def solve(self, q_v, p_ext=None):
        # 1. 构造全量 q (稀疏对角阵)
        q = np.zeros(self.num_elemt)
        if len(q_v) == self.num_group:
            for i, idxs in enumerate(self.elsets):
                q[idxs] = q_v[i]
        else:
            q = q_v
        
        Q = sparse.diags(q)
        
        # 2. 预计算核心矩阵
        # A = Cf.T @ Q @ Cf
        Cf = self.C[:, self.nID_f]
        Cc = self.C[:, self.nID_c]
        
        # 稀疏矩阵相乘
        A = Cf.T @ Q @ Cf
        
        new_coords = self.ncoord.copy()
        for i in range(3): # X, Y, Z
            xc = self.ncoord[self.nID_c, i]
            # 右侧项: pf - Cf.T @ Q @ Cc @ xc
            b = - Cf.T @ Q @ (Cc @ xc)
            if p_ext is not None:
                b += p_ext[self.nID_f, i]
            
            # 使用稀疏矩阵求解器
            new_coords[self.nID_f, i] = spsolve(A, b)

        # 3. 计算张力
        diffs = self.C @ new_coords
        lengths = np.linalg.norm(diffs, axis=1)
        tensions = q * lengths
        
        return new_coords, tensions
    
    
    def compute_q(self, current_tensions):
        """
        标准迭代法核心逻辑：q_new = T_current / L_target
        旨在将当前受力状态下的张力，映射到目标几何位置上
        """
        q_groups = np.zeros(self.num_group)

        # 1. 计算目标坐标（即初始输入的 ncoord）下的全量单元长度
        # diffs_ideal 形状为 (num_elemt, 3)
        diffs_ideal = self.C @ self.ncoord 
        # lengths_ideal 形状为 (num_elemt,)
        lengths_ideal = np.linalg.norm(diffs_ideal, axis=1)

        # 2. 按组统计新的力密度
        for i, element_indices in enumerate(self.elsets):
            # 提取当前求解出来的张力
            group_t = current_tensions[element_indices]
            # 提取目标坐标下的理想长度
            group_l_ideal = lengths_ideal[element_indices]
            
            # 计算理想力密度配比 q = T / L
            # 加入 1e-9 防止长度为 0 的异常（如重合点）
            group_qs = group_t / (group_l_ideal + 1e-9)
            
            # 取该组内所有单元力密度的均值作为该组下一轮的输入 q
            q_groups[i] = np.mean(group_qs)
            
        return q_groups