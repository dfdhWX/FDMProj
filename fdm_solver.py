import numpy as np

class FDMSolver:
    def __init__(self, ncoord, conn, bcs, elsets):
        """力密度法求解"""
        self.ncoord = ncoord
        self.conn = conn
        self.bcs = bcs
        self.elsets = elsets
        # 网格信息
        self.num_elemt = self.conn.shape[0] # 单元总数
        self.num_node = self.ncoord.shape[0]
        ## 拓扑矩阵
        self.C = None
        ## 约束节点和自由节点编号
        # 自由节点节点
        self.nID_xf = None
        self.nID_yf = None
        self.nID_zf = None
        # 约束节点
        self.nID_xc = None
        self.nID_yc = None
        self.nID_zc = None
        
        ## 
        self.num_group = len(elsets)
        
        self.__init_matrix()
        
    
    def solve(self, q_v, p_ext=None):
        """
        计算自由位移并返回张力
        params:
            - q_v: 力密度输入（可以是按 elsets 分组的向量，也可以是全量向量）
        returns:
            - new_ncoord: 节点坐标 (num_node, 3)
            - tensions: 单元张力向量 (num_elemt,)
        """
        # 1. 组装全量力密度向量 q
        q = np.zeros(self.num_elemt, dtype=float)
        # 修正：如果 q_v 是按分组传入的，则按 elsets 分配
        if len(q_v) == len(self.elsets):
            for i, elset_indices in enumerate(self.elsets):
                q[elset_indices] = q_v[i]
        else:
            q = q_v # 否则假设传入的是全量向量

        Q = np.diag(q)
        new_ncoord = self.ncoord.copy()
        
        if p_ext is None:
            p_ext = np.zeros((self.num_node, 3))

        # 2. 求解各维度坐标 (X, Y, Z)
        dims = [
            ('x', self.nID_xf, self.nID_xc, 0),
            ('y', self.nID_yf, self.nID_yc, 1),
            ('z', self.nID_zf, self.nID_zc, 2)
        ]

        for _, nID_f, nID_c, col_idx in dims:
            if len(nID_f) == 0: continue
            Cf = self.C[:, nID_f]
            Cc = self.C[:, nID_c]
            A = Cf.T @ Q @ Cf
            pf = p_ext[nID_f, col_idx]
            xc = self.ncoord[nID_c, col_idx]
            b = pf - Cf.T @ Q @ (Cc @ xc)
            xf = np.linalg.solve(A, b)
            new_ncoord[nID_f, col_idx] = xf

        # 3. 计算单元张力 (T = q * L)
        # 计算每个单元两个端点之间的矢量差: dX = C @ X
        # C 是拓扑矩阵 [num_elemt x num_node]
        diffs = self.C @ new_ncoord 
        
        # 计算单元长度 L = sqrt(dx^2 + dy^2 + dz^2)
        lengths = np.linalg.norm(diffs, axis=1)
        
        # 张力 T = q * L
        tensions = q * lengths

        return new_ncoord, tensions
    
    
    def compute_q(self, current_tensions):
        """
        逻辑：q_new = T_current / L_ideal
        以此强行将结构向目标坐标（ncoord）拉拢
        """
        q_groups = np.zeros(self.num_group)

        # 1. 计算目标坐标（理想状态）下的全量单元长度
        # 使用 self.ncoord（目标位置）计算
        diffs_ideal = self.C @ self.ncoord 
        lengths_ideal = np.linalg.norm(diffs_ideal, axis=1)

        # 2. 按组统计新的力密度
        for i, element_indices in enumerate(self.elsets):
            # 提取当前求解出来的张力
            group_t = current_tensions[element_indices]
            # 提取目标坐标下的理想长度
            group_l_ideal = lengths_ideal[element_indices]
            
            # 计算理想力密度配比
            # 如果长度极小，通常是支座或重合点，需避开
            group_qs = group_t / (group_l_ideal + 1e-9)
            
            # 取均值作为该组下一轮的输入 q
            q_groups[i] = np.mean(group_qs)
            
        return q_groups
    
    def __init_matrix(self):
        """初始化拓扑矩阵和节点约束向量"""
        ##========= 拓扑矩阵 ==========
        self.C = np.zeros((self.num_elemt, self.num_node), dtype=float)
        # 装配
        for i in range(self.num_elemt):
            # 节点编号
            nIDs = self.conn[i, :]
            self.C[i, nIDs] = np.array([1.0, -1.0])
        
        ##========== 节点约束向量===========
        self.nID_xc = []
        self.nID_yc = []
        self.nID_zc = []
        # 边界
        for bdir, bc in self.bcs.items():
            if bdir == "x":
                self.nID_xc.extend(bc)
            elif bdir == "y":
                self.nID_yc.extend(bc)
            elif bdir == "z":
                self.nID_zc.extend(bc)
            else:
                raise ValueError(f"wrong boundary direction: {bdir}")
            
        # 确保约束节点列表是去重后的（以防输入的 bcs 有重复）
        self.nID_xc = sorted(list(set(self.nID_xc)))
        self.nID_yc = sorted(list(set(self.nID_yc)))
        self.nID_zc = sorted(list(set(self.nID_zc)))

        # 生成所有节点的索引集合
        all_nodes = set(range(self.num_node))

        # 计算自由节点：所有节点 减去 约束节点
        self.nID_xf = sorted(list(all_nodes - set(self.nID_xc)))
        self.nID_yf = sorted(list(all_nodes - set(self.nID_yc)))
        self.nID_zf = sorted(list(all_nodes - set(self.nID_zc)))
        