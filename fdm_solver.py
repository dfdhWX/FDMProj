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
    
    
    def compute_q(self, tension, ncoord=None):
        """
        计算每个单元集的初始力密度 q = T / L
        
        参数:
        :param tension: 目标张力，可以是标量（所有组统一）或列表/数组（每组对应一个张力）
        
        返回:
        :return q_groups: 长度等于 len(self.elsets) 的力密度数组
        """
        if ncoord is None:
            ncoord = self.ncoord
            
        q_groups = np.zeros(self.num_group)
        
        # 确保 tension 格式统一为数组
        if isinstance(tension, (int, float)):
            tension_values = [tension] * self.num_group
        else:
            tension_values = tension

        for i, element_indices in enumerate(self.elsets):
            group_lengths = []
            
            # 遍历该组内的每一个单元，计算其实际长度
            for e_idx in element_indices:
                node_i, node_j = self.conn[e_idx]
                p1 = ncoord[node_i]
                p2 = ncoord[node_j]
                
                # 计算欧式距离（单元长度 L）
                length = np.linalg.norm(p1 - p2)
                if length > 1e-9: # 防止除以零
                    group_lengths.append(length)
            
            if not group_lengths:
                q_groups[i] = 1.0 # 如果组内无有效单元，给个默认值
                continue
                
            # 计算该组单元的平均长度
            avg_length = np.mean(group_lengths)
            
            # 根据 q = T / L 计算该组的力密度
            q_groups[i] = tension_values[i] / avg_length
            
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
        