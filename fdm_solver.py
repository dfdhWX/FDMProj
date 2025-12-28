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
    
    def solve(self, q_v):
        """计算自由位移\n
        params:
            - q_v: 力密度向量(每种类型)
        returns:
            - new_ncoord: 节点坐标"""
        pass