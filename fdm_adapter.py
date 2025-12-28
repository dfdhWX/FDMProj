import numpy as np
from generate_HCA_mesh import HCA_Mesh_Generator

class FDMAdapter:
    def __init__(self, hca_model:HCA_Mesh_Generator):
        """
        :param hca_model: HCA 实例
        """
        self.model = hca_model
        # 建立 逻辑ID 到 数组索引 的映射
        self.active_nids = np.where(self.model.nid_map != -1)[0]
        self.nid_to_idx = {nid: i for i, nid in enumerate(self.active_nids)}
        self.idx_to_nid = {i: nid for i, nid in enumerate(self.active_nids)}
        # 索单元ID 到数组索引的映射
        self.ceid_to_idx = None
        self.idx_to_ceid = None

    def get_solver_inputs(self):
        """将 HCA 业务数据转化为 Solver 矩阵\n
        returns:
            - ncoords: 节点坐标数组
            - conn: 单元节点连接表
            - q_v: 力密度向量
            - elemIDs: 不同类型索单元ID
            - bcs: 边界条件列表"""
        ## =========== 获取所有索单元===============
        elsets = self.__get_cable_eids()
        ceIDs = []
        # 展平索单元集
        for elset in elsets:
            ceIDs.extend(elset)
        # 生成单元ID 与数组索引映射表
        self.ceid_to_idx = {eid: i for i, eid in enumerate(ceIDs)}
        self.idx_to_ceid = {i: eid for i, eid in enumerate(ceIDs)}
        # 将单元ID转换为索引
        ceIdx = [self.model.eid_map[i] for i in ceIDs]
        elemIDs = [[self.ceid_to_idx[i] for i in eset] for eset in elsets]
        # 生成单元节点连接数组
        conn = self.model.conn[ceIdx,:]

        # ================获取坐标==================
        p_indices = self.model.nid_map[self.active_nids]
        ncoord = self.model.node[p_indices, :]
        
        # ===============力密度向量生成===================
        q_v = np.full(len(elsets), 1, dtype=float)
        
        #================ 边界条件生成===================
        bcs = {"x":[], "y":[], "z":[]}
        # 固定节点
        for nid in self.model.nset["fix"]:
            nidx = self.nid_to_idx[nid]
            bcs["x"].extend([nidx])
            bcs["y"].extend([nidx])
            bcs["z"].extend([nidx])
        # hoop 节点--约束 X Y
        for nid in self.model.nset["hoop"]:
            nidx = self.nid_to_idx[nid]
            bcs["x"].extend([nidx])
            bcs["y"].extend([nidx])

        return ncoord, conn, q_v, elemIDs, bcs
    
    
    def __get_cable_eids(self, cable_type = "all"):
        """
        从嵌套的 elset 中提取所有索单元的 ID
        """
        cable_eids = []
        # 遍历面索集合
        for sub_type in self.model.elset["surface_cable"]:
            eIDs = self.model.elset["surface_cable"][sub_type]
            cable_eids.extend(eIDs)
        # 遍历支撑索集合
        for sub_type in self.model.elset["support_cable"]:
            eIDs = self.model.elset["support_cable"][sub_type]
            cable_eids.extend(eIDs)
             
        return cable_eids

    def update_hca_model(self, new_xyz, tensions):
        """将计算结果同步回 HCA 类"""
        # 更新节点坐标
        for i, xyz in enumerate(new_xyz):
            nid = self.idx_to_nid[i]
            p_idx = self.model.nid_map[nid]
            self.model.node[p_idx] = xyz
        
        # 更新单元张力（可以存入一个字典方便导出）
        self.model.element_tensions = {}
        for i, eid in enumerate(self.current_active_eids):
            self.model.element_tensions[eid] = tensions[i]