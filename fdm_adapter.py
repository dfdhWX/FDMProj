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
        connID = self.model.conn[ceIdx,:]
        # 将节点ID替换为数组索引
        connIdx = np.zeros_like(connID, dtype=int)
        for i in range(connID.shape[0]):
            nidx1 = self.nid_to_idx[connID[i, 0]]
            nidx2 = self.nid_to_idx[connID[i, 1]]
            # 记录节点索引
            connIdx[i,:] = np.array([nidx1, nidx2]) 

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

        return ncoord, connIdx, q_v, elemIDs, bcs
    
    def update_hca_model(self, new_xyz, tensions, force_densities=None):
        """
        将计算结果同步回 HCA 类
        :param new_xyz: Solver 返回的节点坐标 (N_active, 3)
        :param tensions: Solver 返回的索单元张力向量 (E_cables,)
        :param force_densities: 可选，Solver 返回的力密度向量 (E_cables,)
        """
        # 1. 更新节点坐标
        new_coord = self.model.node.copy()
        for i, xyz in enumerate(new_xyz):
            nid = self.idx_to_nid[i]
            p_idx = self.model.nid_map[nid]
            new_coord[p_idx] = xyz
        
        # 2. 更新单元张力 (核心逻辑：类似于 conn 的逆向映射)
        # 假设 self.model.element_data 是存储全量单元张力的字典或数组
        full_tensions = np.zeros(len(self.model.eid_map)) # 这里的长度应对应 HCA 全量单元
        
        # 遍历 Solver 中的每一个索单元索引
        for i, t in enumerate(tensions):
            eid = self.idx_to_ceid[i]      # 获取原始单元 ID
            e_idx_hca = self.model.eid_map[eid] # 获取在 HCA 模型全量数组中的索引
            full_tensions[e_idx_hca] = t

        # 3. 同步力密度 (如果有)
        full_qs = None
        if force_densities is not None:
            full_qs = np.zeros(len(self.model.eid_map))
            for i, q in enumerate(force_densities):
                eid = self.idx_to_ceid[i]
                e_idx_hca = self.model.eid_map[eid]
                full_qs[e_idx_hca] = q

        # 返回更新后的全量数据，或者直接赋值给 self.model
        return new_coord, full_tensions, full_qs
            
        
    def __get_cable_eids(self):
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
    
    
    def get_surf_cable_eids(self,):
        """
        从嵌套的 elset 中提取面索单元的 ID
        """
        elsets = []
        surf_eids = []
        # 遍历面索集合
        for sub_type in self.model.elset["surface_cable"]:
            eIDs = self.model.elset["surface_cable"][sub_type]
            elsets.extend(eIDs)
            
        # 展开
        for elset in elsets:
            ## 将单元ID转换为索引
            for eID in elset:
                eIdx = self.ceid_to_idx[eID]
                surf_eids.extend([eIdx])
            
        return surf_eids
