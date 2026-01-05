import numpy as np
import pyvista as pv


class HCA_Mesh_Generator:
    def __init__(self, D=10.0, F=6.0, H=3.0, h=1.0, n_r=4, n_theta=36):
        self.D, self.F, self.H, self.h = D, F, H, h
        self.n_r, self.n_theta = n_r, n_theta

        # 单元集合容器
        self.elset = {
            "beam": {
                "column": [],  # 立柱单元
                "hoop": [],  # 环梁单元
            },  # 主梁单元
            "surface_cable": {
                "radial": [],  # 径向索单元
                "circumferential": [],  # 环向索单元
                "edge": [],  # 边缘索单元
                "tie": [],  # 连接索单元
            },  # 面索单元--组成网面
            "support_cable": {
                "upper": [],  # 上部支撑索单元
                "lower": [],  # 下部支撑索单元
            },  # 支撑索单元--牵拉hoop
        }

        # 节点集容器
        self.nset = {
            "hoop": [],
            "fix": [],
        }

        # 核心存储
        self.node = np.empty((0, 3))
        self.conn = np.empty((0, 2), dtype=int)

        # ID 映射表 (ID -> Array Index)
        self.nid_map = np.full(1000000, -1, dtype=int)
        self.eid_map = np.full(1000000, -1, dtype=int)

        # 平移
        self.__node_offset = None
        self.__elem_offset = None

    def generate_mesh(self):
        """生成HCA网格"""
        # =========索单元===============
        ##======== 生成径向节点、单元=======
        rncoord = np.zeros((self.n_r + 1, 3))
        rconn = np.zeros((self.n_r, 2), dtype=int)
        dr = 0.5 * self.D / self.n_r
        for i in range(self.n_r + 1):
            # 计算节点坐标
            xr = i * dr
            yr = 0.0
            zr = self.__compute_z(xr, yr)
            # 添加节点坐标
            rncoord[i, :] = np.array([xr, yr, zr])
            # 更新节点 ID 映射表
            self.nid_map[i + 1] = i
            # 单元连接
            if i == self.n_r:
                continue
            rconn[i, :] = np.array([i + 1, i + 2])
            # 单元ID 映射表
            self.eid_map[i + 1] = i
        # ========== 生成环向节点、单元========
        # 旋转角
        dtheta = 2 * np.pi / self.n_theta
        # 坐标变换矩阵
        Rz = np.array(
            [
                [np.cos(dtheta), -np.sin(dtheta), 0],
                [np.sin(dtheta), np.cos(dtheta), 0],
                [0, 0, 1],
            ]
        )
        # 环向节点坐标
        cncoord = np.zeros((self.n_r - 1, 3))
        cconn = np.zeros((self.n_r - 1, 2), dtype=int)
        # 更新节点坐标
        for i in range(self.n_r - 1):
            # 节点坐标
            coord = Rz @ rncoord[i + 1, :].T
            cncoord[i, :] = coord
            # 节点 ID 映射表
            self.nid_map[self.n_r + 2 + i] = 1 + self.n_r + i
            # 连接
            cconn[i, :] = np.array([i + 2, self.n_r + 2 + i])
            # 单元ID 映射表
            self.eid_map[1 + self.n_r + i] = self.n_r + i
        # 将径向和环向网格添加
        self.node = np.vstack((rncoord, cncoord))
        self.conn = np.vstack((rconn, cconn))
        ## 径向
        # 1. 生成单元并分配 ID
        seed_eids = [[v] for v in range(1, self.n_r + 1)]
        # 2. 存入单元集合
        self.elset["surface_cable"]["radial"].extend(seed_eids)
        ## 环向
        # 1. 生成单元并分配 ID
        seed_eids = [[v + self.n_r + 1] for v in range(self.n_r - 1)]
        # 2. 存入单元集合
        self.elset["surface_cable"]["circumferential"].extend(seed_eids)
        ##=====================生成张拉索单元、节点=================
        # 节点坐标
        tncoord = np.zeros((1, 3))
        self.nid_map[2 * self.n_r + 1] = 2 * self.n_r
        # 连接单元
        tconn = np.zeros((self.n_r - 1, 2), dtype=int)
        for i in range(self.n_r - 1):
            tconn[i, :] = np.array([2 * self.n_r + 1, i + 2])
            # 分类
            self.elset["surface_cable"]["tie"].append([2 * self.n_r + i])
            # 单元 ID
            self.eid_map[2 * self.n_r + i] = 2 * self.n_r + i - 1
        ## 添加进全局
        self.conn = np.vstack((self.conn, tconn))
        self.node = np.vstack((self.node, tncoord))

        ##================支撑索===================
        # 新增节点坐标
        sncoord = np.array([0.0, 0.0, self.H])
        # 添加到全局
        self.node = np.vstack((self.node, sncoord))
        # 跟新节点ID映射表
        self.nid_map[2 * self.n_r + 2] = 2 * self.n_r + 1
        # 连接单元
        sconn = np.array(
            [
                [2 * self.n_r + 2, self.n_r + 1],  # 上支持索
                [2 * self.n_r + 1, self.n_r + 1],  # 下支持索
            ]
        )
        # 添加到全局
        self.conn = np.vstack((self.conn, sconn))
        # 更新单元ID映射表
        self.eid_map[3 * self.n_r - 1 : 3 * self.n_r + 1] = np.array(
            [3 * self.n_r - 2, 3 * self.n_r - 1]
        )
        # 添加进单元集
        self.elset["support_cable"]["upper"].append([3 * self.n_r - 1])
        self.elset["support_cable"]["lower"].append([3 * self.n_r])

        ##================= 梁单元=====================
        # =========== Hoop 单元===================
        hncoord = rncoord[-1, :] @ Rz.T
        hconn = np.array([self.n_r + 1, 2 * self.n_r + 3])
        # 添加到全局
        self.node = np.vstack((self.node, hncoord))
        self.nid_map[2 * self.n_r + 3] = 2 * self.n_r + 2
        self.conn = np.vstack((self.conn, hconn))
        self.eid_map[3 * self.n_r + 1] = 3 * self.n_r
        self.elset["beam"]["hoop"].append(3 * self.n_r + 1)
        # =========== Column 单元==============
        clconn = np.array([[1, 2 * self.n_r + 1], [1, 2 * self.n_r + 2]])
        # 添加到全局
        self.conn = np.vstack((self.conn, clconn))
        # 更新单元ID映射表
        self.eid_map[3 * self.n_r + 2 : 3 * self.n_r + 4] = np.array(
            [3 * self.n_r + 1, 3 * self.n_r + 2]
        )
        # 分类
        self.elset["beam"]["column"].extend([3 * self.n_r + 2, 3 * self.n_r + 3])
        ##============= 单元、节点平移=================
        self.__node_offset = self.nid_map.max()
        self.__elem_offset = self.eid_map.max()
        ##================旋转阵列=================
        # 径向
        for i in range(self.n_r):
            elset = self.elset["surface_cable"]["radial"][i]
            self.rotate_mesh(elset)
        # 环向
        for i in range(self.n_r - 1):
            elset = self.elset["surface_cable"]["circumferential"][i]
            self.rotate_mesh(elset)
        # 张拉
        for i in range(self.n_r - 1):
            elset = self.elset["surface_cable"]["tie"][i]
            self.rotate_mesh(elset)
        # 支撑索
        for elset in self.elset["support_cable"].values():
            self.rotate_mesh(elset[0])
        # hoop
        elset = self.elset["beam"]["hoop"]
        self.rotate_mesh(elset)
        # # 3. 类似 HyperMesh 的共节点处理
        self.unify_mesh()
        
        ##================ 添加节点集=====================
        ## 固定节点
        fix_nodes = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, self.h],
            [0.0, 0.0, self.H]
        ])
        for i in range(fix_nodes.shape[0]):
            nIDs = self.get_node_ID(fix_nodes[i])
            self.nset["fix"].extend(nIDs)
        ## hoop 节点
        xh = 0.5*self.D 
        zh = self.__compute_z(xh, 0.0)
        hoop_nodes = np.array([xh, zh])
        nIDs = self.get_node_ID(hoop_nodes, mode='rz')
        # 添加进集合
        self.nset["hoop"].extend(nIDs)


    def rotate_mesh(self, target_elset: list):
        base_eids = np.array(target_elset)
        base_e_indices = self.eid_map[base_eids]
        base_nids = np.unique(self.conn[base_e_indices].flatten())
        r_ncoord = self.node[self.nid_map[base_nids]]

        # 自动计算能避开所有潜在冲突的偏移步进
        node_offset = int(np.ceil((self.__node_offset + 2) / 10.0) * 10)
        elem_offset = int(np.ceil((self.__elem_offset + 2) / 10.0) * 10)

        dtheta = 2 * np.pi / self.n_theta
        for itheta in range(1, self.n_theta):
            theta = itheta * dtheta
            rot_matrix = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )

            # 物理存储追加
            new_coords = r_ncoord @ rot_matrix.T
            n_start_idx = len(self.node)
            self.node = np.vstack([self.node, new_coords])

            # 映射表更新 (ID = 原始 ID + 旋转次数 * 步进)
            new_nids = base_nids + itheta * node_offset

            # 动态扩容 nid_map 以防止溢出
            if np.max(new_nids) >= len(self.nid_map):
                self.nid_map = np.resize(self.nid_map, np.max(new_nids) + 1000)

            self.nid_map[new_nids] = np.arange(
                n_start_idx, n_start_idx + len(new_coords)
            )

            # 单元连接关系 (存储 ID)
            new_conn = self.conn[base_e_indices] + itheta * node_offset
            e_start_idx = len(self.conn)
            self.conn = np.vstack([self.conn, new_conn])

            # 单元 ID 映射更新
            new_eids = base_eids + itheta * elem_offset
            if np.max(new_eids) >= len(self.eid_map):
                self.eid_map = np.resize(self.eid_map, np.max(new_eids) + 1000)

            self.eid_map[new_eids] = np.arange(e_start_idx, e_start_idx + len(new_conn))

            # 更新 elset
            target_elset.extend(new_eids.tolist())
            
    
    def get_node_ID(self, target_coord: np.array, tol=1.0e-6, mode='xyz'):
        """
        通过坐标获取节点ID
        :param target_coord: 目标坐标数组 [x, y, z] 或 [r, theta, z] (或简写 [r, z])
        :param tol: 容差
        :param mode: 'xyz' (笛卡尔) 或 'rz' (柱坐标)
        """
        if len(self.node) == 0:
            return None

        # 1. 获取所有活跃的逻辑 ID 和对应的物理坐标
        active_nIDs = np.where(self.nid_map != -1)[0]
        p_indices = self.nid_map[active_nIDs]
        ncoords = self.node[p_indices, :] # 这里的存储始终是 [x, y, z]

        # 2. 根据模式计算距离
        if mode.lower() == 'xyz':
            # 直接计算欧氏距离
            distances = np.linalg.norm(ncoords - target_coord, axis=1)
            
        elif mode.lower() == 'rz':
            # 将物理坐标的 [x, y, z] 转换为 [r, z]
            # r = sqrt(x^2 + y^2)
            curr_r = np.sqrt(ncoords[:, 0]**2 + ncoords[:, 1]**2)
            curr_z = ncoords[:, 2]
            curr_rz = np.column_stack((curr_r, curr_z))
            
            # 目标坐标处理 (如果输入是 [r, theta, z]，只取 r 和 z)
            if len(target_coord) == 3:
                target_rz = np.array([target_coord[0], target_coord[2]])
            else:
                target_rz = target_coord
                
            distances = np.linalg.norm(curr_rz - target_rz, axis=1)
        else:
            raise ValueError("Mode 必须是 'xyz' 或 'rz'")

        # 3. 寻找匹配项
        matching_indices = np.where(distances < tol)[0]
        
        if len(matching_indices) == 0:
            return None
        
        # 4. 映射回逻辑 ID
        matched_ids = active_nIDs[matching_indices]
        # 5. 转换为列表
        matched_ids = matched_ids.tolist()
        
        return matched_ids
    
    
    def unify_mesh(self, tol=1e-6):
        """
        深度合并：物理坐标去重 + 逻辑 ID 塌陷 + 映射表清理
        """
        if len(self.node) == 0:
            return

        # 1. 查找空间重合点
        scaled = np.round(self.node / tol).astype(int)
        _, first_indices, inverse_indices = np.unique(
            scaled, axis=0, return_index=True, return_inverse=True
        )

        # 2. 更新物理节点数组 (移除重复坐标)
        old_node_count = len(self.node)
        self.node = self.node[first_indices]

        # 3. 构建 ID 映射关系
        # 找出所有活跃的 NID
        active_nids = np.where(self.nid_map != -1)[0]
        # 找出它们原本指向的旧物理索引
        old_phys_idxs = self.nid_map[active_nids].astype(int)
        # 找出它们现在对应的新物理索引
        new_phys_idxs = inverse_indices[old_phys_idxs]

        # 4. 确定每个新位置保留哪一个 NID (取最小 ID)
        # phys_to_keep_nid 字典: {新物理索引: 要保留的最小NID}
        phys_to_keep_nid = {}
        for nid, p_idx in zip(active_nids, new_phys_idxs):
            if p_idx not in phys_to_keep_nid:
                phys_to_keep_nid[p_idx] = nid
            else:
                phys_to_keep_nid[p_idx] = min(phys_to_keep_nid[p_idx], nid)

        # 5. 更新 self.conn 中的节点 ID
        # 创建一个“旧 ID -> 保留 ID”的查找表
        # nid_to_keep_nid: {100: 0, 101: 101, ...}
        nid_to_keep_nid = {
            nid: phys_to_keep_nid[new_phys_indices]
            for nid, new_phys_indices in zip(active_nids, new_phys_idxs)
        }

        # 使用向量化操作替换 self.conn 里的所有旧 ID
        # 注意：如果 ID 不在字典里（理论上不应该），保留原值
        def map_nid(x):
            return nid_to_keep_nid.get(x, x)

        v_map_nid = np.vectorize(map_nid)
        self.conn = v_map_nid(self.conn).astype(np.int32)

        # 6. 清理 self.nid_map (将被删除节点置为 -1)
        # 先全置为 -1，再重新填入保留下来的节点关系
        self.nid_map.fill(-1)
        for p_idx, keep_nid in phys_to_keep_nid.items():
            self.nid_map[keep_nid] = p_idx

        print(f"--- 深度合并完成 ---")
        print(f"节点压缩: {old_node_count} -> {len(self.node)}")
        print(f"保留 ID 数: {len(phys_to_keep_nid)}")

    def __compute_z(self, x, y):
        return self.h + (x**2 + y**2) / (4 * self.F)

    def mesh_plot(self, show_labels=True, label_ratio=1.0):
        """
        优化后的绘图函数
        :param show_labels: 是否显示 ID 标签
        :param label_ratio: 标签显示比例 (0.0~1.0)，防止模型过大时文字重叠
        """
        # --- 1. 提取物理拓扑 (处理 nid_map 映射) ---
        # 确保只绘制有效单元 (eid_map 中有记录的)
        active_eids = np.where(self.eid_map != -1)[0]
        active_conn_ids = self.conn[self.eid_map[active_eids]]

        # 将 NID 映射为 node 数组的行索引
        # 注意：此处 nid_map 可能很大，直接索引比循环快几个数量级
        plot_indices = self.nid_map[active_conn_ids]

        # --- 2. 构建 PyVista 网格 ---
        nelem = len(plot_indices)
        # 批量构建 [2, i, j] 结构
        cells = np.empty((nelem, 3), dtype=np.int32)
        cells[:, 0] = 2
        cells[:, 1:] = plot_indices

        mesh = pv.UnstructuredGrid(
            cells.flatten(), np.full(nelem, pv.CellType.LINE), self.node
        )

        plotter = pv.Plotter(title="HCA Mesh Viewer")
        plotter.add_mesh(mesh, color="blue", line_width=2, label="Cables")

        if show_labels:
            # --- 3. 节点标签优化 (红色) ---
            active_nids = np.where(self.nid_map != -1)[0]
            # 抽样逻辑
            if label_ratio < 1.0:
                mask = np.random.rand(len(active_nids)) < label_ratio
                active_nids = active_nids[mask]

            node_coords = self.node[self.nid_map[active_nids]]
            node_labels = [f"N{n}" for n in active_nids]
            plotter.add_point_labels(
                node_coords,
                node_labels,
                font_size=12,
                text_color="red",
                name="nodes",
                always_visible=False,
            )

            # --- 4. 单元标签优化 (绿色) ---
            # 直接通过 active_conn_ids 计算中心点，避免二次查询
            p1 = self.node[self.nid_map[active_conn_ids[:, 0]]]
            p2 = self.node[self.nid_map[active_conn_ids[:, 1]]]
            centers = (p1 + p2) / 2.0

            # 抽样
            if label_ratio < 1.0:
                mask = np.random.rand(len(active_eids)) < label_ratio
                centers = centers[mask]
                active_eids = active_eids[mask]

            elem_labels = [f"E{e}" for e in active_eids]
            plotter.add_point_labels(
                centers,
                elem_labels,
                font_size=10,
                text_color="green",
                name="elems",
                shadow=True,
            )

        plotter.add_legend(
            labels=[("Node ID", "red"), ("Elem ID", "green")], bcolor=None
        )
        plotter.view_isometric()
        plotter.show()

    def export_to_inp(self, filename="HCA_Model.inp"):
        """
        导出 Inp 文件，仅保留带编号后缀的子集合 (例如 RADIAL-1, RADIAL-2)
        """
        with open(filename, "w", encoding="utf-8") as f:
            # 1. 文件头
            f.write("*Heading\n** Generated by HCA_Mesh_Generator\n")
            # 网格信息
            f.write(
                f"Information of Mesh: Node = {self.node.shape[0]}, Element = {self.conn.shape[0]}\n"
            )

            # 2. 节点 (*NODE)
            f.write("*Node\n")
            active_nids = np.where(self.nid_map != -1)[0]
            for nid in active_nids:
                coord = self.node[self.nid_map[nid]]
                f.write(f"{nid}, {coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f}\n")

            # 3. 单元 (*ELEMENT)
            f.write("*Element, type=T3D2\n")
            active_eids = np.where(self.eid_map != -1)[0]
            for eid in active_eids:
                nodes = self.conn[self.eid_map[eid]]
                f.write(f"{eid}, {nodes[0]}, {nodes[1]}\n")

            # 4. 单元集合 (*ELSET) - 仅导出原子级子集合
            def write_only_sub_elsets(elset_dict, prefix=""):
                for name, content in elset_dict.items():
                    base_name = f"{prefix}{name}".upper()

                    if isinstance(content, dict):
                        # 如果是字典，继续向下递归
                        write_only_sub_elsets(content, prefix=f"{base_name}_")

                    elif isinstance(content, list):
                        if not content:
                            continue

                        # 核心逻辑：如果是嵌套列表，只导出子项
                        if isinstance(content[0], list):
                            for i, sub_list in enumerate(content):
                                if sub_list:
                                    set_name = f"{base_name}-{i+1}"
                                    self._write_single_elset(f, set_name, sub_list)
                        else:
                            # 如果本身就是普通列表，直接导出
                            self._write_single_elset(f, base_name, content)

            write_only_sub_elsets(self.elset)

        print(f"成功导出 Inp 文件: {filename}")

    def _write_single_elset(self, file_handle, set_name, id_list):
        """辅助方法：按照标准格式写入单个 *Elset"""
        file_handle.write(f"*Elset, elset={set_name}\n")
        # 将列表打平并去重（防止 unify_elements 后仍有残余重复 ID，虽然理论上不会）
        flat_ids = sorted(list(set(id_list)))
        for i in range(0, len(flat_ids), 10):
            line = ", ".join(map(str, flat_ids[i : i + 10]))
            file_handle.write(f"{line}\n")

    def export_to_inp1(self, filename="HCA_Model.inp"):
        """
        主导出方法：组织 Part/Assembly 结构并调用各组件写入方法
        """
        TYPE_MAP = {
            "beam": "B31",
            "surface_cable": "T3D2",
            "support_cable": "T3D2",
        }

        with open(filename, "w", encoding="utf-8") as f:
            # --- Header ---
            f.write("*Heading\n** Generated by HCA_Mesh_Generator\n")
            
            # --- Part Start ---
            f.write("*Part, name=HCA_STRUCTURE\n")
            
            # 1. 写入节点
            f.write("*Node\n")
            active_nids = np.where(self.nid_map != -1)[0]
            for nid in active_nids:
                coord = self.node[self.nid_map[nid]]
                f.write(f"{nid}, {coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f}\n")

            # 2. 写入单元组件 (根据 elset 结构)
            written_eids = set()
            self.__write_elset_recursive(f, self.elset, "", "T3D2", TYPE_MAP, written_eids)

            # 3. 写入节点集 (Nset)
            self.__write_nset_recursive(f, self.nset, "")

            f.write("*End Part\n")
            
            # --- Assembly ---
            f.write("*Assembly, name=Assembly\n")
            f.write("*Instance, name=HCA_INST, part=HCA_STRUCTURE\n")
            f.write("*End Instance\n")
            f.write("*End Assembly\n")

        print(f"成功导出 Inp 文件: {filename}")

    # --------------------------- 私有辅助方法 ---------------------------

    def __write_elset_recursive(self, f, elset_dict, prefix, current_type, type_map, written_eids):
        """私有方法：负责递归解析并写入单元集合"""
        for name, content in elset_dict.items():
            branch_type = type_map.get(name, current_type)
            base_name = f"{prefix}{name}".upper()

            if isinstance(content, dict):
                # 递归字典
                self.__write_elset_recursive(f, content, f"{base_name}_", branch_type, type_map, written_eids)
            
            elif isinstance(content, list) and content:
                # 区分是嵌套列表（多根索）还是普通列表
                if isinstance(content[0], list):
                    for i, sub_list in enumerate(content):
                        if sub_list:
                            self._write_element_block(f, f"{base_name}-{i+1}", sub_list, branch_type, written_eids)
                else:
                    self._write_element_block(f, base_name, content, branch_type, written_eids)

    def __write_nset_recursive(self, f, nset_dict, prefix):
        """私有方法：负责递归解析并写入节点集合"""
        for name, content in nset_dict.items():
            base_name = f"{prefix}{name}".upper()
            
            if isinstance(content, dict):
                self.__write_nset_recursive(f, content, f"{base_name}_")
            
            elif isinstance(content, list) and content:
                # 写入 Nset 关键字
                f.write(f"*Nset, nset={base_name}\n")
                # 每行 10 个 ID 保证 Inp 易读性
                for i in range(0, len(content), 10):
                    line = ", ".join(map(str, content[i : i + 10]))
                    f.write(f"{line}\n")

    def _write_element_block(self, f, set_name, eid_list, element_type, written_set):
        """底层写入：将单元 ID 转换为 Node IDs 并写入 *Element 块"""
        valid_eids = [eid for eid in eid_list if eid not in written_set]
        if not valid_eids: return

        f.write(f"*Element, type={element_type}, elset={set_name}\n")
        for eid in valid_eids:
            p_idx = self.eid_map[eid]
            nodes = self.conn[p_idx]
            f.write(f"{eid}, {nodes[0]}, {nodes[1]}\n")
            written_set.add(eid)


if __name__ == "__main__":
    # 测试代码
    generator = HCA_Mesh_Generator(n_r=2, n_theta=6)
    generator.generate_mesh()
    generator.mesh_plot(show_labels=False)
    generator.export_to_inp1(r"D:\hm-file\inp_file\HCA.inp")
