import numpy as np
import pyvista as pv


class HCA_Mesh_Generator:
    def __init__(self, D=16.0, F=10.0, H=10.0, h=5.0, n_r=4, n_theta=12):
        """Initialize the HCA Mesh Generator with given parameters.
        - D : Diameter of the HCA
        - F : Focal length
        - H : Height of the HCA
        - h : The bottom height from the base plane
        - n_r : Number of radial divisions
        - n_theta : Number of angular divisions
        """
        self.D = D
        self.F = F
        self.H = H
        self.h = h
        self.n_r = n_r
        self.n_theta = n_theta
        # 单元集合容器
        self.elset = {
            "beam": {
                "column": [],  # 立柱单元
                "hoop": [],  # 环梁单元
            },  # 主梁单元
            "surface_cable": {
                "radiual": [],  # 径向索单元
                "circumferential": [],  # 环向索单元
                "edge": [],  # 边缘索单元
                "tie": [],  # 连接索单元
            },  # 面索单元--组成网面
            "support_cable": {
                "upper": [],  # 上部支撑索单元
                "lower": [],  # 下部支撑索单元
            },  # 支撑索单元--牵拉hoop
        }
        # 单元节点连接关系(数组)
        self.conn = None
        # 节点(数组)
        self.node = None

    def generate_mesh(self):
        """
        生成HCA网格
        """
        if self.n_r < 2:
            raise ValueError("n_r must be at least 2 to generate a valid mesh.")
        if self.n_theta % 6 != 0:
            raise ValueError("n_theta must be a multiple of 6 for symmetry.")

        # ======= 生成初始网格 =======
        # 径向网格
        self.node = np.zeros((self.n_r + 1, 3))  # 节点坐标数组
        self.node[0, -1] = self.h  # 设置中心节点高度
        self.conn = np.zeros((self.n_r, 2), dtype=int)  # 单元连接关系数组
        dr = (self.D / 2) / self.n_r  # 径向增量
        for ir in range(1, self.n_r + 1):
            r = ir * dr
            x = r
            y = 0.0
            z = self.__compute_z(x, y)
            self.node[ir, :] = np.array([x, y, z])
            self.conn[ir - 1, :] = np.array([ir - 1, ir])

        # 旋转生成完整网格
        self.rotate_mesh(elset=list(range(self.n_r)))
        # 共节点处理
        # self.unify_mesh(self.node, self.conn)

    def rotate_mesh(self, elset: list):
        """对指定单元(elset)进行旋转阵列, 并将阵列后的节点和单元连接关系添加到 self.node 和 self.conn 中"""
        # 获取需要阵列的单元和节点数
        nelem = len(elset)
        unique_node_id = np.unique(self.conn[elset].flatten())
        nid_map = {nid: i for i, nid in enumerate(unique_node_id)}
        nnode = len(unique_node_id)
        # 提取需要阵列的节点坐标
        r_ncoord = self.node[unique_node_id, :]
        r_coon = self.conn[elset, :]
        # 初始节点坐标和单元
        node, conn = self.node, self.conn

        # 最终节点坐标和单元数组
        self.node = np.zeros(
            (node.shape[0] + nnode * (self.n_theta - 1), node.shape[1])
        )
        self.conn = np.zeros(
            (conn.shape[0] + nelem * (self.n_theta - 1), conn.shape[1])
        )
        # 拷贝初始网格信息
        self.node[: node.shape[0]] = node
        self.conn[: conn.shape[0]] = conn
        ##
        # 计算旋转角度
        dtheta = 2 * np.pi / self.n_theta
        # 旋转阵列
        for itheta in range(1, self.n_theta):
            theta = itheta * dtheta
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # 旋转节点坐标
            rot_matrix = np.array(
                [
                    [cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0, 0, 1],
                ]
            )
            rotated_ncoord = r_ncoord @ rot_matrix.T
            # 确定新节点在总体节点数组中的位置
            nstart_idx = node.shape[0] + (itheta - 1) * nnode
            nend_idx = node.shape[0] + (itheta * nnode)
            self.node[nstart_idx:nend_idx] = rotated_ncoord
            # 确定新单元在总体单元数组中的位置
            estart_idx = conn.shape[0] + (itheta - 1) * nelem
            eend_idx = conn.shape[0] + (itheta * nelem)
            # 更新单元连接关系
            offset = itheta * nnode
            rotated_conn = np.array(
                [
                    [
                        nid_map[self.conn[elset[i], 0]] + offset,
                        nid_map[self.conn[elset[i], 1]] + offset,
                    ]
                    for i in range(nelem)
                ]
            )
            self.conn[estart_idx:eend_idx] = rotated_conn

    def unify_mesh(self, ncoord: np.array, connect: np.array, tol=1e-6):
        """对网格进行共节点处理"""
        # 对节点坐标进行四舍五入以减少浮点误差
        ncoord = np.round(ncoord / tol) * tol

        # 创建节点索引映射
        unique_nodes = np.unique(ncoord, axis=0)
        node_map = {tuple(node): i for i, node in enumerate(unique_nodes)}

        # 更新连接关系
        new_connect = np.array(
            [node_map[tuple(ncoord[i])] for i in connect.flatten()]
        ).reshape(-1, 2)

        return unique_nodes, new_connect

    def export_mesh(self, filename: str):
        """导出网格数据到inp文件"""
        pass

    def __compute_z(self, x: float, y: float) -> float:
        """计算给定坐标(x,y)处的z坐标"""
        z = self.h + (x**2 + y**2) / (4 * self.F)
        return z

    def mesh_plot(self, ncoord: np.array, conn: np.array, show_labels: bool = True):
        """绘制网格图，并可选显示节点和单元编号"""

        nelem = conn.shape[0]

        # 构建 connectivity（LINE 单元）
        conn_full = np.hstack(
            [np.full((nelem, 1), 2, dtype=np.int32), conn.astype(np.int32)]
        ).flatten()

        # 创建 UnstructuredGrid
        mesh = pv.UnstructuredGrid(conn_full, np.full(nelem, pv.CellType.LINE), ncoord)

        # 创建 Plotter
        plotter = pv.Plotter()

        # 添加网格线
        plotter.add_mesh(mesh, color="blue", line_width=2)

        # === 显示节点编号 ===
        point_labels = [str(i) for i in range(ncoord.shape[0])]
        plotter.add_point_labels(
            ncoord,
            point_labels,
            font_size=20,
            text_color="red",
            always_visible=True,
            shadow=True,
            fill_shape=False,  # 不加背景框（可选）
        )

        if show_labels:
            # === 显示单元编号 ===
            # 计算每个单元的中心点作为标签位置
            cell_centers = mesh.cell_centers().points  # shape: (nelem, 3)
            cell_labels = [str(i) for i in range(nelem)]
            plotter.add_point_labels(
                cell_centers,
                cell_labels,
                font_size=20,
                text_color="green",
                always_visible=True,
                shadow=True,
                shape_opacity=0.2,  # 轻微背景（可选）
            )

        # 隐藏坐标轴和边界
        plotter.hide_axes()
        plotter.show_bounds = False

        plotter.show()


if __name__ == "__main__":
    # 测试代码
    generator = HCA_Mesh_Generator(n_r=4, n_theta=24)
    generator.generate_mesh()
    conn = generator.conn
    node = generator.node
    generator.mesh_plot(node, conn)
