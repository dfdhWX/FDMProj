import numpy as np
import os

class AbaqusExporter:
    def __init__(self, solver):
        """
        :param solver: 已初始化的 FDMSolver 对象
        """
        self.solver = solver

    def write_inp(self, filename, final_coords=None, final_tensions=None, area=1.0):
        """
        导出 INP 文件。
        如果没有提供 final_coords，则使用 solver.ncoord (初始坐标)。
        如果没有提供 final_tensions，则仅导出网格，不包含初始应力和分析步。
        """
        # 确定使用的坐标：优先使用传入的坐标，若无则使用 solver 内部的
        coords = final_coords if final_coords is not None else self.solver.ncoord
        
        print(f"--- 正在生成 Abaqus 文件: {filename} ---")
        
        with open(filename, 'w') as f:
            # 1. 标题
            f.write("*HEADING\n")
            f.write(f"**FDM Mesh Export - {'Full Analysis' if final_tensions is not None else 'Mesh Only'}\n")

            # 2. 节点定义
            f.write("*NODE\n")
            for i, (x, y, z) in enumerate(coords):
                f.write(f"{i + 1}, {x}, {y}, {z}\n")

            # 3. 单元定义
            f.write("*ELEMENT, TYPE=T3D2, ELSET=ALL_CABLES\n")
            for i, (n1, n2) in enumerate(self.solver.conn):
                f.write(f"{i + 1}, {int(n1) + 1}, {int(n2) + 1}\n")

            # 4. 集合定义 (NSET) 用于边界条件
            fix_nodes, hoop_nodes = [], []
            if isinstance(self.solver.bcs, dict):
                set_x = set(self.solver.bcs.get("x", []))
                set_y = set(self.solver.bcs.get("y", []))
                set_z = set(self.solver.bcs.get("z", []))

                fix_nodes = sorted(list(set_x & set_y & set_z))
                if fix_nodes:
                    f.write("*NSET, NSET=NFIX\n")
                    for i in range(0, len(fix_nodes), 10):
                        chunk = [str(int(n) + 1) for n in fix_nodes[i:i+10]]
                        f.write(", ".join(chunk) + "\n")

                hoop_nodes = sorted(list((set_x & set_y) - set_z))
                if hoop_nodes:
                    f.write("*NSET, NSET=NHOOP\n")
                    for i in range(0, len(hoop_nodes), 10):
                        chunk = [str(int(n) + 1) for n in hoop_nodes[i:i+10]]
                        f.write(", ".join(chunk) + "\n")

            # 5. 单元集合 (ELSET)
            if hasattr(self.solver, 'elsets'):
                for g_idx, e_indices in enumerate(self.solver.elsets):
                    f.write(f"*ELSET, ELSET=GROUP_{g_idx}\n")
                    for i in range(0, len(e_indices), 10):
                        chunk = [str(int(idx) + 1) for idx in e_indices[i:i+10]]
                        f.write(", ".join(chunk) + "\n")

            # 6. 属性与材料
            f.write("*SOLID SECTION, ELSET=ALL_CABLES, MATERIAL=STEEL_MAT\n")
            f.write(f"{area},\n")
            f.write("*MATERIAL, NAME=STEEL_MAT\n*ELASTIC\n2.1E11, 0.3\n")

            # 7. 施加边界条件
            f.write("*BOUNDARY\n")
            if fix_nodes:
                f.write("NFIX, 1, 3, 0.0\n")
            if hoop_nodes:
                f.write("NHOOP, 1, 2, 0.0\n")

            # --- 关键逻辑分支：只有存在张力数据时才导出计算步 ---
            if final_tensions is not None:
                # 8. 初始应力
                f.write("*INITIAL CONDITIONS, TYPE=STRESS\n")
                for i, t in enumerate(final_tensions):
                    f.write(f"{i + 1}, {t / area}\n")

                # 9. 分析步
                f.write("*STEP, NAME=VERIFY, NLGEOM=YES\n*STATIC\n")
                f.write("1.0, 1.0, 1e-05, 1.0\n")
                f.write("*NODE PRINT, FREQUENCY=1\nU, RF\n*END STEP\n")
            else:
                print(f"[Info] 未提供张力数据，仅导出网格及集合定义。")

        print(f"✨ 成功导出: {filename}")