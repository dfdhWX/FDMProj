import numpy as np

class AbaqusExporter:
    def __init__(self, solver):
        """
        :param solver: 已初始化的 FDMSolver 对象
        """
        self.solver = solver

    def write_inp(self, filename, final_coords, final_tensions, area=1.0):
        """
        使用 NSET 方式导出 INP 文件
        """
        print(f"--- 正在生成 Abaqus 校验文件: {filename} ---")
        
        with open(filename, 'w') as f:
            # 1. 标题
            f.write("*HEADING\n")
            f.write("**FDM Equilibrium Check\n")

            # 2. 节点定义
            f.write("*NODE\n")
            for i, (x, y, z) in enumerate(final_coords):
                f.write(f"{i + 1}, {x}, {y}, {z}\n")

            # 3. 单元定义
            f.write("*ELEMENT, TYPE=T3D2, ELSET=ALL_CABLES\n")
            for i, (n1, n2) in enumerate(self.solver.conn):
                f.write(f"{i + 1}, {int(n1) + 1}, {int(n2) + 1}\n")

            # 4. 定义节点集合 (NSET) 用于边界条件
            if isinstance(self.solver.bcs, dict):
                set_x = set(self.solver.bcs.get("x", []))
                set_y = set(self.solver.bcs.get("y", []))
                set_z = set(self.solver.bcs.get("z", []))

                # --- 全约束集合 NFIX ---
                fix_nodes = sorted(list(set_x & set_y & set_z))
                if fix_nodes:
                    f.write("*NSET, NSET=NFIX\n")
                    for i in range(0, len(fix_nodes), 10):
                        chunk = [str(int(n) + 1) for n in fix_nodes[i:i+10]]
                        f.write(", ".join(chunk) + "\n")

                # --- 平面约束集合 NHOOP ---
                hoop_nodes = sorted(list((set_x & set_y) - set_z))
                if hoop_nodes:
                    f.write("*NSET, NSET=NHOOP\n")
                    for i in range(0, len(hoop_nodes), 10):
                        chunk = [str(int(n) + 1) for n in hoop_nodes[i:i+10]]
                        f.write(", ".join(chunk) + "\n")

            # 5. 定义单元集合 (ELSET) 用于结果查看
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

            # 7. 施加边界条件 (通过 NSET)
            f.write("*BOUNDARY\n")
            if "fix_nodes" in locals() and fix_nodes:
                # 约束 NFIX 集合的 1, 2, 3 自由度
                f.write("NFIX, 1, 3, 0.0\n")
            if "hoop_nodes" in locals() and hoop_nodes:
                # 约束 NHOOP 集合的 1, 2 自由度 (Z方向自由)
                f.write("NHOOP, 1, 2, 0.0\n")

            # 8. 初始应力
            f.write("*INITIAL CONDITIONS, TYPE=STRESS\n")
            for i, t in enumerate(final_tensions):
                f.write(f"{i + 1}, {t / area}\n")

            # 9. 分析步
            f.write("*STEP, NAME=VERIFY, NLGEOM=YES\n*STATIC\n")
            f.write("1.0, 1.0, 1e-05, 1.0\n")
            f.write("*NODE PRINT, FREQUENCY=1\nU, RF\n*END STEP\n")

        print(f"成功导出: {filename}")