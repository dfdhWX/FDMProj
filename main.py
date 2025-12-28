import numpy as np
from generate_HCA_mesh import HCA_Mesh_Generator
from fdm_adapter import FDMAdapter

def main():
    # =================创建HCA网格生成器实例=================
    hca_model = HCA_Mesh_Generator()
    # 生成HCA网格
    hca_model.generate_mesh()
    # 绘制网格图
    # hca_model.mesh_plot(show_labels=False)
    
    #=================
    adapter = FDMAdapter(hca_model)
    adapter.get_solver_inputs()


if __name__ == "__main__":
    main()
