import numpy as np
import pyvisa as pv
from generate_HCA_mesh import HCA_Mesh_Generator


def main():
    # 创建HCA网格生成器实例
    generator = HCA_Mesh_Generator(n_r=4)
    # 生成HCA网格
    generator.generate_mesh()
    #
    conn = generator.conn
    # 绘制网格图
    generator.mesh_plot()


if __name__ == "__main__":
    main()
