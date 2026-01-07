from scipy.io import savemat
import os
import numpy as np

def save_to_matlab(adapter, final_coords, tensions, filename="HCA_Full_Data.mat"):
    """
    导出全量数据，包含节点、拓扑、位移、张力以及所有单元集合(Elsets)
    """
    target_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    file_path = os.path.join(target_dir, filename)

    # 1. 获取基础物理数据
    # 注意：adapter 内部已经处理了 NID 到 0-based 索引的转换
    initial_coords, connIdx, _, _, _ = adapter.get_solver_inputs()
    displacement = final_coords - initial_coords

    # 2. 核心：处理单元集合 (Elsets)
    # 我们需要将 HCA 模型中的 EID 转换为 Solver 里的 1-based 索引 (MATLAB用)
    def process_elset_recursive(d):
        new_dict = {}
        for k, v in d.items():
            if isinstance(v, dict):
                new_dict[k] = process_elset_recursive(v)
            elif isinstance(v, list):
                # 展平嵌套列表并转换 ID 到索引
                flat_eids = []
                for item in v:
                    if isinstance(item, list): flat_eids.extend(item)
                    else: flat_eids.append(item)
                
                # 转换为 Solver 索引并转为 1-based
                indices = []
                for eid in flat_eids:
                    if eid in adapter.ceid_to_idx:
                        indices.append(adapter.ceid_to_idx[eid] + 1)
                new_dict[k] = np.array(indices).reshape(-1, 1)
        return new_dict

    mat_elsets = process_elset_recursive(adapter.model.elset)

    # 3. 构造导出字典
    mat_data = {
        "nodes": final_coords,
        "initial_nodes": initial_coords,
        "elements": connIdx + 1,  # 全量连接关系 (1-based)
        "U": displacement,
        "T": tensions.reshape(-1, 1),
        "elsets": mat_elsets      # 嵌套结构体
    }

    savemat(file_path, mat_data)
    print(f"📊 [IO] 包含集合的数据已导出: {file_path}")
    return file_path