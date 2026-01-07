import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_result(data, title="Optimization Results", mode='line', use_log=True):
    # ... (你定义的 plot_result 代码保持不变) ...
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data

    x = df.iloc[:, 0]
    y_cols = df.iloc[:, 1:]

    plt.figure(figsize=(10, 6), dpi=100)
    
    for i, col_name in enumerate(y_cols.columns):
        y = y_cols[col_name]
        # 为 GA 数据定制标签
        if i == 0: label = "Best Fitness (f_min)"
        elif i == 1: label = "Average Fitness (f_avg)"
        else: label = f"Column {col_name}"
        
        if mode == 'scatter':
            plt.scatter(x, y, label=label, s=10, alpha=0.7)
        else:
            plt.plot(x, y, label=label, linewidth=1.5)

    if use_log:
        plt.yscale('log')
        plt.ylabel('Value (Log Scale)')
    else:
        plt.ylabel('Value')

    plt.xlabel('Generation (Index)')
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.show()

def main():
    # 1. 文件路径 (使用 raw string 防止 \F \d 被识别为转义字符)
    fpath = r"D:\Python-file\FDMProj\data\Final_R0.5716_T14.49.csv"
    
    if not os.path.exists(fpath):
        print(f"❌ 找不到文件: {fpath}")
        return

    # 2. 读取文件
    # 假设你的 CSV 是有表头的 (gen, f_min, f_avg)
    df = pd.read_csv(fpath)
    
    # 确保第一列 gen 是整型（虽然 pandas 会自动识别，但这样更保险）
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    
    # 3. 绘图 
    # 建议使用 'line' 模式观察收敛，'scatter' 模式观察波动
    plot_result(
        df.iloc[:, [0, 2]], 
        title="Reflector Optimization Convergence History", 
        mode='scatter', 
        use_log=False
    )

if __name__ == "__main__":
    main()