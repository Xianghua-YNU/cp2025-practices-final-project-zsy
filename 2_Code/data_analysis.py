"""
数据后处理与分析模块：用于对模拟结果进行分析。
"""
import numpy as np
import pandas as pd

def analyze_data(data):
    """
    示例：对模拟数据进行基本分析。
    """
    print("Analyzing data...")
    # 这里可以进行统计分析、特征提取等
    # 例如：计算平均值、标准差、拟合曲线等
    # analyzed_result = {
    #     'mean': np.mean(data),
    #     'std': np.std(data)
    # }
    return data # 暂时返回原始数据

# 可以添加更多数据分析函数
def mesh(Nx, Ny):
    dx, dy = 1 / Nx, 1 / Ny
    x_line = np.linspace(0 - dx / 2, 1 + dx / 2, Nx + 2)
    y_line = np.linspace(1 + dy / 2, 0 - dy / 2, Ny + 2)
    m, n = np.meshgrid(x_line, y_line, indexing='xy')
    return m, n

def boundary_condition(T, Nx, Ny):
    [m, n] = mesh(Nx, Ny)

    T[0, :] = 2*np.sin(np.pi * m[0, :])-T[1,:]
    T[Nx + 1, :] = -T[Nx,:]
    T[:, Ny + 1] = -T[:,Ny]
    T[:, 0] = -T[:,1]

    return

def T_exact(Nx, Ny):
    m, n = mesh(Nx, Ny)
    x, y = m[1:-1, 1:-1], n[1:-1, 1:-1]
    T_ext = np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)
    return T_ext
def norm(T, Nx, Ny):
    T_ext = T_exact(Nx, Ny)
    Error = np.abs(T_ext - T)
    L_1 = np.mean(Error)
    L_2 = np.sqrt(np.mean(Error ** 2))
    L_inf = np.max(Error)
    return L_1,L_2,L_inf
