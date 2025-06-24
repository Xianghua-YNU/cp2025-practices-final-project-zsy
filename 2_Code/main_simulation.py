"""
主程序入口：用于运行主要的物理模拟。
"""
import numpy as np
from scipy.sparse.linalg import spsolve

def run_simulation():
    """
    运行整个模拟流程。
    """
    print("Running main simulation...")
    # 示例：
    # initial_conditions = np.array([0.0, 1.0])
    # time_points = np.linspace(0, 10, 100)
    # solution = solve_ode(initial_conditions, time_points)
    # analyzed_data = analyze_data(solution)
    # plot_results(analyzed_data)

    dx, dy = 1 / Nx, 1 / Ny
    x_line = np.linspace(0 - dx / 2, 1 + dx / 2, Nx + 2)
    y_line = np.linspace(1 + dy / 2, 0 - dy / 2, Ny + 2)
    m, n = np.meshgrid(x_line, y_line, indexing='xy')

    [m, n] = mesh(Nx, Ny)

    T[0, :] = 2*np.sin(np.pi * m[0, :])-T[1,:]
    T[Nx + 1, :] = -T[Nx,:]
    T[:, Ny + 1] = -T[:,Ny]
    T[:, 0] = -T[:,1]

    m, n = mesh(Nx, Ny)
    x, y = m[1:-1, 1:-1], n[1:-1, 1:-1]
    T_ext = np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)

    T_ext = T_exact(Nx, Ny)
    Error = np.abs(T_ext - T)
    L_1 = np.mean(Error)
    L_2 = np.sqrt(np.mean(Error ** 2))
    L_inf = np.max(Error)

[T_num_g , iter_g, L2_g] = Gauss_Seidel(10,10,1e-10)
    print("Simulation finished.")

if __name__ == "__main__":
    run_simulation()
