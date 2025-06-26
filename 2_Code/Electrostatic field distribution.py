import numpy as np
import matplotlib.pyplot as plt

# 设置网格大小和边界条件
h = 0.1  # 网格间距
L = 1.0  # 区域大小
N = int(L / h) + 1  # 网格节点数
iterations = 1000  # 迭代次数

# 初始化电位矩阵
phi = np.zeros((N, N))

# 设置边界条件（示例：上边界为1，其他边界为0）
phi[-1, :] = 1.0

# Gauss-Seidel 迭代法
for k in range(iterations):
    phi_old = phi.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            phi[i, j] = 0.25 * (phi_old[i-1, j] + phi_old[i+1, j] + phi_old[i, j-1] + phi_old[i, j+1])

# 绘制结果
plt.imshow(phi, origin='lower', cmap='hot')
plt.colorbar(label='electric potential')
plt.title('Electrostatic field distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Electrostatic field distribution')
plt.show()
