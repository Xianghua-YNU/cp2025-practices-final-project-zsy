
import numpy as np

def jacobi_method(matrix, threshold=1e-5, max_iterations=1000):
    rows, cols = matrix.shape
    current_matrix = matrix.copy()
    next_matrix = np.zeros_like(matrix)
    iteration = 0

    while True:
        max_diff = 0
        for i in range(rows):
            for j in range(cols):
                # 获取相邻点的值，注意边界条件
                neighbors = []
                if i > 0:
                    neighbors.append(current_matrix[i - 1, j])
                if i < rows - 1:
                    neighbors.append(current_matrix[i + 1, j])
                if j > 0:
                    neighbors.append(current_matrix[i, j - 1])
                if j < cols - 1:
                    neighbors.append(current_matrix[i, j + 1])

                # 计算相邻点的平均值
                next_matrix[i, j] = sum(neighbors) / len(neighbors)

                # 计算增量
                max_diff = max(max_diff, abs(next_matrix[i, j] - current_matrix[i, j]))

        # 更新矩阵
        current_matrix[:] = next_matrix

        iteration += 1
        if max_diff < threshold or iteration >= max_iterations:
            break

    return current_matrix, iteration

# 示例矩阵
initial_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=float)

result_matrix, iterations = jacobi_method(initial_matrix)
print("结果矩阵：\n", result_matrix)
print("迭代次数：", iterations)


