"""
可视化函数模块：用于绘制模拟结果和分析数据。
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_results(data):
    """
    示例：绘制模拟结果。
    """
    print("Plotting results...")
    # plt.figure()
    # plt.plot(data)
    # plt.title("Simulation Results")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.grid(True)
    # plt.show()
    print("Plotting finished.")

# 可以添加更多绘图函数，如散点图、三维图等
等高线图：
[T_num_g , iter_g, L2_g] = Gauss_Seidel(10,10,1e-10)
print(np.round(T_num_g,4))
[m,n] = mesh(10,10)
plt.contourf(m[1:11, 1:11],n[1:11, 1:11],T_num_g,50, cmap='rainbow')
plt.title('Numerical Solution')
plt.colorbar()
plt.show()


不同网格大小下运行时间：
start_time_g =timeit.default_timer()
[T_num_g , iter_g, L2_g] = Gauss_Seidel(10,10,1e-12)
end_time_g =timeit.default_timer()
CPU_Time_g = end_time_g - start_time_g
print(f"CPU Time 10*10: {CPU_Time_g} seconds ,","Number of iterations:",iter_g[-1])

start_time_g2 =timeit.default_timer()
[T_num_g2 , iter_g2, L2_g2] = Gauss_Seidel(20,20,1e-12)
end_time_g2 =timeit.default_timer()
CPU_Time_g2 = end_time_g2 - start_time_g2
print(f"CPU Time 20*20: {CPU_Time_g2} seconds ,","Number of iterations:",iter_g2[-1])

start_time_g3 =timeit.default_timer()
[T_num_g3 , iter_g3, L2_g3] = Gauss_Seidel(30,30,1e-12)
end_time_g3 =timeit.default_timer()
CPU_Time_g3 = end_time_g3 - start_time_g3
print(f"CPU Time 30*30: {CPU_Time_g3} seconds ,","Number of iterations:",iter_g3[-1])

start_time_g4 =timeit.default_timer()
[T_num_g4 , iter_g4, L2_g4] = Gauss_Seidel(40,40,1e-12)
end_time_g4 =timeit.default_timer()
CPU_Time_g4 = end_time_g4 - start_time_g4
print(f"CPU Time 40*40: {CPU_Time_g4} seconds ,","Number of iterations:",iter_g4[-1])


plt.plot(iter_g, L2_g, '-.', label = '10*10')
plt.plot(iter_g2, L2_g2, ':', label = '20*20')
plt.plot(iter_g3, L2_g3, '-.', label = '30*30')
plt.plot(iter_g4, L2_g4, '--', label = '40*40')
plt.xlabel('Iteration number')
plt.ylabel('L2')
plt.legend()
plt.show()
