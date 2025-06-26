import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve



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
def Gauss_Seidel(Nx, Ny, l_2):
    T = np.zeros([Nx + 2, Ny + 2])
    dx, dy = 1 / Nx, 1 / Ny
    L = []
    iteration = []

    L2 = 1
    counter = 0
    while L2 > l_2:

        boundary_condition(T, Nx, Ny)
        T_0 = T.copy()

        for m in range(1,Nx+1):
            for n in range(1, Ny+1):
                T[m,n] = 0.5*(dy**2/((dy**2+dx**2))*(T_0[m,n+1]+T[m,n-1])+dx**2/((dy**2+dx**2))*(T[m-1,n]+T_0[m+1,n]))

        Error = np.abs(T_0[1:Nx+1, 1:Ny+1] - T[1:Nx+1, 1:Ny+1])
        L2 = np.sqrt(np.mean(Error**2))
        L.append(L2)
        counter += 1
        iteration.append(counter)
    L = np.array(L)
    iteration = np.array(iteration)
    return T[1:Nx+1, 1:Ny+1], iteration, L

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
