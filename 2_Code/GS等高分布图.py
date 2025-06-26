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


[T_num_g , iter_g, L2_g] = Gauss_Seidel(10,10,1e-10)
print(np.round(T_num_g,4))
[m,n] = mesh(10,10)
plt.contourf(m[1:11, 1:11],n[1:11, 1:11],T_num_g,50, cmap='rainbow')
plt.title('Numerical Solution')
plt.colorbar()
plt.savefig('Numerical Solution')
plt.show()
