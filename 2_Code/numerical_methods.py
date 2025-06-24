"""
核心数值算法模块：包含各种数值方法实现。
"""
import numpy as np

def solve_ode(initial_conditions, time_points):
    """
    示例：常微分方程求解器。
    """
    print("Solving ODE...")
    # 这里是数值求解算法的实现
    # 例如：欧拉法、龙格-库塔法等
    solution = np.zeros((len(time_points), len(initial_conditions)))
    solution[0] = initial_conditions
    # 假设一个简单的增长模型
    # for i in range(1, len(time_points)):
    #     dt = time_points[i] - time_points[i-1]
    #     solution[i] = solution[i-1] + dt * solution[i-1] * 0.1 # 简单的指数增长
    return solution

# 可以添加更多数值方法，如积分、求根、矩阵运算等
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

def Jacobi(A,b,x0,TOL):
    D=np.diag(np.diag(A))
    U=-np.triu(A,1)
    L=-np.tril(A,-1)
    Tj=np.linalg.inv(D)@(L+U)
    cj=np.linalg.inv(D)@b
    x=Tj@x0+cj
    itr = 1
    max_itr=50
    while (max(abs(x-x0))/max(abs(x))>TOL) and (itr<max_itr):
        x0=x
        x=Tj@x0+cj
        itr += 1
    return [Tj,x,itr]

def Gauss(A,b,x0,TOL):
    D=np.diag(np.diag(A))
    U=-np.triu(A,1)
    L=-np.tril(A,-1)

    Tg=(np.linalg.inv(D-L))@U
    cg=np.linalg.inv(D-L)@b
    x=Tg@x0+cg
    itr = 1
    max_itr=50
    while (max(abs(x-x0))/max(abs(x))>TOL) and (itr<max_itr):
        x0=x
        x=Tg@x0+cg
        itr += 1
    return [Tg,x,itr]

