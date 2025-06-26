import matplotlib.pyplot as plt
import numpy as np
a=1
ds=4e-2

N=int(a/ds)
print(N)
r=[]
for i in range(N):
    lt=[]
    for j in range(N):
        lt.append(0.0)
    r.append(lt)

V=r
'''
V0=5

for i in range(N):
    V[i][0]=V0
    V[i][N-1]=V0
    '''

V1=2
V2=5
V3=4
V4=3

for i in range(N):
    V[i][0]=V1
    V[i][N-1]=V2
    V[0][i]=V3
    V[N-1][i]=V4

#for i in V:
#    print(i)
print('''
iterative relaxation in progress
     ''')
S=[] 
nit=[]
nl=450
for k in range(nl):
    s=0
    for i in range(1,N-1):
        for j in range(1,N-1):
            V[i][j]=(V[i][j+1]+V[i][j-1]+V[i+1][j]+V[i-1][j])/(4)
    
    for i in range(1,N-1):
        for j in range(1,N-1):
            s+=(V[i+1][j]-V[i][j])**2+(V[i][j+1]-V[i][j])**2
    print(s)
    S.append(s)
    nit.append(k+1)

for i in range(1,N-1):
    for j in range(1,N-1):
        V[i][j]=round(V[i][j],4)
#for i in V:
#    print(i)
 
L=[]
for i in range(N):
    lt=[]
    for j in range(N):
        lt.append(0.0)
    L.append(lt)
#for i in L:
#    print(i)

for i in range(1,N-1):
    for j in range(1,N-1):
        L[i][j] = 4*V[i][j]-(V[i+1][j]+V[i-1][j]+V[i][j+1]+V[i][j-1])

#for i in V:
#    print(i)
 
    
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
 
xs=[]
ys=[]
zs=[]
for i in range(N): 
    for j in range(N):
        xs.append(i*ds)
        ys.append(j*ds)
        zs.append(V[i][j])
        
ax.scatter(xs, ys, zs)




ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z (V)')

plt.show()

xs=[]
ys1=[]
ys2=[]
for i in range(N):
    xs.append(i*ds)
    ys1.append(V[int(N/2)][i])
    ys2.append(V[i][int(N/2)])

plt.plot(nit,S)
plt.xlabel('Number of iterations')
plt.ylabel('Convergence function value')
#plt.xlabel('Distance')
#plt.ylabel('Electrostatic Potential (V)')
