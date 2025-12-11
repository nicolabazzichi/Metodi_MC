import numpy as np
import matplotlib.pyplot as plt


L = 2*np.pi
T =5 * 2 * np.pi
#nu = 1e-3

nt = 25000
nx = 100

dt = T/nt
dx = L/nx

k = nx/8
nu = 1 / ((k ** 2.0) * T)
print(nu)

lim = L - L/nx
P = nu * (dt/dx**2)
C = dt / dx

if P > 0.5:
    print(f'Parametro P sbagliato, P = {P:.4f}, cambiare griglia o nu')
else:
    print(P)

if C > 0.1:
    print(f'Parametro C sbagliato, C = {C:.4f}, cambiare griglia')
else:
    print(C)


x = np.linspace(0, lim, nx)

def func(x):
    return np.sin(x)

f = func(x)
plt.figure('Evoluzione Temporale')
plt.plot(x, func(x), label='Funzione di partenza', c='black')


u_old = f.copy()       
u_new = np.zeros(nx)
for i in range(nt):

    uc = np.zeros(nx)
    uii = np.zeros(nx)
    for k in range(nx):
        ip = (k + 1) % nx
        im = (k - 1) % nx
        uc[k] = (u_old[ip] - u_old[im]) / (2*dx)
        uii[k] = (u_old[ip] - 2 * u_old[k] + u_old[im]) / (dx**2.0)

    u_new = u_old + dt * (-uc + nu * uii) #con il più si sposta indietro mentre con il meno si sposta in avanti
    
    u_old = u_new.copy()
    if i == nt*0.2:
        plt.plot(x, u_old, label='Spostamento temporale 1', c='darkgreen')
    elif i == nt*0.5:
        plt.plot(x, u_old, label='Spostamento temporale 2', c='yellowgreen')
    elif i == nt*0.8:
        plt.plot(x, u_old, label='Spostamento temporale 3', c='palegreen')
        plt.legend()
plt.show()
