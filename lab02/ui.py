import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit

L = 0.4
Tl = 0.0
Tr = 200.0
ro = 7800
c = 460
lmd = 46

h = 0.005
dt = 0.1
Nx = int(L / h)

@jit(nopython=True)
def calculate_next_step(T, alpha, beta, A, B, C, Nx, ro, c, dt):
    alpha[0] = 0.0
    beta[0] = T[0]

    for i in range(1, Nx):
        Fi = - (ro * c / dt) * T[i]
        denom = B - C * alpha[i-1]
        alpha[i] = A / denom
        beta[i] = (C * beta[i-1] - Fi) / denom

    T_new = np.zeros(Nx + 1)
    T_new[0] = T[0]
    T_new[-1] = T[-1]

    for i in range(Nx - 1, 0, -1):
        T_new[i] = alpha[i] * T_new[i+1] + beta[i]
    
    return T_new

x_coords = np.linspace(0, L, Nx + 1)
T = np.zeros(Nx + 1)
T[0] = Tl
T[-1] = Tr

A = lmd / h**2
C = A
B = 2 * lmd / h**2 + ro * c / dt

alpha = np.zeros(Nx + 1)
beta = np.zeros(Nx + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(hspace=0.4)

line, = ax1.plot(x_coords, T, color='red', lw=2, label='Температурный профиль')
ax1.set_xlim(0, L)
ax1.set_ylim(-10, Tr + 20)
ax1.set_xlabel('Длина пластины, м')
ax1.set_ylabel('Температура, °C')
ax1.grid(True, ls='--')
ax1.legend()

im = ax2.imshow([T], aspect='auto', cmap='hot', extent=[0, L, 0, 0.1], vmin=0, vmax=Tr)
ax2.set_yticks([])
ax2.set_xlabel('Длина пластины, м')
ax2.set_title('Визуализация нагрева пластины')

time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12, fontweight='bold')

def update(frame):
    global T

    steps_per_frame = 5 
    for _ in range(steps_per_frame):
        T = calculate_next_step(T, alpha, beta, A, B, C, Nx, ro, c, dt)
    
    current_time = frame * dt * steps_per_frame
    
    line.set_ydata(T)
    im.set_data([T])
    time_text.set_text(f'Время: {current_time:.1f} сек')
    
    return line, im, time_text

ani = FuncAnimation(fig, update, frames=1000, interval=20, blit=True)

plt.show()
