import numpy as np
from numba import jit
import time

@jit(nopython=True)
def simulate(Tl, Tr, L, h, total_time=30, dt=0.01):
    ro = 7800
    c = 460
    lmd = 46

    Nx = int(L / h)
    Nt = int(total_time / dt)

    T = np.zeros(Nx+1)
    T[0] = Tl
    T[-1] = Tr

    for i in range(1, Nx):
        T[i] = 0.0

    A = lmd / h**2
    C = A
    B = 2*lmd / h**2 + ro*c / dt

    alpha = np.zeros(Nx+1)
    beta = np.zeros(Nx+1)

    for _ in range(Nt):
        alpha[0] = 0.0
        beta[0] = Tl

        for i in range(1, Nx):
            Fi = - (ro*c / dt) * T[i]
            denom = B - C * alpha[i-1]
            alpha[i] = A / denom
            beta[i] = (C * beta[i-1] - Fi) / denom

        T_new = np.zeros(Nx+1)
        T_new[-1] = Tr

        for i in range(Nx-1, 0, -1):
            T_new[i] = alpha[i] * T_new[i+1] + beta[i]

        for i in range(Nx+1):
            T[i] = T_new[i]

    return T, T[Nx//2]

if __name__ == "__main__":
    dts = [0.1, 0.01, 0.001, 0.0001]
    hs = [0.1, 0.01, 0.001, 0.0001]

    L = 0.4
    Tl = 0
    Tr = 200
    total_time = 600.0

    # вывод таблички
    n_rows = len(dts)
    n_cols = len(hs)
    T_table = [[0.0]*n_cols for _ in range(n_rows)]
    time_table = [[0.0]*n_cols for _ in range(n_rows)]

    for i, dt in enumerate(dts):
        for j, h in enumerate(hs):
            print(f"Расчёт: dt={dt}, h={h}...", end=" ", flush=True)

            start = time.time()
            _, T_center = simulate(Tl, Tr, L, h, total_time, dt)
            elapsed = time.time() - start

            T_table[i][j] = T_center
            time_table[i][j] = elapsed

            print(f"T={T_center:.2f}, время={elapsed:.3f} с")

    print("\n" + "="*60)
    print("Температура в центре пластины через 2 с, °C")
    print("="*60)
    header = "dt\\h   |"
    for h in hs:
        header += f" {h:>8}"
    print(header)
    print("-" * (10 + 9 * len(hs)))
    for i, dt in enumerate(dts):
        line = f"{dt:<6} |"
        for j in range(n_cols):
            line += f" {T_table[i][j]:>8.2f}"
        print(line)
    
    print("\n" + "="*60)
    print("Время расчёта, с")
    print("="*60)
    header = "dt\\h   |"
    for h in hs:
        header += f" {h:>8}"
    print(header)
    print("-" * (10 + 9 * len(hs)))
    for i, dt in enumerate(dts):
        line = f"{dt:<6} |"
        for j in range(n_cols):
            line += f" {time_table[i][j]:>8.4f}"
        print(line)
