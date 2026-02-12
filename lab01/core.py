import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math

class FlightSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделирование полёта тела в атмосфере")
        self.g = 9.81
        self.results_table = {}
        self.animation = None
        self.is_animating = False
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        left = ttk.LabelFrame(root, text="Параметры")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.inputs = {}
        for label, default, key in [
            ("Начальная скорость (м/с):", "50", "v0"),
            ("Угол бросания (град):", "45", "angle"),
            ("Масса тела (кг):", "1.0", "m"),
            ("Коэфф. сопротивления (k):", "0.02", "k"),
            ("Шаг моделирования (dt, с):", "0.05", "dt")
        ]:
            f = ttk.Frame(left)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label).pack(side=tk.LEFT)
            e = ttk.Entry(f, width=10)
            e.insert(0, default)
            e.pack(side=tk.RIGHT)
            self.inputs[key] = e
        
        for text, cmd in [("Запуск (Анимация)", self.animate),
                          ("Расчет (График)", self.calculate),
                          ("Запустить все шаги", self.run_all_steps),
                          ("Очистить график", self.clear_plot)]:
            ttk.Button(left, text=text, command=cmd).pack(pady=5, fill=tk.X)
        
        self.result_text = tk.Text(left, height=15, width=35, font=("Consolas", 9))
        self.result_text.pack(pady=10)
        
        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title("Траектория полёта")
        self.ax.set_xlabel("Дальность (x), м")
        self.ax.set_ylabel("Высота (y), м")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def get_params(self):
        try:
            return {k: float(self.inputs[k].get()) for k in ["v0", "angle", "m", "k", "dt"]}
        except:
            self.result_text.insert(tk.END, "Ошибка ввода!\n")
            return None
    
    def simulate(self, v0, angle, m, k, dt):
        x, y = 0.0, 0.0
        rad = math.radians(angle)
        vx, vy = v0 * math.cos(rad), v0 * math.sin(rad)
        xs, ys, max_h = [x], [y], 0.0
        
        while y >= 0:
            v = math.sqrt(vx**2 + vy**2)
            ax = -(k / m) * v * vx
            ay = -self.g - (k / m) * v * vy
            
            x += vx * dt
            y += vy * dt
            vx += ax * dt
            vy += ay * dt
            
            if y >= 0:
                xs.append(x)
                ys.append(y)
                max_h = max(max_h, y)
        
        return xs, ys, x, max_h, math.sqrt(vx**2 + vy**2)
    
    def update_plot(self, xs, ys, dt, fx, mh, fv):
        self.results_table[dt] = {'distance': fx, 'max_height': mh, 'final_speed': fv}
        self.ax.plot(xs, ys, label=f"dt={dt}")
        self.ax.legend()
        self.canvas.draw()
        self.show_table()
    
    def show_table(self):
        self.result_text.delete(1.0, tk.END)
        if not self.results_table:
            return
        
        dts = sorted(self.results_table.keys())
        sep = "|" + "-" * 15 + "|" + "|".join("-" * 10 for _ in dts) + "|"
        
        hdr = "| Time step (dt) " + "|".join(f" {dt:.3f} " for dt in dts) + " |"
        txt = hdr + "\n" + sep + "\n"
        
        for metric, key in [("Distance (m)", "distance"), ("Max height (m)", "max_height"), ("Speed (m/s)", "final_speed")]:
            row = f"| {metric:<13} " + "|".join(f" {self.results_table[dt][key]:>8.2f} " for dt in dts) + " |"
            txt += row + "\n" + sep + "\n"
        
        self.result_text.insert(tk.END, txt)
    
    def calculate(self):
        params = self.get_params()
        if not params:
            return
        xs, ys, fx, mh, fv = self.simulate(**params)
        self.update_plot(xs, ys, params['dt'], fx, mh, fv)
    
    def animate(self):
        if self.is_animating:
            return
        params = self.get_params()
        if not params:
            return
        
        self.is_animating = True
        xs, ys, fx, mh, fv = self.simulate(**params)
        
        self.ax.clear()
        self.ax.set_xlim(-5, max(fx + 10, 50))
        self.ax.set_ylim(-5, max(mh + 10, 50))
        self.ax.grid(True, alpha=0.3)
        self.ax.plot(xs, ys, 'b--', alpha=0.3, linewidth=1)
        
        line, = self.ax.plot([], [], 'b-', linewidth=2)
        ball, = self.ax.plot([], [], 'ro', markersize=8)
        
        def init():
            line.set_data([], [])
            ball.set_data([], [])
            return line, ball
        
        def frame_update(i):
            if i < len(xs):
                line.set_data(xs[:i+1], ys[:i+1])
                ball.set_data([xs[i]], [ys[i]])
            return line, ball
        
        self.animation = FuncAnimation(self.figure, frame_update, init_func=init,
                                       frames=len(xs), interval=50, blit=True, repeat=True)
        self.ax.legend(loc='upper right')
        self.canvas.draw()
    
    def run_all_steps(self):
        params = self.get_params()
        if not params:
            return
        
        self.results_table.clear()
        self.clear_plot()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Расчёты для всех шагов...\n\n")
        
        for dt in [1, 0.1, 0.01, 0.001, 0.0001]:
            params['dt'] = dt
            xs, ys, fx, mh, fv = self.simulate(**params)
            self.results_table[dt] = {'distance': fx, 'max_height': mh, 'final_speed': fv}
            self.ax.plot(xs, ys, label=f"dt={dt}")
            self.result_text.insert(tk.END, f"✓ dt={dt}\n")
        
        self.ax.legend()
        self.canvas.draw()
        self.show_table()
        self.result_text.insert(tk.END, "\nЗавершено. Результаты в консоли.\n")
    
    def clear_plot(self):
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.ax.clear()
        self.ax.set_title("Траектория полёта")
        self.ax.set_xlabel("Дальность (x), м")
        self.ax.set_ylabel("Высота (y), м")
        self.ax.grid(True)
        self.result_text.delete(1.0, tk.END)
        self.canvas.draw()
        self.is_animating = False
    
    def print_table(self):
        if not self.results_table:
            return
        
        dts = sorted(self.results_table.keys())
        sep = "|" + "-" * 22 + "|" + "|".join("-" * 10 for _ in dts) + "|"
        
        print("Результаты моделирования для разных шагов (dt):")

        hdr = "| Шаг моделирования, с "
        for dt in dts:
            fmt = ".0f" if dt >= 1 else ".4f" if dt < 0.01 else ".2f"
            hdr += f"| {dt:{fmt}} "
        hdr += "|"
        print(hdr)
        print(sep)
        
        for metric, key in [("Дальность полёта, м", "distance"),
                            ("Максимальная высота, м", "max_height"),
                            ("Скорость в конечной точке, м/с", "final_speed")]:
            row = f"| {metric:<21} "
            for dt in dts:
                row += f"| {self.results_table[dt][key]:>8.2f} "
            row += "|"
            print(row)
            print(sep)
    
    def on_closing(self):
        self.print_table()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FlightSimulatorApp(root)
    root.mainloop()
