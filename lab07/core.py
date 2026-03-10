import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
                             QLabel, QFrame, QSlider)
from PyQt6.QtCore import Qt, QTimer
from scipy.linalg import eig

COLORS = {
    "sunny": "#f1c40f",
    "cloudy": "#bdc3c7",
    "overcast": "#3498db",
    "bg": "#06090f",
    "card": "#10141d",
    "accent": "#00ffcc"
}

STYLE = f"""
    QMainWindow {{ background-color: {COLORS['bg']}; }}
    QWidget {{ background-color: {COLORS['bg']}; color: #ecf0f1; font-family: 'Segoe UI'; }}
    QFrame#Card {{ 
        background-color: {COLORS['card']}; 
        border: 1px solid #232d3d; 
        border-radius: 15px; 
    }}
    QLabel#Title {{ color: {COLORS['accent']}; font-weight: bold; font-size: 16px; }}
    QLabel#StatLabel {{ color: #7f8c8d; font-size: 11px; text-transform: uppercase; }}
    QLabel#StatValue {{ color: #ffffff; font-size: 18px; font-weight: bold; font-family: 'Consolas'; }}
    
    QPushButton {{ 
        background-color: {COLORS['accent']}; color: #000; border-radius: 8px; 
        padding: 10px; font-weight: bold; font-size: 13px; 
    }}
    QPushButton:hover {{ background-color: #00cca3; }}
    
    QTableWidget {{ 
        background-color: transparent; border: none; gridline-color: #232d3d; 
    }}
    QHeaderView::section {{ background-color: #1a1f2b; color: {COLORS['accent']}; border: 1px solid #232d3d; }}
"""

class DashboardCard(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        lbl = QLabel(title.upper())
        lbl.setObjectName("StatLabel")
        self.val = QLabel("---")
        self.val.setObjectName("StatValue")
        layout.addWidget(lbl)
        layout.addWidget(self.val)

class WeatherMarkovApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Метео-Симулятор: Цепи Маркова")
        self.resize(1400, 900)
        self.setStyleSheet(STYLE)
        
        self.states = ["Ясно", "Облачно", "Пасмурно"]
        self.current_state = 0
        self.history = []
        self.counts = np.zeros(3)
        self.steps = 0
        
        self.P = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5]
        ])
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        left_panel = QVBoxLayout()
        
        title = QLabel("WEATHER CONTROL")
        title.setObjectName("Title")
        left_panel.addWidget(title)
        
        left_panel.addWidget(QLabel("Матрица переходов P:"))
        self.matrix_table = QTableWidget(3, 3)
        self.matrix_table.setHorizontalHeaderLabels(["Я", "О", "П"])
        self.matrix_table.setVerticalHeaderLabels(["Я", "О", "П"])
        for i in range(3):
            for j in range(3):
                self.matrix_table.setItem(i, j, QTableWidgetItem(str(self.P[i, j])))
        self.matrix_table.setFixedHeight(130)
        left_panel.addWidget(self.matrix_table)
        
        self.btn_run = QPushButton("ЗАПУСТИТЬ СИМУЛЯЦИЮ")
        self.btn_run.clicked.connect(self.toggle_sim)
        left_panel.addWidget(self.btn_run)
        
        left_panel.addWidget(QLabel("Скорость симуляции:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 1000)
        self.speed_slider.setValue(100)
        left_panel.addWidget(self.speed_slider)
        
        self.card_state = DashboardCard("Текущее состояние")
        self.card_steps = DashboardCard("Прошло дней")
        left_panel.addWidget(self.card_state)
        left_panel.addWidget(self.card_steps)
        left_panel.addStretch()
        
        main_layout.addLayout(left_panel, 1)
        
        mid_panel = QVBoxLayout()
        self.fig_graph, self.ax_graph = plt.subplots(figsize=(5, 4))
        self.canvas_graph = FigureCanvas(self.fig_graph)
        mid_panel.addWidget(self.canvas_graph)
        
        self.fig_time, self.ax_time = plt.subplots(figsize=(5, 3))
        self.canvas_time = FigureCanvas(self.fig_time)
        mid_panel.addWidget(self.canvas_time)
        
        main_layout.addLayout(mid_panel, 2)
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("СТАЦИОНАРНОЕ РАСПРЕДЕЛЕНИЕ (ТЕОРИЯ VS ЭМПИРИКА)"))
        
        self.fig_stat, self.ax_stat = plt.subplots(figsize=(4, 6))
        self.canvas_stat = FigureCanvas(self.fig_stat)
        right_panel.addWidget(self.canvas_stat)
        
        main_layout.addLayout(right_panel, 1)
        
        self.update_stationary()
        self.draw_graph()

    def get_matrix_from_table(self):
        try:
            new_p = np.zeros((3, 3))
            for i in range(3):
                row_sum = 0
                for j in range(3):
                    val = float(self.matrix_table.item(i, j).text())
                    new_p[i, j] = val
                    row_sum += val
                new_p[i] /= row_sum
            return new_p
        except Exception:
            return self.P

    def update_stationary(self):
        # Решение уравнения pi * P = pi  =>  (P^T - I) * pi = 0
        P = self.get_matrix_from_table()
        S, V = eig(P, left=True, right=False)

        pi = V[:, np.isclose(S, 1.0)]
        pi = pi[:, 0].real
        
        self.stationary = pi / pi.sum()

    def toggle_sim(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("ЗАПУСТИТЬ СИМУЛЯЦИЮ")
        else:
            self.P = self.get_matrix_from_table()
            self.update_stationary()
            self.timer.start(1100 - self.speed_slider.value())
            self.btn_run.setText("СТОП")

    def simulation_step(self):
        self.current_state = np.random.choice([0, 1, 2], p=self.P[self.current_state])
        self.history.append(self.current_state)
        self.counts[self.current_state] += 1
        self.steps += 1
        
        if len(self.history) > 50:
            self.history.pop(0)
        
        self.card_state.val.setText(self.states[self.current_state])
        self.card_state.val.setStyleSheet(f"color: {list(COLORS.values())[self.current_state]}")
        self.card_steps.val.setText(str(self.steps))
        
        self.draw_plots()

    def draw_graph(self):
        self.ax_graph.clear()
        self.fig_graph.patch.set_facecolor(COLORS['card'])
        self.ax_graph.set_facecolor(COLORS['card'])
        
        pos = {0: (0, 1), 1: (1, 0), 2: (-1, 0)}
        labels = self.states
        colors = [COLORS['sunny'], COLORS['cloudy'], COLORS['overcast']]
        
        for i in range(3):
            for j in range(3):
                p = self.P[i, j]
                if p > 0.05:
                    start, end = pos[i], pos[j]
                    if i == j:
                        circle = plt.Circle((start[0], start[1]+0.15), 0.1, color=colors[i], fill=False, alpha=p)
                        self.ax_graph.add_patch(circle)
                    else:
                        self.ax_graph.annotate("", xy=end, xytext=start,
                                               arrowprops=dict(arrowstyle="->", color=colors[i], 
                                                               lw=p*5, alpha=p, connectionstyle="arc3,rad=.2"))
        
        for i in range(3):
            is_active = (i == self.current_state)
            size = 0.3 if is_active else 0.2
            circle = plt.Circle(pos[i], size, color=colors[i], zorder=3, alpha=0.8 if is_active else 0.4)
            self.ax_graph.add_patch(circle)
            self.ax_graph.text(pos[i][0], pos[i][1], labels[i], ha='center', va='center', fontweight='bold', color='white')

        self.ax_graph.set_xlim(-1.5, 1.5)
        self.ax_graph.set_ylim(-0.5, 1.5)
        self.ax_graph.axis('off')
        self.canvas_graph.draw()

    def draw_plots(self):
        self.ax_time.clear()
        self.fig_time.patch.set_facecolor(COLORS['card'])
        self.ax_time.set_facecolor(COLORS['bg'])
        self.ax_time.step(range(len(self.history)), self.history, where='post', color=COLORS['accent'])
        self.ax_time.set_yticks([0, 1, 2])
        self.ax_time.set_yticklabels(self.states)
        self.ax_time.tick_params(colors='#7f8c8d', labelsize=8)
        self.canvas_time.draw()
        
        self.ax_stat.clear()
        self.fig_stat.patch.set_facecolor(COLORS['card'])
        self.ax_stat.set_facecolor(COLORS['card'])
        
        emp_dist = self.counts / self.steps if self.steps > 0 else np.zeros(3)
        
        x = np.arange(3)
        width = 0.35
        self.ax_stat.bar(x - width/2, self.stationary, width, label='Теория', color='#2c3e50', edgecolor=COLORS['accent'])
        self.ax_stat.bar(x + width/2, emp_dist, width, label='Эмпирика', color=COLORS['accent'], alpha=0.7)
        
        self.ax_stat.set_xticks(x)
        self.ax_stat.set_xticklabels(self.states)
        self.ax_stat.legend(frameon=False, labelcolor='white')
        self.ax_stat.tick_params(colors='white')
        self.canvas_stat.draw()
        
        self.draw_graph()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WeatherMarkovApp()
    window.show()
    sys.exit(app.exec())
