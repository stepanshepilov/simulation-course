import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
                             QLabel, QFrame, QSlider, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from scipy.linalg import eig

COLORS = {
    "sunny": "#f1c40f",
    "cloudy": "#bdc3c7",
    "overcast": "#3498db",
    "bg": "#06090f",
    "card": "#10141d",
    "accent": "#00ffcc",
    "accent_hover": "#00cca3"
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
        border: none;
    }}
    QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}
    
    QTableWidget {{ 
        background-color: transparent; 
        border: none; 
        gridline-color: #232d3d; 
    }}
    QTableWidget::item {{
        padding: 5px;
    }}
    QTableWidget::item:selected {{
        background-color: #2c3e50;
        color: {COLORS['accent']};
    }}
    QTableWidget::item:hover {{
        background-color: #1e2a3a;
        color: {COLORS['accent']};
    }}
    QHeaderView::section {{ 
        background-color: #1a1f2b; 
        color: {COLORS['accent']}; 
        border: 1px solid #232d3d; 
        padding: 5px;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: #232d3d;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {COLORS['accent']};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {COLORS['accent_hover']};
    }}
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
        self.is_simulating = False
        
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
        self.matrix_table.setHorizontalHeaderLabels(["Ясно", "Облачно", "Пасмурно"])
        self.matrix_table.setVerticalHeaderLabels(["Из Ясно", "Из Облачно", "Из Пасмурно"])
        
        for i in range(3):
            for j in range(3):
                item = QTableWidgetItem(f"{self.P[i, j]:.3f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.matrix_table.setItem(i, j, item)
        
        self.matrix_table.setFixedHeight(150)
        self.matrix_table.itemChanged.connect(self.on_matrix_changed)
        left_panel.addWidget(self.matrix_table)
        
        self.btn_reset = QPushButton("СБРОСИТЬ МАТРИЦУ")
        self.btn_reset.clicked.connect(self.reset_matrix)
        left_panel.addWidget(self.btn_reset)
        
        self.btn_run = QPushButton("ЗАПУСТИТЬ СИМУЛЯЦИЮ")
        self.btn_run.clicked.connect(self.toggle_sim)
        left_panel.addWidget(self.btn_run)
        
        left_panel.addWidget(QLabel("Скорость симуляции:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 500)
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
        self.draw_all_plots()
    
    def reset_matrix(self):
        self.P = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5]
        ])
        
        self.matrix_table.blockSignals(True)
        for i in range(3):
            for j in range(3):
                self.matrix_table.item(i, j).setText(f"{self.P[i, j]:.3f}")
        self.matrix_table.blockSignals(False)
        
        self.update_stationary()
        self.draw_all_plots()
        
        if self.is_simulating:
            pass
    
    def on_matrix_changed(self, item):
        row = item.row()
        col = item.column()
        
        try:
            new_value = float(item.text())
            
            if new_value < 0:
                QMessageBox.warning(self, "Ошибка", 
                                   "Вероятности не могут быть отрицательными!")
                self.matrix_table.blockSignals(True)
                item.setText(f"{self.P[row, col]:.3f}")
                self.matrix_table.blockSignals(False)
                return
            
            row_sum = 0
            for j in range(3):
                if j == col:
                    row_sum += new_value
                else:
                    val_text = self.matrix_table.item(row, j).text()
                    if val_text:
                        row_sum += float(val_text)
            
            if row_sum > 1.0001:
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Нормировка вероятностей")
                msg_box.setText(f"Сумма вероятностей в строке {row_sum:.3f} превышает 1.\n"
                               f"Хотите ли вы нормировать строку (разделить каждое значение на сумму)?")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                msg_box.setStyleSheet(STYLE)
                
                reply = msg_box.exec()
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.matrix_table.blockSignals(True)
                    for j in range(3):
                        val = float(self.matrix_table.item(row, j).text())
                        normalized_val = val / row_sum
                        self.matrix_table.item(row, j).setText(f"{normalized_val:.3f}")
                        self.P[row, j] = normalized_val
                    self.matrix_table.blockSignals(False)
                else:
                    self.matrix_table.blockSignals(True)
                    item.setText(f"{self.P[row, col]:.3f}")
                    self.matrix_table.blockSignals(False)
                    return
            else:
                self.P[row, col] = new_value
                
                final_sum = 0
                for j in range(3):
                    final_sum += self.P[row, j]
                
                if abs(final_sum - 1.0) > 0.0001 and final_sum < 1:
                    diff = 1.0 - final_sum
                    self.P[row, 2] += diff
                    self.matrix_table.blockSignals(True)
                    self.matrix_table.item(row, 2).setText(f"{self.P[row, 2]:.3f}")
                    self.matrix_table.blockSignals(False)
            
            self.update_stationary()
            self.draw_all_plots()
            
        except ValueError:
            self.matrix_table.blockSignals(True)
            item.setText(f"{self.P[row, col]:.3f}")
            self.matrix_table.blockSignals(False)
            QMessageBox.warning(self, "Ошибка", "Введите корректное число!")
    
    def update_stationary(self):
        # Решение уравнения pi * P = pi  =>  (P^T - I) * pi = 0
        P = self.P.copy()
        S, V = eig(P, left=True, right=False)
        
        pi = V[:, np.isclose(S, 1.0)]

        if pi.size > 0:
            pi = pi[:, 0].real
            self.stationary = pi / pi.sum()
        else:
            self.stationary = np.ones(3) / 3
    
    def toggle_sim(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("ЗАПУСТИТЬ СИМУЛЯЦИЮ")
            self.is_simulating = False
        else:
            self.update_matrix_from_table()
            self.update_stationary()
            interval = max(10, 1100 - self.speed_slider.value())
            self.timer.start(interval)
            self.btn_run.setText("СТОП")
            self.is_simulating = True
    
    def update_matrix_from_table(self):
        for i in range(3):
            row_sum = 0
            for j in range(3):
                try:
                    val = float(self.matrix_table.item(i, j).text())
                    self.P[i, j] = val
                    row_sum += val
                except Exception:
                    pass
            
            if abs(row_sum - 1.0) > 0.0001 and row_sum > 0:
                for j in range(3):
                    self.P[i, j] /= row_sum

                self.matrix_table.blockSignals(True)
                for j in range(3):
                    self.matrix_table.item(i, j).setText(f"{self.P[i, j]:.3f}")
                
                self.matrix_table.blockSignals(False)
    
    def simulation_step(self):
        self.current_state = np.random.choice([0, 1, 2], p=self.P[self.current_state])
        self.history.append(self.current_state)
        self.counts[self.current_state] += 1
        self.steps += 1
        
        if len(self.history) > 50:
            self.history.pop(0)
        
        self.card_state.val.setText(self.states[self.current_state])
        state_colors = [COLORS['sunny'], COLORS['cloudy'], COLORS['overcast']]
        self.card_state.val.setStyleSheet(f"color: {state_colors[self.current_state]};")
        self.card_steps.val.setText(str(self.steps))
        
        self.draw_all_plots()
    
    def draw_all_plots(self):
        self.draw_graph()
        self.draw_time_series()
        self.draw_stationary()
    
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
                        circle = plt.Circle((start[0], start[1] + 0.2), 0.12, 
                                          color=colors[i], fill=False, alpha=p, linewidth=2)
                        self.ax_graph.add_patch(circle)
                        self.ax_graph.text(start[0], start[1] + 0.35, f"{p:.2f}", 
                                         ha='center', fontsize=8, color=colors[i])
                    else:
                        self.ax_graph.annotate("", xy=end, xytext=start,
                                             arrowprops=dict(arrowstyle="->", color=colors[i], 
                                                           lw=p*3, alpha=p, 
                                                           connectionstyle="arc3,rad=0.2"))
                        
                        mid_x = (start[0] + end[0]) / 2
                        mid_y = (start[1] + end[1]) / 2 + 0.1
                        self.ax_graph.text(mid_x, mid_y, f"{p:.2f}", 
                                         ha='center', fontsize=8, color=colors[i])
        
        for i in range(3):
            is_active = (i == self.current_state) and self.is_simulating
            size = 0.35 if is_active else 0.25
            circle = plt.Circle(pos[i], size, color=colors[i], zorder=3, 
                              alpha=0.9 if is_active else 0.6)
            self.ax_graph.add_patch(circle)
            self.ax_graph.text(pos[i][0], pos[i][1], labels[i], 
                             ha='center', va='center', fontweight='bold', 
                             color='white', fontsize=10)
        
        self.ax_graph.set_xlim(-1.8, 1.8)
        self.ax_graph.set_ylim(-0.5, 1.8)
        self.ax_graph.axis('off')
        self.canvas_graph.draw()
    
    def draw_time_series(self):
        self.ax_time.clear()
        self.fig_time.patch.set_facecolor(COLORS['card'])
        self.ax_time.set_facecolor(COLORS['bg'])
        
        if len(self.history) > 0:
            self.ax_time.step(range(len(self.history)), self.history, 
                            where='post', color=COLORS['accent'], linewidth=2)
            
            for i, state in enumerate(self.states):
                self.ax_time.axhline(y=i, color='#2c3e50', linestyle='--', alpha=0.3)
        
        self.ax_time.set_yticks([0, 1, 2])
        self.ax_time.set_yticklabels(self.states)
        self.ax_time.set_xlabel("Шаги симуляции", color='#7f8c8d')
        self.ax_time.set_ylabel("Состояние", color='#7f8c8d')
        self.ax_time.tick_params(colors='#7f8c8d', labelsize=8)
        self.ax_time.grid(True, alpha=0.2, color='#2c3e50')
        self.canvas_time.draw()
    
    def draw_stationary(self):
        self.ax_stat.clear()
        self.fig_stat.patch.set_facecolor(COLORS['card'])
        self.ax_stat.set_facecolor(COLORS['card'])
        
        emp_dist = self.counts / self.steps if self.steps > 0 else np.zeros(3)
        
        x = np.arange(3)
        width = 0.35
        
        bars1 = self.ax_stat.bar(x - width/2, self.stationary, width, 
                                label='Теория', color='#2c3e50', 
                                edgecolor=COLORS['accent'], linewidth=2)
    
        bars2 = self.ax_stat.bar(x + width/2, emp_dist, width, 
                                label='Эмпирика', color=COLORS['accent'], 
                                alpha=0.8, edgecolor='white', linewidth=1)
        
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            self.ax_stat.text(bar1.get_x() + bar1.get_width()/2., height1,
                            f'{height1:.3f}', ha='center', va='bottom', 
                            fontsize=8, color='white')
            if height2 > 0:
                self.ax_stat.text(bar2.get_x() + bar2.get_width()/2., height2,
                                f'{height2:.3f}', ha='center', va='bottom', 
                                fontsize=8, color=COLORS['accent'])
        
        self.ax_stat.set_xticks(x)
        self.ax_stat.set_xticklabels(self.states)
        self.ax_stat.set_ylabel("Вероятность", color='white')
        self.ax_stat.legend(frameon=False, labelcolor='white', loc='upper right')
        self.ax_stat.tick_params(colors='white')
        self.ax_stat.set_ylim(0, 1)
        self.ax_stat.grid(True, alpha=0.2, axis='y', color='#2c3e50')
        self.canvas_stat.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WeatherMarkovApp()
    window.show()
    sys.exit(app.exec())
