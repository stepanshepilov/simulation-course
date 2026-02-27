import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.stats import chi2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
                             QComboBox, QLabel, QHeaderView, QSpinBox, QTextEdit)
from abc import ABC, abstractmethod

DARK_STYLE = """
    QMainWindow { background-color: #000000; }
    QWidget { background-color: #000000; color: #FFFFFF; font-family: 'Consolas', monospace; }
    QTableWidget { background-color: #121212; border: 1px solid #FF8C00; color: white; gridline-color: #333; }
    QHeaderView::section { background-color: #1e1e1e; color: #FF8C00; border: 1px solid #333; }
    QPushButton { background-color: #1e1e1e; color: #FF8C00; border: 1px solid #FF8C00; border-radius: 4px; padding: 10px; font-weight: bold; }
    QPushButton:hover { background-color: #FF8C00; color: #000000; }
    QSpinBox { background-color: #1e1e1e; border: 1px solid #FF8C00; color: white; padding: 5px; }
    QTextEdit { background-color: #0a0a0a; border: 1px solid #444; color: #00FF00; font-size: 12px; }
    QComboBox { background-color: #1e1e1e; border: 1px solid #FF8C00; color: white; padding: 5px; }
"""

class DistributionStrategy(ABC):
    @abstractmethod
    def get_data(self): pass

class UniformStrategy(DistributionStrategy):
    def get_data(self): return [(i, 0.2) for i in range(1, 6)]

class BernoulliStrategy(DistributionStrategy):
    def get_data(self): return [(0, 0.4), (1, 0.6)]

class LabWorkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Имитационное моделирование СВ - Лабораторная работа")
        self.resize(1300, 800)
        self.setStyleSheet(DARK_STYLE)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        
        left_panel.addWidget(QLabel("1. ПАРАМЕТРЫ ВЫБОРКИ"))
        self.sample_count = QSpinBox()
        self.sample_count.setRange(10, 100000)
        self.sample_count.setValue(1000)
        self.sample_count.setSingleStep(100)
        left_panel.addWidget(self.sample_count)

        left_panel.addSpacing(20)
        left_panel.addWidget(QLabel("2. РЯД РАСПРЕДЕЛЕНИЯ"))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["X", "P"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        left_panel.addWidget(self.table)

        self.strat_combo = QComboBox()
        self.strategies = {"Пресеты...": None, "Равномерное": UniformStrategy(), "Бернулли": BernoulliStrategy()}
        self.strat_combo.addItems(self.strategies.keys())
        self.strat_combo.currentIndexChanged.connect(self.apply_preset)
        left_panel.addWidget(self.strat_combo)

        self.btn_discrete = QPushButton("СЭМПЛИТЬ ДИСКРЕТНУЮ СВ")
        self.btn_discrete.clicked.connect(self.run_discrete)
        left_panel.addWidget(self.btn_discrete)

        left_panel.addSpacing(20)
        left_panel.addWidget(QLabel("3. НОРМАЛЬНАЯ СВ"))
        self.btn_normal = QPushButton("ГЕНЕРИРОВАТЬ НОРМАЛЬНУЮ СВ")
        self.btn_normal.setStyleSheet("border-color: #00BFFF; color: #00BFFF;")
        self.btn_normal.clicked.connect(self.run_normal)
        left_panel.addWidget(self.btn_normal)

        main_layout.addLayout(left_panel, 1)

        mid_panel = QVBoxLayout()
        mid_panel.addWidget(QLabel("СТАТИСТИЧЕСКИЙ ОТЧЕТ"))
        self.stats_output = QTextEdit()
        self.stats_output.setReadOnly(True)
        mid_panel.addWidget(self.stats_output)
        main_layout.addLayout(mid_panel, 1)

        self.figure, self.ax = plt.subplots()
        self.figure.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#0a0a0a')
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, 2)

    def apply_preset(self):
        strat = self.strategies[self.strat_combo.currentText()]
        if strat:
            self.table.setRowCount(0)
            for x, p in strat.get_data():
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(x)))
                self.table.setItem(r, 1, QTableWidgetItem(str(p)))

    def run_discrete(self):
        try:
            x_th = [float(self.table.item(i, 0).text()) for i in range(self.table.rowCount())]
            p_th = [float(self.table.item(i, 1).text()) for i in range(self.table.rowCount())]
            N = self.sample_count.value()

            samples = self.inverse_transform_sampling(x_th, p_th, N)
            
            self.calculate_discrete_stats(x_th, p_th, samples, N)
            
            self.plot_discrete(x_th, samples)
        except Exception as e:
            self.stats_output.setText(f"Ошибка: {e}")

    def inverse_transform_sampling(self, x_list, p_list, n):
        p_list = [p / sum(p_list) for p in p_list]
        
        cdf = np.cumsum(p_list)
        samples = []
        for _ in range(n):
            u = random.random()
            for i, c in enumerate(cdf):
                if u <= c:
                    samples.append(x_list[i])
                    break
        
        return samples

    def calculate_discrete_stats(self, x_th, p_th, samples, N):
        m_th = sum(x * p for x, p in zip(x_th, p_th))
        d_th = sum((x**2) * p for x, p in zip(x_th, p_th)) - m_th**2

        unique, counts = np.unique(samples, return_counts=True)
        freq_map = dict(zip(unique, counts))
        p_emp = [freq_map.get(x, 0) / N for x in x_th]
        
        m_emp = np.mean(samples)
        d_emp = np.var(samples)

        err_m = abs(m_th - m_emp) / (m_th if m_th != 0 else 1)
        err_d = abs(d_th - d_emp) / (d_th if d_th != 0 else 1)

        chi_val = 0
        for i in range(len(x_th)):
            e_i = N * p_th[i]
            o_i = freq_map.get(x_th[i], 0)
            chi_val += ((o_i - e_i)**2) / e_i
        
        df = len(x_th) - 1
        chi_crit = chi2.ppf(0.95, df)

        report = f""">>> ОТЧЕТ ПО ДИСКРЕТНОЙ СВ (N={N})
                    Мат. ожидание (теор): {m_th:.4f}
                    Мат. ожидание (эмп) : {m_emp:.4f}
                    Погрешность M       : {err_m:.2%}

                    Дисперсия (теор)    : {d_th:.4f}
                    Дисперсия (эмп)     : {d_emp:.4f}
                    Погрешность D       : {err_d:.2%}

                    Критерий Пирсона χ²:
                    Значение статистики : {chi_val:.4f}
                    Критическое (df={df}): {chi_crit:.4f}
                    Результат: {'ПРОЙДЕН' if chi_val < chi_crit else 'ОТКЛОНЕН'}
                    Эмпирические вероятности:
                    {p_emp}
                    """
        self.stats_output.setText(report)

    def run_normal(self):
        N = self.sample_count.value()

        samples = []
        for _ in range(N // 2):
            u1, u2 = random.random(), random.random()
            z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            samples.extend([z0, z1])
        
        self.plot_normal(samples, N)
        self.stats_output.setText(f">>> ОТЧЕТ ПО НОРМАЛЬНОЙ СВ\nN={N}\nСреднее: {np.mean(samples):.4f}\nДисперсия: {np.var(samples):.4f}")

    def plot_discrete(self, x, samples):
        self.ax.clear()
        self.ax.hist(samples, bins=len(x)*2, color='#FF8C00', alpha=0.7, density=True, label='Empirical')
        self.ax.set_title(f"Discrete Distribution (N={len(samples)})", color='#FF8C00')
        self.canvas.draw()

    def plot_normal(self, samples, N):
        self.ax.clear()
        count, bins, ignored = self.ax.hist(samples, bins=50, density=True, color='#00BFFF', alpha=0.6)

        mu, sigma = 0, 1
        self.ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), lw=2, color='white')
        self.ax.set_title(f"Normal Distribution (N={N})", color='#00BFFF')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabWorkApp()
    window.show()
    sys.exit(app.exec())
