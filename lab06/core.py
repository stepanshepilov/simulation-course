import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.stats import chi2, norm
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
                             QComboBox, QLabel, QHeaderView, QSpinBox, QFrame)
from abc import ABC, abstractmethod

DARK_STYLE = """
    QMainWindow { background-color: #0b0e14; }
    QWidget { background-color: #0b0e14; color: #ecf0f1; font-family: 'Segoe UI', sans-serif; }
    
    QTableWidget { 
        background-color: #151921; 
        border: 1px solid #34495e; 
        gridline-color: #2c3e50; 
        border-radius: 5px;
    }
    QHeaderView::section { 
        background-color: #1c222d; 
        color: #3498db; 
        padding: 5px; 
        border: 1px solid #2c3e50;
    }

    QPushButton { 
        background-color: #1c222d; 
        color: #3498db; 
        border: 2px solid #3498db; 
        border-radius: 5px; 
        padding: 12px; 
        font-weight: bold; 
        font-size: 12px;
    }
    QPushButton:hover { background-color: #3498db; color: #ffffff; }
    QPushButton#btn_normal { border-color: #e74c3c; color: #e74c3c; }
    QPushButton#btn_normal:hover { background-color: #e74c3c; color: #ffffff; }

    QSpinBox, QComboBox { 
        background-color: #151921; 
        border: 1px solid #34495e; 
        border-radius: 3px; 
        padding: 5px; 
        color: white; 
    }

    QLabel#Header { color: #3498db; font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px; }
"""

class StatCard(QFrame):
    def __init__(self, title, unit="%"):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            StatCard {
                background-color: #1c222d;
                border: 1px solid #2c3e50;
                border-left: 5px solid #3498db;
                border-radius: 8px;
            }
            QLabel#Title { color: #bdc3c7; font-size: 11px; text-transform: uppercase; }
            QLabel#Value { color: #ffffff; font-size: 20px; font-weight: bold; }
            QLabel#Status { font-size: 10px; font-weight: bold; }
        """)
        layout = QVBoxLayout(self)
        
        self.title_label = QLabel(title)
        self.title_label.setObjectName("Title")
        
        self.value_label = QLabel("0.00")
        self.value_label.setObjectName("Value")
        
        self.status_label = QLabel("ОЖИДАНИЕ")
        self.status_label.setObjectName("Status")
        self.status_label.setStyleSheet("color: #7f8c8d;")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.status_label)

    def update_val(self, value, is_error=True, threshold=0.05, custom_status=None):
        if isinstance(value, float):
            self.value_label.setText(f"{value:.4f}")
        else:
            self.value_label.setText(str(value))
            
        if custom_status:
            self.status_label.setText(custom_status.upper())
            color = "#2ecc71" if "ПРОЙДЕН" in custom_status or "OK" in custom_status else "#e74c3c"
        
        elif is_error:
            err_percent = value * 100
            self.value_label.setText(f"{err_percent:.2f}%")
            if err_percent < threshold * 100:
                self.status_label.setText("В ПРЕДЕЛАХ НОРМЫ")
                color = "#2ecc71"
            else:
                self.status_label.setText("ВЫСОКАЯ ПОГРЕШНОСТЬ")
                color = "#f1c40f"
        
        else:
            color = "#3498db"
            
        self.status_label.setStyleSheet(f"color: {color};")

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
        self.setWindowTitle("Simulation Lab: Discrete & Normal Distributions")
        self.resize(1400, 850)
        self.setStyleSheet(DARK_STYLE)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        
        lbl1 = QLabel("ПАРАМЕТРЫ ВЫБОРКИ")
        lbl1.setObjectName("Header")
        left_panel.addWidget(lbl1)
        
        self.sample_count = QSpinBox()
        self.sample_count.setRange(100, 1000000)
        self.sample_count.setValue(5000)
        self.sample_count.setSingleStep(1000)
        left_panel.addWidget(self.sample_count)

        lbl_bins = QLabel("ИНТЕРВАЛОВ (для Нормального)")
        lbl_bins.setObjectName("Header")
        left_panel.addWidget(lbl_bins)
        
        self.bins_count_spin = QSpinBox()
        self.bins_count_spin.setRange(5, 100)
        self.bins_count_spin.setValue(15)
        left_panel.addWidget(self.bins_count_spin)

        left_panel.addSpacing(10)
        
        lbl2 = QLabel("ДИСКРЕТНАЯ СВ")
        lbl2.setObjectName("Header")
        left_panel.addWidget(lbl2)
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["X", "P"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        left_panel.addWidget(self.table)

        self.strat_combo = QComboBox()
        self.strategies = {"Пресеты...": None, "Равномерное [1..5]": UniformStrategy(), "Бернулли": BernoulliStrategy()}
        self.strat_combo.addItems(self.strategies.keys())
        self.strat_combo.currentIndexChanged.connect(self.apply_preset)
        left_panel.addWidget(self.strat_combo)

        self.btn_discrete = QPushButton("СЭМПЛИТЬ ДИСКРЕТНУЮ")
        self.btn_discrete.clicked.connect(self.run_discrete)
        left_panel.addWidget(self.btn_discrete)

        left_panel.addSpacing(15)
        lbl3 = QLabel("НОРМАЛЬНАЯ СВ (Box-Muller)")
        lbl3.setObjectName("Header")
        left_panel.addWidget(lbl3)

        self.btn_normal = QPushButton("СГЕНЕРИРОВАТЬ НОРМАЛЬНУЮ")
        self.btn_normal.setObjectName("btn_normal")
        self.btn_normal.clicked.connect(self.run_normal)
        left_panel.addWidget(self.btn_normal)

        main_layout.addLayout(left_panel, 1)

        mid_panel = QVBoxLayout()
        lbl4 = QLabel("АНАЛИТИЧЕСКИЕ ПОКАЗАТЕЛИ")
        lbl4.setObjectName("Header")
        mid_panel.addWidget(lbl4)

        self.card_chi = StatCard("Критерий Пирсона χ²")
        self.card_mean_err = StatCard("Отклонение среднего (M)")
        self.card_std_err = StatCard("Отклонение СКО (σ)")
        
        mid_panel.addWidget(self.card_chi)
        mid_panel.addWidget(self.card_mean_err)
        mid_panel.addWidget(self.card_std_err)
        mid_panel.addStretch()

        main_layout.addLayout(mid_panel, 1)

        right_panel = QVBoxLayout()
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.figure.patch.set_facecolor('#0b0e14')
        self.ax.set_facecolor('#151921')
        self.canvas = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas)
        
        main_layout.addLayout(right_panel, 2)

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

            p_sum = sum(p_th)
            p_th = [p/p_sum for p in p_th]

            samples = []
            cdf = np.cumsum(p_th)
            for _ in range(N):
                u = random.random()

                for i, c in enumerate(cdf):
                    if u <= c:
                        samples.append(x_th[i])
                        break
            
            m_th = sum(x * p for x, p in zip(x_th, p_th))
            d_th = sum((x**2) * p for x, p in zip(x_th, p_th)) - m_th**2
            s_th = np.sqrt(d_th)

            m_emp = np.mean(samples)
            s_emp = np.std(samples)

            err_m = abs(m_th - m_emp) / (abs(m_th) if m_th != 0 else 1)
            err_s = abs(s_th - s_emp) / (s_th if s_th != 0 else 1)

            unique, counts = np.unique(samples, return_counts=True)
            freq_map = dict(zip(unique, counts))
            chi_val = 0
            for i in range(len(x_th)):
                e_i = N * p_th[i]
                o_i = freq_map.get(x_th[i], 0)
                chi_val += ((o_i - e_i)**2) / e_i
            
            df = len(x_th) - 1
            chi_crit = chi2.ppf(0.95, df)
            chi_status = "Пройден" if chi_val < chi_crit else "Отклонен"

            self.card_chi.update_val(chi_val, is_error=False, custom_status=f"{chi_status} (кр: {chi_crit:.2f})")
            self.card_mean_err.update_val(err_m)
            self.card_std_err.update_val(err_s)
            
            self.plot_discrete(x_th, samples)
        except Exception as e:
            print(f"Error: {e}")

    def run_normal(self):
        N = self.sample_count.value()
        K = self.bins_count_spin.value()

        samples = []
        for _ in range(N // 2):
            u1, u2 = random.random(), random.random()
            z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            samples.extend([z0, z1])
        
        samples = np.array(samples)
        
        m_emp = np.mean(samples)
        s_emp = np.std(samples)
        
        err_m = abs(0 - m_emp) 
        err_s = abs(1 - s_emp)

        observed_freq, bin_edges = np.histogram(samples, bins=K)
        
        chi_val = 0
        for i in range(K):
            p_i = norm.cdf(bin_edges[i+1]) - norm.cdf(bin_edges[i])
            expected_freq = N * p_i
            
            if expected_freq > 0:
                chi_val += ((observed_freq[i] - expected_freq)**2) / expected_freq
        
        df = K - 1
        chi_crit = chi2.ppf(0.95, df)
        chi_status = "Пройден" if chi_val < chi_crit else "Отклонен"

        self.card_chi.update_val(chi_val, is_error=False, custom_status=f"{chi_status} (кр: {chi_crit:.2f})")
        self.card_mean_err.update_val(err_m, threshold=0.05)
        self.card_std_err.update_val(err_s, threshold=0.05)
        
        self.plot_normal(samples, N, K)

    def plot_discrete(self, x, samples):
        self.ax.clear()
        self.ax.hist(samples, bins=np.arange(min(x)-0.5, max(x)+1.5, 1), 
                     color='#3498db', alpha=0.7, rwidth=0.8, density=True)
        self.ax.set_title("Гистограмма Дискретной СВ", color='#3498db')
        self.ax.tick_params(colors='#7f8c8d')
        self.canvas.draw()

    def plot_normal(self, samples, N, K):
        self.ax.clear()

        _, bins, _ = self.ax.hist(samples, bins=K, density=True, color='#e74c3c', alpha=0.6, rwidth=0.9)
        
        x = np.linspace(min(bins), max(bins), 100)
        y = norm.pdf(x, 0, 1)
        self.ax.plot(x, y, '--', color='white', linewidth=2, label='Theory')
        
        self.ax.set_title(f"Нормальное распределение (N={N}, K={K})", color='#e74c3c')
        self.ax.tick_params(colors='#7f8c8d')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabWorkApp()
    window.show()
    sys.exit(app.exec())
