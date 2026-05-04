import sys
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QDoubleSpinBox,
    QGroupBox, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, QTimer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DARK_STYLE = """
QMainWindow {
    background-color: #0a0e27;
}

QGroupBox {
    font-size: 13px;
    font-weight: bold;
    color: #00d4ff;
    border: 2px solid #1a2a4f;
    border-radius: 10px;
    margin-top: 10px;
    padding-top: 10px;
    background-color: rgba(20, 30, 55, 0.5);
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 10px 0 10px;
}

QLabel {
    color: #c0d4f0;
    font-size: 12px;
}

QPushButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                stop: 0 #00d4ff, stop: 1 #0088cc);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                stop: 0 #00eaff, stop: 1 #0099dd);
}

QPushButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                stop: 0 #0088cc, stop: 1 #0066aa);
}

QSpinBox, QDoubleSpinBox, QSlider {
    background-color: #1a1f3a;
    border: 1px solid #2a3a5a;
    border-radius: 5px;
    padding: 5px;
    color: #00d4ff;
    font-size: 12px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #00d4ff;
}

QProgressBar {
    border: 1px solid #2a3a5a;
    border-radius: 5px;
    text-align: center;
    color: white;
}

QProgressBar::chunk {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #00d4ff, stop: 1 #0088cc);
    border-radius: 4px;
}
"""

class ModernPlotCanvas(FigureCanvas):
    """Холст для построения распределения числа заявок в M/M/1."""
    def __init__(self, parent=None, width=12, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0e27')
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111, facecolor='#0f1433')
        self.fig.subplots_adjust(right=0.75, left=0.1)
        self.legend_ax = self.fig.add_axes([0.78, 0.3, 0.18, 0.4])
        self.legend_ax.axis('off')
        self.setup_styles()

    def setup_styles(self):
        """Единообразное оформление осей."""
        for spine in self.ax.spines.values():
            spine.set_color('#2a3a5a')
        self.ax.tick_params(colors='#c0d4f0', labelsize=10)
        self.ax.xaxis.label.set_color('#00d4ff')
        self.ax.yaxis.label.set_color('#00d4ff')
        self.ax.title.set_color('#ffffff')

    def plot_results(self, state_counts: np.ndarray, rho: float):
        """
        Построение гистограммы распределения числа заявок в системе
        и теоретического геометрического распределения.
        """
        self.ax.clear()
        self.legend_ax.clear()
        self.legend_ax.axis('off')
        self.setup_styles()

        # Нормировка эмпирических частот
        total = state_counts.sum()
        if total > 0: 
            empirical_probs = state_counts / total
        else:
            empirical_probs = np.zeros_like(state_counts, dtype=float)

        max_state = len(state_counts) - 1
        states = np.arange(0, max_state + 1)

        # Гистограмма
        self.ax.bar(states, empirical_probs, color='#00d4ff', alpha=0.7,
                    edgecolor='#0088cc', linewidth=1.5, label='Эмпирическое')

        # Теоретическая вероятность P(n) = (1-rho)*rho^n
        theoretical_probs = (1 - rho) * (rho ** states) if rho < 1 else np.zeros_like(states)
        self.ax.plot(states, theoretical_probs, 'o-', color='#ff6b6b',
                    markersize=8, linewidth=2.5, markerfacecolor='#ff6b6b',
                    markeredgecolor='white', markeredgewidth=1.5, label='Теоретическое')

        self.ax.grid(True, alpha=0.2, linestyle='--', color='#2a3a5a')
        self.ax.set_xlabel('Число заявок в системе', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Вероятность', fontsize=12, fontweight='bold')
        self.ax.set_title(f'Распределение числа заявок в M/M/1\n'
                         f'ρ = {rho:.3f}', fontsize=13, fontweight='bold', pad=20)

        # Легенда на отдельной оси
        legend_elements = [
            Patch(facecolor='#00d4ff', alpha=0.7, edgecolor='#0088cc', label='Эмпирическое'),
            Line2D([0], [0], marker='o', color='#ff6b6b', markerfacecolor='#ff6b6b',
                   markeredgecolor='white', linewidth=2.5, markersize=8,
                   label='Теор. P(n)=(1-ρ)ρⁿ')
        ]
        self.legend_ax.legend(handles=legend_elements, loc='center', fontsize=10,
                              framealpha=0.9, facecolor='#1a1f3a', edgecolor='#00d4ff')

        # Аннотация
        if rho < 1:
            mean_theor = rho / (1 - rho)
        else:
            mean_theor = float('inf')
        mean_emp = np.average(states, weights=empirical_probs) if total > 0 else 0
        textstr = f'Среднее (эмп.): {mean_emp:.3f}\nСреднее (теор.): {mean_theor:.3f}'
        props = dict(boxstyle='round', facecolor='#1a1f3a', alpha=0.8,
                     edgecolor='#00d4ff', linewidth=1.5)
        self.ax.text(0.95, 0.95, textstr, transform=self.ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props, color='#c0d4f0')

        self.draw()


class MM1Simulator:
    def __init__(self):
        pass

    def simulate(self, lam: float, mu: float, T_end: float, warmup: float = 0.0, progress_callback=None) -> dict:
        queue = deque()
        server_busy = False
        events = []

        t_arrival = np.random.exponential(1.0 / lam)
        events.append((t_arrival, 'arrival'))

        state_changes = []
        wait_times = []
        last_event_time = 0.0
        current_queue_length = 0
        server_busy_time_total = 0.0
        num_served = 0
        arrival_times = {}

        next_progress_update = 0.0
        progress_step = max(1, int(T_end / 100))

        while events and last_event_time < T_end:
            events.sort(key=lambda x: x[0])
            t, etype = events.pop(0)

            if t >= T_end:
                break

            delta = t - last_event_time
            if last_event_time >= warmup:
                state_changes.append((current_queue_length, delta))
                if server_busy:
                    server_busy_time_total += delta

            if etype == 'arrival':
                current_queue_length += 1
                arr_id = id(object())
                arrival_times[arr_id] = t

                if not server_busy:
                    server_busy = True
                    service_time = np.random.exponential(1.0 / mu)
                    events.append((t + service_time, 'departure'))
                else:
                    queue.append(arr_id)

                next_arrival = t + np.random.exponential(1.0 / lam)
                if next_arrival < T_end:
                    events.append((next_arrival, 'arrival'))

            elif etype == 'departure':
                current_queue_length -= 1
                if current_queue_length < 0:
                    current_queue_length = 0

                if queue:
                    next_id = queue.popleft()
                    wait_time = t - arrival_times[next_id]

                    if t >= warmup:
                        wait_times.append(wait_time)
                    
                    num_served += 1
                    service_time = np.random.exponential(1.0 / mu)
                    events.append((t + service_time, 'departure'))
                else:
                    server_busy = False
            
            last_event_time = t

            if progress_callback and t >= next_progress_update:
                progress = min(100, int(t / T_end * 100))
                progress_callback(progress)
                next_progress_update += progress_step

        if last_event_time < T_end:
            delta = T_end - last_event_time
            if last_event_time >= warmup:
                state_changes.append((current_queue_length, delta))
                if server_busy:
                    server_busy_time_total += delta

        max_state = max([s for s, _ in state_changes]) if state_changes else 0
        state_counts = np.zeros(max_state + 1)
        for s, duration in state_changes:
            state_counts[s] += duration

        return {
            'states': state_counts,
            'wait_times': wait_times,
            'total_time': T_end - warmup,
            'server_busy_time': server_busy_time_total,
            'num_served': len(wait_times)
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Имитационное моделирование M/M/1 | СМО")
        self.setGeometry(100, 100, 1400, 850)
        self.setStyleSheet(DARK_STYLE)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)

        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, stretch=2)

        self.simulator = MM1Simulator()
        self.current_results = None

        self.update_rho_display()

    def create_control_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        header = QLabel("⚙️ КОНФИГУРАЦИЯ M/M/1")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff; padding: 10px;")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2a3a5a; max-height: 2px;")
        layout.addWidget(sep)

        # Параметры λ и μ
        flow_group = QGroupBox("Интенсивности")
        flow_layout = QVBoxLayout()

        # λ
        lambda_layout = QVBoxLayout()
        lambda_layout.addWidget(QLabel("λ (заявок/сек):"))
        self.lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.lambda_slider.setRange(1, 200)  # 0.1 .. 2.0 (масштаб x100)
        self.lambda_slider.setValue(80)      # 0.8
        self.lambda_slider.valueChanged.connect(self.on_lambda_changed)
        lambda_layout.addWidget(self.lambda_slider)

        lambda_value_layout = QHBoxLayout()
        self.lambda_value_label = QLabel("0.800")
        self.lambda_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lambda_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        self.lambda_spinbox = QDoubleSpinBox()
        self.lambda_spinbox.setRange(0.1, 2.0)
        self.lambda_spinbox.setValue(0.8)
        self.lambda_spinbox.setSingleStep(0.1)
        self.lambda_spinbox.valueChanged.connect(self.on_lambda_spinbox_changed)
        lambda_value_layout.addWidget(QLabel("Значение:"))
        lambda_value_layout.addWidget(self.lambda_spinbox)
        lambda_value_layout.addWidget(self.lambda_value_label)
        lambda_layout.addLayout(lambda_value_layout)
        flow_layout.addLayout(lambda_layout)

        # μ
        mu_layout = QVBoxLayout()
        mu_layout.addWidget(QLabel("μ (обслуживаний/сек):"))
        self.mu_slider = QSlider(Qt.Orientation.Horizontal)
        self.mu_slider.setRange(10, 300)  # 0.1 .. 3.0
        self.mu_slider.setValue(100)      # 1.0
        self.mu_slider.valueChanged.connect(self.on_mu_changed)
        mu_layout.addWidget(self.mu_slider)

        mu_value_layout = QHBoxLayout()
        self.mu_value_label = QLabel("1.000")
        self.mu_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.mu_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        self.mu_spinbox = QDoubleSpinBox()
        self.mu_spinbox.setRange(0.1, 3.0)
        self.mu_spinbox.setValue(1.0)
        self.mu_spinbox.setSingleStep(0.1)
        self.mu_spinbox.valueChanged.connect(self.on_mu_spinbox_changed)
        mu_value_layout.addWidget(QLabel("Значение:"))
        mu_value_layout.addWidget(self.mu_spinbox)
        mu_value_layout.addWidget(self.mu_value_label)
        mu_layout.addLayout(mu_value_layout)
        flow_layout.addLayout(mu_layout)

        # Отображение ρ
        self.rho_display = QLabel()
        self.rho_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rho_display.setStyleSheet("""
            background-color: #1a2a4f;
            border-radius: 8px;
            padding: 10px;
            font-size: 13px;
            color: #00d4ff;
        """)
        flow_layout.addWidget(self.rho_display)

        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)

        # Параметры моделирования
        sim_group = QGroupBox("Параметры моделирования")
        sim_layout = QVBoxLayout()

        sim_layout.addWidget(QLabel("Время моделирования (сек):"))
        self.T_spinbox = QDoubleSpinBox()
        self.T_spinbox.setRange(100, 100000)
        self.T_spinbox.setValue(5000)
        self.T_spinbox.setSingleStep(1000)
        sim_layout.addWidget(self.T_spinbox)

        sim_layout.addWidget(QLabel("Разогрев (сек):"))
        self.warmup_spinbox = QDoubleSpinBox()
        self.warmup_spinbox.setRange(0, 10000)
        self.warmup_spinbox.setValue(500)
        self.warmup_spinbox.setSingleStep(100)
        sim_layout.addWidget(self.warmup_spinbox)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Кнопка запуска
        self.run_button = QPushButton("🚀 ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setMinimumHeight(45)
        layout.addWidget(self.run_button)

        # Статистика
        stats_group = QGroupBox("Операционные характеристики")
        stats_layout = QVBoxLayout()
        self.stats_display = QLabel()
        self.stats_display.setStyleSheet("""
            background-color: #1a1f3a;
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 11px;
        """)
        self.stats_display.setWordWrap(True)
        stats_layout.addWidget(self.stats_display)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        return panel

    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        header = QLabel("📊 РАСПРЕДЕЛЕНИЕ ЧИСЛА ЗАЯВОК В СИСТЕМЕ")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff; padding: 10px;")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2a3a5a; max-height: 2px;")
        layout.addWidget(sep)

        self.plot_canvas = ModernPlotCanvas(width=10, height=7)
        layout.addWidget(self.plot_canvas)

        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #1a2a4f;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        self.info_text = QLabel("Настройте параметры и запустите моделирование.")
        self.info_text.setStyleSheet("color: #c0d4f0; font-size: 11px;")
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_frame)

        return panel

    # Слоты для λ
    def on_lambda_changed(self, value):
        lam = value / 100.0
        self.lambda_spinbox.blockSignals(True)
        self.lambda_spinbox.setValue(lam)
        self.lambda_spinbox.blockSignals(False)
        self.lambda_value_label.setText(f"{lam:.3f}")
        self.update_rho_display()

    def on_lambda_spinbox_changed(self, value):
        self.lambda_slider.blockSignals(True)
        self.lambda_slider.setValue(int(value * 100))
        self.lambda_slider.blockSignals(False)
        self.lambda_value_label.setText(f"{value:.3f}")
        self.update_rho_display()

    # Слоты для μ
    def on_mu_changed(self, value):
        mu = value / 100.0
        self.mu_spinbox.blockSignals(True)
        self.mu_spinbox.setValue(mu)
        self.mu_spinbox.blockSignals(False)
        self.mu_value_label.setText(f"{mu:.3f}")
        self.update_rho_display()

    def on_mu_spinbox_changed(self, value):
        self.mu_slider.blockSignals(True)
        self.mu_slider.setValue(int(value * 100))
        self.mu_slider.blockSignals(False)
        self.mu_value_label.setText(f"{value:.3f}")
        self.update_rho_display()

    def update_rho_display(self):
        lam = self.lambda_spinbox.value()
        mu = self.mu_spinbox.value()
        rho = lam / mu if mu > 0 else 0
        if rho < 1:
            self.rho_display.setText(f"Загрузка ρ = λ/μ = {rho:.4f}   (стационарный режим)")
        else:
            self.rho_display.setText(f"ρ = {rho:.4f}   ⚠ Нестационарный режим (ρ ≥ 1)")
            self.rho_display.setStyleSheet("""
                background-color: #1a2a4f; border-radius: 8px; padding: 10px;
                font-size: 13px; color: #ff6b6b;
            """)
            return
        self.rho_display.setStyleSheet("""
            background-color: #1a2a4f; border-radius: 8px; padding: 10px;
            font-size: 13px; color: #00d4ff;
        """)

    def run_simulation(self):
        self.run_button.setEnabled(False)
        self.run_button.setText("⏳ МОДЕЛИРОВАНИЕ...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        lam = self.lambda_spinbox.value()
        mu = self.mu_spinbox.value()
        T_end = self.T_spinbox.value()
        warmup = self.warmup_spinbox.value()

        def update_progress(progress):
            self.progress_bar.setValue(progress)
            QApplication.processEvents()

        def run_in_background():
            results = self.simulator.simulate(lam, mu, T_end, warmup, update_progress)
            self.current_results = results
            self.display_results(results, lam, mu)
            self.run_button.setEnabled(True)
            self.run_button.setText("🚀 ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ")
            self.progress_bar.setVisible(False)

        QTimer.singleShot(50, run_in_background)

    def display_results(self, results: dict, lam: float, mu: float):
        rho = lam / mu if mu > 0 else 0
        total_time = results['total_time']
        state_counts = results['states']
        wait_times = results['wait_times']

        # Характеристики
        L_emp = np.sum(np.arange(len(state_counts)) * state_counts) / total_time
        server_busy = results['server_busy_time']
        U_emp = server_busy / total_time if total_time > 0 else 0
        W_emp = np.mean(wait_times) if wait_times else 0
        num_served = results['num_served']

        # Теоретические (для стационарного режима)
        if rho < 1:
            L_theor = rho / (1 - rho)
            U_theor = rho
            W_theor = 1 / (mu * (1 - rho))
        else:
            L_theor = float('inf')
            U_theor = 1.0
            W_theor = float('inf')

        # Построение графика
        self.plot_canvas.plot_results(state_counts, rho)

        # Текст статистики
        stats_text = (
            f"<b>ЭМПИРИЧЕСКИЕ ХАРАКТЕРИСТИКИ (после разогрева)</b><br>"
            f"Среднее число заявок в системе L: {L_emp:.4f}<br>"
            f"Коэффициент загрузки сервера U: {U_emp:.4f}<br>"
            f"Среднее время ожидания W: {W_emp:.4f} сек<br>"
            f"Число обслуженных заявок: {num_served}<br>"
            f"<br>"
            f"<b>ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ (стац. режим)</b><br>"
            f"L = ρ/(1-ρ) = {L_theor:.4f}<br>"
            f"U = ρ = {U_theor:.4f}<br>"
            f"W = 1/(μ(1-ρ)) = {W_theor:.4f}<br>"
        )

        if rho < 1:
            if abs(L_emp - L_theor) < 0.1 * L_theor:
                stats_text += '<span style="color:#00ff88;">✓ Хорошее совпадение с теорией</span>'
            else:
                stats_text += '<span style="color:#ffaa00;">⚠ Заметное расхождение, увеличьте время моделирования</span>'
        else:
            stats_text += '<span style="color:#ff6b6b;">⚠ Режим перегрузки (ρ ≥ 1), очередь растёт неограниченно</span>'

        self.stats_display.setText(stats_text)

        self.info_text.setText(
            f"Моделирование завершено. Длина прогона: {total_time:.0f} с. "
            f"Обслужено: {num_served}. L={L_emp:.3f}, W={W_emp:.3f} с."
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
