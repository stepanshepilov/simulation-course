import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Стиль темной темы
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

class SimulationWorker(QThread):
    """Поток для выполнения имитационного моделирования M/M/1"""
    finished = pyqtSignal(dict, list, list)  # результаты, времена, длины очереди
    progress = pyqtSignal(int)               # прогресс в процентах

    def __init__(self, lambd, mu, sim_time):
        super().__init__()
        self.lambd = lambd
        self.mu = mu
        self.sim_time = sim_time

    def run(self):
        t = 0.0
        t_arrival = np.random.exponential(1/self.lambd)
        t_departure = float('inf')
        n_queue = 0
        n_system = 0
        server_busy = False

        arrival_times = []
        total_busy_time = 0.0
        total_wait_time = 0.0
        total_system_time = 0.0
        served_count = 0
        area_queue = 0.0
        area_system = 0.0
        last_event_time = 0.0

        time_points = [0.0]
        queue_lengths = [0]

        last_progress_update = 0.0
        progress_step = self.sim_time / 100.0

        while t < self.sim_time:
            if t_arrival <= t_departure and t_arrival <= self.sim_time:
                t = t_arrival
                dt = t - last_event_time
                area_queue += n_queue * dt
                area_system += n_system * dt
                if server_busy:
                    total_busy_time += dt
                last_event_time = t

                n_system += 1
                if not server_busy:
                    server_busy = True
                    service_time = np.random.exponential(1/self.mu)
                    t_departure = t + service_time
                    arrival_times.append(t)
                else:
                    n_queue += 1
                    arrival_times.append(t)

                time_points.append(t)
                queue_lengths.append(n_queue)
                t_arrival = t + np.random.exponential(1/self.lambd)

            elif t_departure <= t_arrival and t_departure <= self.sim_time:
                # Уход
                t = t_departure
                dt = t - last_event_time
                area_queue += n_queue * dt
                area_system += n_system * dt
                if server_busy:
                    total_busy_time += dt
                last_event_time = t

                n_system -= 1
                if arrival_times:
                    arrival_t = arrival_times.pop(0)
                    time_in_system = t - arrival_t
                    total_system_time += time_in_system
                served_count += 1

                if n_queue > 0:
                    n_queue -= 1
                    service_time = np.random.exponential(1/self.mu)
                    t_departure = t + service_time
                    wait_time = t - arrival_times[0]
                    total_wait_time += wait_time
                else:
                    server_busy = False
                    t_departure = float('inf')

                time_points.append(t)
                queue_lengths.append(n_queue)
            else:
                break

            # Обновление прогресса
            if t - last_progress_update >= progress_step:
                progress_pct = int((t / self.sim_time) * 100)
                self.progress.emit(progress_pct)
                last_progress_update = t

        if t < self.sim_time:
            dt = self.sim_time - last_event_time
            area_queue += n_queue * dt
            area_system += n_system * dt
            if server_busy:
                total_busy_time += dt

        avg_queue_len = area_queue / self.sim_time
        avg_system_len = area_system / self.sim_time
        server_utilization = total_busy_time / self.sim_time

        if served_count > 0:
            avg_wait_time = total_wait_time / served_count
            avg_system_time = total_system_time / served_count
        else:
            avg_wait_time = 0.0
            avg_system_time = 0.0

        rho = self.lambd / self.mu
        if rho < 1:
            L_analytical = rho / (1 - rho)
            Lq_analytical = rho**2 / (1 - rho)
            W_analytical = 1 / (self.mu - self.lambd)
            Wq_analytical = rho / (self.mu - self.lambd)
        else:
            L_analytical = Lq_analytical = W_analytical = Wq_analytical = float('inf')

        results = {
            'lambda': self.lambd,
            'mu': self.mu,
            'rho': rho,
            'sim_time': self.sim_time,
            'served': served_count,
            'utilization': server_utilization,
            'L_sim': avg_system_len,
            'Lq_sim': avg_queue_len,
            'W_sim': avg_system_time,
            'Wq_sim': avg_wait_time,
            'L_analytical': L_analytical,
            'Lq_analytical': Lq_analytical,
            'W_analytical': W_analytical,
            'Wq_analytical': Wq_analytical
        }

        self.finished.emit(results, time_points, queue_lengths)


class ModernPlotCanvas(FigureCanvas):
    """Холст для отображения графика длины очереди"""
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0e27')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, facecolor='#0f1433')
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
        self.setup_styles()

    def setup_styles(self):
        self.ax.spines['top'].set_color('#2a3a5a')
        self.ax.spines['right'].set_color('#2a3a5a')
        self.ax.spines['bottom'].set_color('#2a3a5a')
        self.ax.spines['left'].set_color('#2a3a5a')
        self.ax.tick_params(colors='#c0d4f0', labelsize=10)
        self.ax.xaxis.label.set_color('#00d4ff')
        self.ax.yaxis.label.set_color('#00d4ff')
        self.ax.title.set_color('#ffffff')

    def plot_queue(self, time_points, queue_lengths, lambda_val, mu_val):
        self.ax.clear()
        self.setup_styles()
        self.ax.step(time_points, queue_lengths, where='post', linewidth=2, color='#00d4ff')
        self.ax.fill_between(time_points, queue_lengths, step='post', alpha=0.2, color='#00d4ff')
        self.ax.grid(True, alpha=0.2, linestyle='--', color='#2a3a5a')
        self.ax.set_xlabel('Время моделирования', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Длина очереди', fontsize=12, fontweight='bold')
        self.ax.set_title(f'Динамика длины очереди (λ={lambda_val:.2f}, μ={mu_val:.2f})',
                          fontsize=13, fontweight='bold', pad=15)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симулятор СМО M/M/1 | Имитационное моделирование")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet(DARK_STYLE)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Левая панель управления
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Правая панель с графиком
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, stretch=2)

        self.worker = None

        # Установка начальных значений
        self.update_rho_display()

    def create_control_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        header = QLabel("⚙️ ПАРАМЕТРЫ МОДЕЛИ")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff; padding: 10px;")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2a3a5a; max-height: 2px;")
        layout.addWidget(sep)

        # Интенсивность λ
        lambda_group = QGroupBox("Входной поток")
        lambda_layout = QVBoxLayout()

        lambda_slider_layout = QVBoxLayout()
        lambda_slider_layout.addWidget(QLabel("Интенсивность λ (заявок/ед.времени):"))
        self.lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.lambda_slider.setRange(10, 200)  # 0.1 .. 20.0
        self.lambda_slider.setValue(20)       # 2.0
        self.lambda_slider.valueChanged.connect(self.on_lambda_slider)
        lambda_slider_layout.addWidget(self.lambda_slider)

        lambda_value_layout = QHBoxLayout()
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.1, 20.0)
        self.lambda_spin.setValue(2.0)
        self.lambda_spin.setSingleStep(0.5)
        self.lambda_spin.valueChanged.connect(self.on_lambda_spin)
        lambda_value_layout.addWidget(QLabel("Значение:"))
        lambda_value_layout.addWidget(self.lambda_spin)
        lambda_slider_layout.addLayout(lambda_value_layout)

        lambda_layout.addLayout(lambda_slider_layout)
        lambda_group.setLayout(lambda_layout)
        layout.addWidget(lambda_group)

        # Интенсивность μ
        mu_group = QGroupBox("Обслуживание")
        mu_layout = QVBoxLayout()

        mu_slider_layout = QVBoxLayout()
        mu_slider_layout.addWidget(QLabel("Интенсивность μ (заявок/ед.времени):"))
        self.mu_slider = QSlider(Qt.Orientation.Horizontal)
        self.mu_slider.setRange(10, 200)
        self.mu_slider.setValue(30)           # 3.0
        self.mu_slider.valueChanged.connect(self.on_mu_slider)
        mu_slider_layout.addWidget(self.mu_slider)

        mu_value_layout = QHBoxLayout()
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(0.1, 20.0)
        self.mu_spin.setValue(3.0)
        self.mu_spin.setSingleStep(0.5)
        self.mu_spin.valueChanged.connect(self.on_mu_spin)
        mu_value_layout.addWidget(QLabel("Значение:"))
        mu_value_layout.addWidget(self.mu_spin)
        mu_slider_layout.addLayout(mu_value_layout)

        mu_layout.addLayout(mu_slider_layout)
        mu_group.setLayout(mu_layout)
        layout.addWidget(mu_group)

        # Время моделирования
        time_group = QGroupBox("Эксперимент")
        time_layout = QVBoxLayout()
        time_layout.addWidget(QLabel("Время моделирования T:"))
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(100.0, 100000.0)
        self.time_spin.setValue(10000.0)
        self.time_spin.setSingleStep(1000.0)
        time_layout.addWidget(self.time_spin)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Отображение загрузки ρ
        self.rho_label = QLabel()
        self.rho_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rho_label.setStyleSheet("""
            background-color: #1a2a4f;
            border-radius: 8px;
            padding: 10px;
            font-size: 13px;
            color: #00d4ff;
        """)
        layout.addWidget(self.rho_label)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Кнопка запуска
        self.run_button = QPushButton("🚀 ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ")
        self.run_button.clicked.connect(self.start_simulation)
        self.run_button.setMinimumHeight(45)
        layout.addWidget(self.run_button)

        # Статистика
        stats_group = QGroupBox("Результаты")
        stats_layout = QVBoxLayout()
        self.stats_display = QLabel()
        self.stats_display.setStyleSheet("""
            background-color: #1a1f3a;
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 11px;
            color: #c0d4f0;
        """)
        self.stats_display.setWordWrap(True)
        self.stats_display.setText("Ожидание запуска...")
        stats_layout.addWidget(self.stats_display)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        return panel

    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        header = QLabel("📈 ДИНАМИКА ОЧЕРЕДИ")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff; padding: 10px;")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2a3a5a; max-height: 2px;")
        layout.addWidget(sep)

        self.plot_canvas = ModernPlotCanvas(width=10, height=6)
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
        self.info_text = QLabel("Готов к моделированию. Настройте параметры и нажмите 'Запустить моделирование'")
        self.info_text.setStyleSheet("color: #c0d4f0; font-size: 11px;")
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_frame)

        return panel

    def on_lambda_slider(self, value):
        lambda_val = value / 10.0
        self.lambda_spin.blockSignals(True)
        self.lambda_spin.setValue(lambda_val)
        self.lambda_spin.blockSignals(False)
        self.update_rho_display()

    def on_lambda_spin(self, value):
        self.lambda_slider.blockSignals(True)
        self.lambda_slider.setValue(int(value * 10))
        self.lambda_slider.blockSignals(False)
        self.update_rho_display()

    def on_mu_slider(self, value):
        mu_val = value / 10.0
        self.mu_spin.blockSignals(True)
        self.mu_spin.setValue(mu_val)
        self.mu_spin.blockSignals(False)
        self.update_rho_display()

    def on_mu_spin(self, value):
        self.mu_slider.blockSignals(True)
        self.mu_slider.setValue(int(value * 10))
        self.mu_slider.blockSignals(False)
        self.update_rho_display()

    def update_rho_display(self):
        lambda_val = self.lambda_spin.value()
        mu_val = self.mu_spin.value()
        rho = lambda_val / mu_val if mu_val > 0 else 0
        if rho < 1:
            status = "✅ Стационарный режим"
            color = "#00ff88"
        elif rho == 1:
            status = "⚠️ Критическая загрузка"
            color = "#ffaa00"
        else:
            status = "❌ Нестационарный режим"
            color = "#ff5555"
        self.rho_label.setText(f"Загрузка ρ = {rho:.4f} | {status}")
        self.rho_label.setStyleSheet(f"""
            background-color: #1a2a4f;
            border-radius: 8px;
            padding: 10px;
            font-size: 13px;
            color: {color};
        """)

    def start_simulation(self):
        lambda_val = self.lambda_spin.value()
        mu_val = self.mu_spin.value()
        sim_time = self.time_spin.value()

        if mu_val <= lambda_val:
            # Можно предупредить, но разрешить запуск
            pass

        self.run_button.setEnabled(False)
        self.run_button.setText("⏳ МОДЕЛИРОВАНИЕ...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = SimulationWorker(lambda_val, mu_val, sim_time)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.start()

    def on_simulation_finished(self, results, time_points, queue_lengths):
        self.run_button.setEnabled(True)
        self.run_button.setText("🚀 ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ")
        self.progress_bar.setVisible(False)

        # Обновление графика
        self.plot_canvas.plot_queue(time_points, queue_lengths,
                                    results['lambda'], results['mu'])

        # Формирование текста статистики
        stats_text = f"""
        <b>ИМИТАЦИОННЫЕ ПОКАЗАТЕЛИ</b><br>
        <br>
        Обслужено заявок: {results['served']}<br>
        Коэфф. использования: {results['utilization']:.4f}<br>
        Среднее число в системе L: {results['L_sim']:.4f}<br>
        Средняя длина очереди Lq: {results['Lq_sim']:.4f}<br>
        Среднее время в системе W: {results['W_sim']:.4f}<br>
        Среднее время ожидания Wq: {results['Wq_sim']:.4f}<br>
        <br>
        <b>АНАЛИТИЧЕСКИЕ ЗНАЧЕНИЯ (M/M/1)</b><br>
        <br>
        L (теор): {results['L_analytical']:.4f}<br>
        Lq (теор): {results['Lq_analytical']:.4f}<br>
        W (теор): {results['W_analytical']:.4f}<br>
        Wq (теор): {results['Wq_analytical']:.4f}<br>
        """
        self.stats_display.setText(stats_text)

        # Информационная строка
        self.info_text.setText(
            f"Моделирование завершено | λ={results['lambda']:.2f}, μ={results['mu']:.2f} | "
            f"ρ={results['rho']:.3f} | Обслужено: {results['served']}"
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
