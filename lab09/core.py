import sys
import numpy as np
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
    background-color: #0a0e17;
}

QGroupBox {
    font-size: 13px;
    font-weight: bold;
    color: #00d4ff;
    border: 1px solid #1a2a4f;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 15px;
    background-color: #0f172a;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 8px;
}

QLabel {
    color: #94a3b8;
    font-size: 12px;
}

QPushButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                stop: 0 #0ea5e9, stop: 1 #0284c7);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 12px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                stop: 0 #38bdf8, stop: 1 #0ea5e9);
}

QPushButton:pressed {
    background: #0369a1;
}

QDoubleSpinBox {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px;
    color: #38bdf8;
    font-size: 12px;
}

QDoubleSpinBox:focus {
    border: 1px solid #0ea5e9;
}

QProgressBar {
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    color: white;
    background-color: #0f172a;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #0ea5e9;
    border-radius: 3px;
}
"""

class ModernPlotCanvas(FigureCanvas):
    """Холст для построения распределения состояний в системе M/M/1/0."""
    def __init__(self, parent=None, width=12, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0e17')
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111, facecolor='#0f172a')
        self.fig.subplots_adjust(right=0.75, left=0.1)
        self.legend_ax = self.fig.add_axes([0.78, 0.3, 0.18, 0.4])
        self.legend_ax.axis('off')
        self.setup_styles()

    def setup_styles(self):
        """Оформление координатных осей."""
        for spine in self.ax.spines.values():
            spine.set_color('#1e293b')
        self.ax.tick_params(colors='#94a3b8', labelsize=10)
        self.ax.xaxis.label.set_color('#00d4ff')
        self.ax.yaxis.label.set_color('#00d4ff')
        self.ax.title.set_color('#ffffff')

    def plot_results(self, state_counts: np.ndarray, rho: float):
        """
        Построение гистограммы распределения состояний системы
        и теоретического распределения (формула Эрланга для c=1).
        """
        self.ax.clear()
        self.legend_ax.clear()
        self.legend_ax.axis('off')
        self.setup_styles()

        # Нормировка эмпирических частот
        empirical_probs = np.zeros(2)
        total = state_counts.sum()
        if total > 0:
            for i in range(min(2, len(state_counts))):
                empirical_probs[i] = state_counts[i] / total

        states = np.array([0, 1])

        # Отрисовка столбцов эмпирических вероятностей
        self.ax.bar(states, empirical_probs, color='#0ea5e9', alpha=0.7,
                    edgecolor='#38bdf8', linewidth=1.5, width=0.4, label='Эмпирическое')

        # Теоретические вероятности по формуле Эрланга для M/M/1/0
        p0_theor = 1 / (1 + rho)
        p1_theor = rho / (1 + rho)
        theoretical_probs = np.array([p0_theor, p1_theor])

        # Отрисовка теоретического графика
        self.ax.plot(states, theoretical_probs, 'o-', color='#f43f5e',
                    markersize=10, linewidth=2.5, markerfacecolor='#f43f5e',
                    markeredgecolor='white', markeredgewidth=1.5, label='Теоретическое')

        self.ax.grid(True, alpha=0.15, linestyle='--', color='#334155')
        self.ax.set_xticks([0, 1])
        self.ax.set_xticklabels(['0 (Свободен)', '1 (Занят)'], fontsize=10)
        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(0, 1.1)
        
        self.ax.set_xlabel('Состояние системы (число заявок)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Вероятность состояния', fontsize=12, fontweight='bold')
        self.ax.set_title(f'Распределение состояний в M/M/1/0 (без очереди)\n'
                         f'Загрузка ρ = {rho:.3f}', fontsize=13, fontweight='bold', pad=20)

        # Легенда
        legend_elements = [
            Patch(facecolor='#0ea5e9', alpha=0.7, edgecolor='#38bdf8', label='Эмпирическое'),
            Line2D([0], [0], marker='o', color='#f43f5e', markerfacecolor='#f43f5e',
                   markeredgecolor='white', linewidth=2.5, markersize=8,
                   label='Теор. P(n) [ф-ла Эрланга]')
        ]
        self.legend_ax.legend(handles=legend_elements, loc='center', fontsize=10,
                              framealpha=0.9, facecolor='#0f172a', edgecolor='#1e293b')

        # Информационная панель на графике
        textstr = (
            f"P0 (теор.): {p0_theor:.3f}\nP0 (эмп.): {empirical_probs[0]:.3f}\n\n"
            f"P1 (теор.): {p1_theor:.3f}\nP1 (эмп.): {empirical_probs[1]:.3f}"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='#0f172a', alpha=0.8,
                     edgecolor='#1e293b', linewidth=1.5)
        self.ax.text(0.95, 0.95, textstr, transform=self.ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props, color='#94a3b8')

        self.draw()


class MM10Simulator:
    """Имитационный симулятор СМО типа M/M/1/0 (без очереди)."""
    def __init__(self):
        pass

    def simulate(self, lam: float, mu: float, T_end: float, warmup: float = 0.0, progress_callback=None) -> dict:
        server_busy = False
        events = []

        t_arrival = np.random.exponential(1.0 / lam)
        events.append((t_arrival, 'arrival'))

        state_changes = []
        system_times = []
        last_event_time = 0.0
        
        num_arrivals = 0
        num_served = 0
        num_rejected = 0
        
        current_customer_arrival = 0.0

        next_progress_update = 0.0
        progress_step = max(1, int(T_end / 100))

        while events and last_event_time < T_end:
            events.sort(key=lambda x: x[0])
            t, etype = events.pop(0)

            if t >= T_end:
                break

            delta = t - last_event_time
            current_state = 1 if server_busy else 0
            
            if last_event_time >= warmup:
                state_changes.append((current_state, delta))

            if etype == 'arrival':
                if t >= warmup:
                    num_arrivals += 1
                
                if not server_busy:
                    server_busy = True
                    current_customer_arrival = t
                    service_time = np.random.exponential(1.0 / mu)
                    events.append((t + service_time, 'departure'))
                    if t >= warmup:
                        num_served += 1
                else:
                    if t >= warmup:
                        num_rejected += 1

                next_arrival = t + np.random.exponential(1.0 / lam)
                if next_arrival < T_end:
                    events.append((next_arrival, 'arrival'))

            elif etype == 'departure':
                server_busy = False
                if t >= warmup:
                    system_times.append(t - current_customer_arrival)
            
            last_event_time = t

            if progress_callback and t >= next_progress_update:
                progress = min(100, int(t / T_end * 100))
                progress_callback(progress)
                next_progress_update += progress_step

        if last_event_time < T_end:
            delta = T_end - last_event_time
            current_state = 1 if server_busy else 0
            if last_event_time >= warmup:
                state_changes.append((current_state, delta))

        state_counts = np.zeros(2)
        for s, duration in state_changes:
            if s < 2:
                state_counts[s] += duration

        return {
            'states': state_counts,
            'system_times': system_times,
            'total_time': T_end - warmup,
            'num_arrivals': num_arrivals,
            'num_served': num_served,
            'num_rejected': num_rejected
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Имитационное моделирование M/M/1/0 | СМО без очереди")
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

        self.simulator = MM10Simulator()
        self.current_results = None

        self.update_rho_display()

    def create_control_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        header = QLabel("⚙️ КОНФИГУРАЦИЯ M/M/1/0")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #00d4ff; padding: 5px;")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #1e293b; max-height: 1px;")
        layout.addWidget(sep)

        # Параметры λ и μ
        flow_group = QGroupBox("Интенсивности потоков")
        flow_layout = QVBoxLayout()

        # λ
        lambda_layout = QVBoxLayout()
        lambda_layout.addWidget(QLabel("Интенсивность поступления λ (заявок/сек):"))
        self.lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.lambda_slider.setRange(10, 1000)  # 0.1 .. 10.0 (масштаб x100)
        self.lambda_slider.setValue(150)      # 1.5
        self.lambda_slider.valueChanged.connect(self.on_lambda_changed)
        lambda_layout.addWidget(self.lambda_slider)

        lambda_value_layout = QHBoxLayout()
        self.lambda_value_label = QLabel("1.500")
        self.lambda_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lambda_value_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #00d4ff;")
        self.lambda_spinbox = QDoubleSpinBox()
        self.lambda_spinbox.setRange(0.1, 10.0)
        self.lambda_spinbox.setValue(1.5)
        self.lambda_spinbox.setSingleStep(0.1)
        self.lambda_spinbox.valueChanged.connect(self.on_lambda_spinbox_changed)
        lambda_value_layout.addWidget(QLabel("Значение:"))
        lambda_value_layout.addWidget(self.lambda_spinbox)
        lambda_value_layout.addWidget(self.lambda_value_label)
        lambda_layout.addLayout(lambda_value_layout)
        flow_layout.addLayout(lambda_layout)

        # μ
        mu_layout = QVBoxLayout()
        mu_layout.addWidget(QLabel("Интенсивность обслуживания μ (требований/сек):"))
        self.mu_slider = QSlider(Qt.Orientation.Horizontal)
        self.mu_slider.setRange(10, 1000)  # 0.1 .. 10.0
        self.mu_slider.setValue(200)      # 2.0
        self.mu_slider.valueChanged.connect(self.on_mu_changed)
        mu_layout.addWidget(self.mu_slider)

        mu_value_layout = QHBoxLayout()
        self.mu_value_label = QLabel("2.000")
        self.mu_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.mu_value_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #00d4ff;")
        self.mu_spinbox = QDoubleSpinBox()
        self.mu_spinbox.setRange(0.1, 10.0)
        self.mu_spinbox.setValue(2.0)
        self.mu_spinbox.setSingleStep(0.1)
        self.mu_spinbox.valueChanged.connect(self.on_mu_spinbox_changed)
        mu_value_layout.addWidget(QLabel("Значение:"))
        mu_value_layout.addWidget(self.mu_spinbox)
        mu_value_layout.addWidget(self.mu_value_label)
        mu_layout.addLayout(mu_value_layout)
        flow_layout.addLayout(mu_layout)

        # Отображение параметров нагрузки
        self.rho_display = QLabel()
        self.rho_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rho_display.setStyleSheet("""
            background-color: #0b1329;
            border-radius: 6px;
            padding: 8px;
            font-size: 12px;
            color: #00d4ff;
            border: 1px solid #1e293b;
        """)
        flow_layout.addWidget(self.rho_display)

        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)

        # Параметры прогона
        sim_group = QGroupBox("Параметры моделирования")
        sim_layout = QVBoxLayout()

        sim_layout.addWidget(QLabel("Время прогона (сек):"))
        self.T_spinbox = QDoubleSpinBox()
        self.T_spinbox.setRange(100, 100000)
        self.T_spinbox.setValue(10000)
        self.T_spinbox.setSingleStep(1000)
        sim_layout.addWidget(self.T_spinbox)

        sim_layout.addWidget(QLabel("Период прогрева (сек):"))
        self.warmup_spinbox = QDoubleSpinBox()
        self.warmup_spinbox.setRange(0, 10000)
        self.warmup_spinbox.setValue(500)
        self.warmup_spinbox.setSingleStep(100)
        sim_layout.addWidget(self.warmup_spinbox)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(18)
        layout.addWidget(self.progress_bar)

        # Кнопка запуска
        self.run_button = QPushButton("🚀 ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setMinimumHeight(45)
        layout.addWidget(self.run_button)

        # Характеристики
        stats_group = QGroupBox("Операционные характеристики")
        stats_layout = QVBoxLayout()
        self.stats_display = QLabel("Нажмите кнопку запуска для проведения расчетов.")
        self.stats_display.setStyleSheet("""
            background-color: #0b1329;
            border-radius: 6px;
            padding: 10px;
            font-family: Consolas, monospace;
            font-size: 11px;
            color: #e2e8f0;
            border: 1px solid #1e293b;
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

        header = QLabel("📊 РАСПРЕДЕЛЕНИЕ ЧИСЛА ЗАЯВОК В СИСТЕМЕ (M/M/1/0)")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #00d4ff; padding: 5px;")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #1e293b; max-height: 1px;")
        layout.addWidget(sep)

        self.plot_canvas = ModernPlotCanvas(width=10, height=7)
        layout.addWidget(self.plot_canvas)

        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border-radius: 6px;
                border: 1px solid #1e293b;
                padding: 6px;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        self.info_text = QLabel("Система M/M/1/0 всегда стабильна, так как нет бесконечно растущей очереди.")
        self.info_text.setStyleSheet("color: #94a3b8; font-size: 11px;")
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
        self.rho_display.setText(f"Приведенная нагрузка ρ = λ/μ = {rho:.4f}\n")

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
        system_times = results['system_times']
        num_arrivals = results['num_arrivals']
        num_rejected = results['num_rejected']
        num_served = results['num_served']

        # Эмпирические значения
        U_emp = state_counts[1] / total_time if total_time > 0 else 0
        L_emp = U_emp  # Число требований в системе 0 или 1, среднее равно вероятности занятости
        W_emp = np.mean(system_times) if system_times else 0
        P_loss_emp = num_rejected / num_arrivals if num_arrivals > 0 else 0

        # Теоретические значения (формула Эрланга для c=1 канала)
        P1_theor = rho / (1 + rho)
        L_theor = P1_theor
        U_theor = P1_theor
        W_theor = 1 / mu if mu > 0 else 0
        P_loss_theor = P1_theor

        # Отрисовка результатов
        self.plot_canvas.plot_results(state_counts, rho)

        # Вывод показателей
        stats_text = (
            f"<b>ЭМПИРИКА (после разогрева):</b><br>"
            f"• Ср. число заявок в системе L: <font color='#00d4ff'>{L_emp:.4f}</font><br>"
            f"• Загрузка прибора U: <font color='#00d4ff'>{U_emp:.4f}</font><br>"
            f"• Вероятность отказа P<sub>отк</sub>: <font color='#f43f5e'>{P_loss_emp:.4f}</font><br>"
            f"• Время в системе W (обслуж.): <font color='#00d4ff'>{W_emp:.4f}</font> с<br>"
            f"• Поступило заявок за прогон: {num_arrivals}<br>"
            f"• Обслужено / Отклонено: {num_served} / {num_rejected}<br>"
            f"<br>"
            f"<b>ТЕОРИЯ (M/M/1/0):</b><br>"
            f"• L = U = ρ/(1+ρ) = <font color='#00d4ff'>{L_theor:.4f}</font><br>"
            f"• P<sub>отк</sub> = ρ/(1+ρ) = <font color='#f43f5e'>{P_loss_theor:.4f}</font><br>"
            f"• W = 1/μ = <font color='#00d4ff'>{W_theor:.4f}</font> с<br>"
            f"<br>"
        )

        if abs(L_emp - L_theor) < 0.03 and abs(P_loss_emp - P_loss_theor) < 0.03:
            stats_text += '<span style="color:#10b981; font-weight:bold;">✓ Эмпирические данные соответствуют теории.</span>'
        else:
            stats_text += '<span style="color:#f59e0b; font-weight:bold;">⚠ Есть расхождения. Рекомендуется увеличить время прогона.</span>'

        self.stats_display.setText(stats_text)

        self.info_text.setText(
            f"Прогон завершен за {total_time:.0f} с. Прибыло: {num_arrivals}. "
            f"Обслужено: {num_served}, отклонено: {num_rejected} ({P_loss_emp * 100:.1f}% потерь)."
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
