import sys
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, QTimer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

class Request:
    _id_counter = 0

    def __init__(self, arrival_time: float, patience: float = float('inf')):
        self.id = Request._id_counter
        Request._id_counter += 1
        self.arrival_time = arrival_time
        self.patience = patience
        self.leave_time = arrival_time + patience if patience < float('inf') else float('inf')
        self.start_service_time = None
        self.departure_time = None
        self.is_served = False
        self.is_impatient_leave = False

    def __repr__(self):
        return f"Request(id={self.id}, arr={self.arrival_time:.3f})"


class Server:
    def __init__(self, server_id: int):
        self.id = server_id
        self.busy = False
        self.current_request: Request = None
        self.busy_end_time: float = None

    def start_service(self, request: Request, service_time: float, current_time: float):
        self.busy = True
        self.current_request = request
        self.busy_end_time = current_time + service_time
        request.start_service_time = current_time
        request.departure_time = current_time + service_time

    def finish_service(self):
        req = self.current_request
        req.is_served = True
        self.busy = False
        self.current_request = None
        self.busy_end_time = None
        return req

    def __repr__(self):
        return f"Server(id={self.id}, busy={self.busy})"


class Event:
    ARRIVAL = 'ARRIVAL'
    DEPARTURE = 'DEPARTURE'
    IMPATIENCE = 'IMPATIENCE'

    def __init__(self, time: float, etype: str, request: Request = None, server: Server = None):
        self.time = time
        self.type = etype
        self.request = request
        self.server = server

    def __lt__(self, other):
        return self.time < other.time


class SimulationResults:
    def __init__(self):
        self.state_counts = None
        self.wait_times = []
        self.blocked_count = 0
        self.impatient_count = 0
        self.served_count = 0
        self.total_time = 0.0
        self.server_utilization = 0.0
        self.avg_queue_length = 0.0


class MMcKImpatientSimulator:
    def __init__(self, lam: float, mu: float, c: int, K: int, nu: float):
        self.lam = lam
        self.mu = mu
        self.c = c
        self.K = K
        self.nu = nu

    def simulate(self, T_end: float, warmup: float = 0.0, progress_callback=None) -> SimulationResults:
        queue = deque()
        servers = [Server(i) for i in range(self.c)]
        events = []

        state_changes = []
        wait_times_collected = []
        blocked = 0
        impatient = 0
        served = 0
        server_busy_time_total = 0.0
        last_event_time = 0.0
        last_state = 0

        next_arrival_time = np.random.exponential(1.0 / self.lam)
        events.append(Event(next_arrival_time, Event.ARRIVAL))

        next_progress_update = 0.0
        progress_step = max(1, int(T_end / 100))

        while events and last_event_time < T_end:
            events.sort(key=lambda e: e.time)
            current_event = events.pop(0)
            t = current_event.time

            if t >= T_end:
                break

            delta = t - last_event_time
            if last_event_time >= warmup:
                state_changes.append((last_state, delta))
                busy_count = sum(1 for s in servers if s.busy)
                server_busy_time_total += busy_count * delta

            if current_event.type == Event.ARRIVAL:
                self._handle_arrival(current_event, events, queue, servers, t,
                                     warmup, blocked, impatient, wait_times_collected)

                next_arr = t + np.random.exponential(1.0 / self.lam)
                if next_arr < T_end:
                    events.append(Event(next_arr, Event.ARRIVAL))

            elif current_event.type == Event.DEPARTURE:
                self._handle_departure(current_event, queue, servers, t,
                                       warmup, wait_times_collected, events, served)

            elif current_event.type == Event.IMPATIENCE:
                self._handle_impatience(current_event, queue, servers, t,
                                        warmup, impatient)

            last_state = self._current_state(queue, servers)
            last_event_time = t

            if progress_callback and t >= next_progress_update:
                progress = min(100, int(t / T_end * 100))
                progress_callback(progress)
                next_progress_update += progress_step

        if last_event_time < T_end:
            delta = T_end - last_event_time
            if last_event_time >= warmup:
                state_changes.append((last_state, delta))
                busy_count = sum(1 for s in servers if s.busy)
                server_busy_time_total += busy_count * delta

        res = SimulationResults()
        total_stat_time = T_end - warmup
        res.total_time = total_stat_time
        res.served_count = served
        res.blocked_count = blocked
        res.impatient_count = impatient

        if state_changes:
            max_state = max(s for s, _ in state_changes)
            state_counts = np.zeros(max_state + 1)
            for s, dur in state_changes:
                state_counts[s] += dur
            res.state_counts = state_counts
            if total_stat_time > 0:
                res.avg_queue_length = np.sum(np.arange(len(state_counts)) * state_counts) / total_stat_time
        else:
            res.state_counts = np.array([0.0])

        if total_stat_time > 0:
            res.server_utilization = server_busy_time_total / (self.c * total_stat_time)

        res.wait_times = wait_times_collected if warmup >= 0 else []
        return res

    def _current_state(self, queue, servers):
        busy = sum(1 for s in servers if s.busy)
        return len(queue) + busy

    def _find_free_server(self, servers):
        for server in servers:
            if not server.busy:
                return server
        return None

    def _handle_arrival(self, event, events, queue, servers, t, warmup,
                        blocked, impatient, wait_times):
        req = Request(arrival_time=t, patience=float('inf') if self.nu == 0 else np.random.exponential(1.0 / self.nu))

        free_server = self._find_free_server(servers)
        if free_server:
            service_time = np.random.exponential(1.0 / self.mu)
            free_server.start_service(req, service_time, t)
            events.append(Event(t + service_time, Event.DEPARTURE, request=req, server=free_server))
        else:
            if len(queue) < self.K:
                queue.append(req)
                if self.nu > 0 and req.leave_time < float('inf'):
                    events.append(Event(req.leave_time, Event.IMPATIENCE, request=req))
            else:
                if t >= warmup:
                    blocked += 1

    def _handle_departure(self, event, queue, servers, t, warmup,
                          wait_times, events, served_counter):
        server = event.server
        req = server.finish_service()
        if t >= warmup:
            served_counter += 1
            wait_times.append(t - req.arrival_time)

        if queue:
            next_req = queue.popleft()
            events = [e for e in events if not (e.type == Event.IMPATIENCE and e.request == next_req)]
            service_time = np.random.exponential(1.0 / self.mu)
            server.start_service(next_req, service_time, t)
            events.append(Event(t + service_time, Event.DEPARTURE, request=next_req, server=server))

    def _handle_impatience(self, event, queue, servers, t, warmup, impatient_counter):
        req = event.request

        if req in queue:
            queue.remove(req)
            if t >= warmup:
                impatient_counter += 1
                req.is_impatient_leave = True


# ----------------------------------------------------------------------
# Компонент визуализации графика
# ----------------------------------------------------------------------
class ModernPlotCanvas(FigureCanvas):
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
        for spine in self.ax.spines.values():
            spine.set_color('#2a3a5a')
        self.ax.tick_params(colors='#c0d4f0', labelsize=10)
        self.ax.xaxis.label.set_color('#00d4ff')
        self.ax.yaxis.label.set_color('#00d4ff')
        self.ax.title.set_color('#ffffff')

    def plot_results(self, state_counts: np.ndarray):
        """Построение эмпирической гистограммы распределения числа заявок в системе."""
        self.ax.clear()
        self.legend_ax.clear()
        self.legend_ax.axis('off')
        self.setup_styles()

        total = state_counts.sum()
        if total > 0:
            probs = state_counts / total
        else:
            probs = np.zeros_like(state_counts, dtype=float)

        states = np.arange(len(state_counts))
        self.ax.bar(states, probs, color='#00d4ff', alpha=0.7,
                    edgecolor='#0088cc', linewidth=1.5)
        self.ax.grid(True, alpha=0.2, linestyle='--', color='#2a3a5a')
        self.ax.set_xlabel('Число заявок в системе', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Вероятность', fontsize=12, fontweight='bold')
        self.ax.set_title('Эмпирическое распределение числа заявок', fontsize=13,
                          fontweight='bold', pad=20)

        # Аннотация
        mean_val = np.average(states, weights=probs) if total > 0 else 0
        textstr = f'Среднее (эмп.): {mean_val:.3f}'
        props = dict(boxstyle='round', facecolor='#1a1f3a', alpha=0.8,
                     edgecolor='#00d4ff', linewidth=1.5)
        self.ax.text(0.95, 0.95, textstr, transform=self.ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props, color='#c0d4f0')

        legend_elements = [
            matplotlib.patches.Patch(facecolor='#00d4ff', alpha=0.7, edgecolor='#0088cc',
                                     label='Эмпирическое распределение')
        ]
        self.legend_ax.legend(handles=legend_elements, loc='center', fontsize=10,
                              framealpha=0.9, facecolor='#1a1f3a', edgecolor='#00d4ff')
        self.draw()


# ----------------------------------------------------------------------
# Главное окно приложения
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("M/M/c/K + нетерпеливые заявки | Имитационное моделирование")
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

        self.simulation_results = None

    # ------------------------------------------------------------------
    # Панель управления
    # ------------------------------------------------------------------
    def create_control_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        header = QLabel("⚙️ ПАРАМЕТРЫ СМО")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff; padding: 10px;")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2a3a5a; max-height: 2px;")
        layout.addWidget(sep)

        # Интенсивности
        flow_group = QGroupBox("Интенсивности")
        flow_layout = QVBoxLayout()

        # λ
        lambda_layout = QVBoxLayout()
        lambda_layout.addWidget(QLabel("λ (заявок/сек):"))
        self.lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.lambda_slider.setRange(10, 300)       # 0.1 .. 3.0
        self.lambda_slider.setValue(150)           # 1.5
        self.lambda_slider.valueChanged.connect(self.on_lambda_changed)
        lambda_layout.addWidget(self.lambda_slider)

        lambda_val_layout = QHBoxLayout()
        self.lambda_value_label = QLabel("1.500")
        self.lambda_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        self.lambda_spinbox = QDoubleSpinBox()
        self.lambda_spinbox.setRange(0.1, 3.0)
        self.lambda_spinbox.setValue(1.5)
        self.lambda_spinbox.setSingleStep(0.1)
        self.lambda_spinbox.valueChanged.connect(self.on_lambda_spinbox_changed)
        lambda_val_layout.addWidget(QLabel("Знач.:"))
        lambda_val_layout.addWidget(self.lambda_spinbox)
        lambda_val_layout.addWidget(self.lambda_value_label)
        lambda_layout.addLayout(lambda_val_layout)
        flow_layout.addLayout(lambda_layout)

        # μ
        mu_layout = QVBoxLayout()
        mu_layout.addWidget(QLabel("μ (обслуживаний/сек):"))
        self.mu_slider = QSlider(Qt.Orientation.Horizontal)
        self.mu_slider.setRange(10, 300)
        self.mu_slider.setValue(100)               # 1.0
        self.mu_slider.valueChanged.connect(self.on_mu_changed)
        mu_layout.addWidget(self.mu_slider)

        mu_val_layout = QHBoxLayout()
        self.mu_value_label = QLabel("1.000")
        self.mu_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        self.mu_spinbox = QDoubleSpinBox()
        self.mu_spinbox.setRange(0.1, 3.0)
        self.mu_spinbox.setValue(1.0)
        self.mu_spinbox.setSingleStep(0.1)
        self.mu_spinbox.valueChanged.connect(self.on_mu_spinbox_changed)
        mu_val_layout.addWidget(QLabel("Знач.:"))
        mu_val_layout.addWidget(self.mu_spinbox)
        mu_val_layout.addWidget(self.mu_value_label)
        mu_layout.addLayout(mu_val_layout)
        flow_layout.addLayout(mu_layout)

        # nu (интенсивность ухода)
        nu_layout = QVBoxLayout()
        nu_layout.addWidget(QLabel("ν (интенсивность ухода из очереди):"))
        self.nu_slider = QSlider(Qt.Orientation.Horizontal)
        self.nu_slider.setRange(0, 200)            # 0 .. 2.0
        self.nu_slider.setValue(50)                # 0.5
        self.nu_slider.valueChanged.connect(self.on_nu_changed)
        nu_layout.addWidget(self.nu_slider)

        nu_val_layout = QHBoxLayout()
        self.nu_value_label = QLabel("0.500")
        self.nu_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        self.nu_spinbox = QDoubleSpinBox()
        self.nu_spinbox.setRange(0.0, 2.0)
        self.nu_spinbox.setValue(0.5)
        self.nu_spinbox.setSingleStep(0.1)
        self.nu_spinbox.valueChanged.connect(self.on_nu_spinbox_changed)
        nu_val_layout.addWidget(QLabel("Знач.:"))
        nu_val_layout.addWidget(self.nu_spinbox)
        nu_val_layout.addWidget(self.nu_value_label)
        nu_layout.addLayout(nu_val_layout)
        flow_layout.addLayout(nu_layout)

        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)

        # Число каналов и буфер
        resources_group = QGroupBox("Ресурсы")
        res_layout = QVBoxLayout()

        res_layout.addWidget(QLabel("Число каналов (c):"))
        self.c_spinbox = QSpinBox()
        self.c_spinbox.setRange(1, 20)
        self.c_spinbox.setValue(2)
        res_layout.addWidget(self.c_spinbox)

        res_layout.addWidget(QLabel("Размер очереди (K):"))
        self.K_spinbox = QSpinBox()
        self.K_spinbox.setRange(0, 50)
        self.K_spinbox.setValue(5)
        res_layout.addWidget(self.K_spinbox)

        resources_group.setLayout(res_layout)
        layout.addWidget(resources_group)

        # Время моделирования
        sim_group = QGroupBox("Моделирование")
        sim_layout = QVBoxLayout()

        sim_layout.addWidget(QLabel("Общее время (сек):"))
        self.T_spinbox = QDoubleSpinBox()
        self.T_spinbox.setRange(100, 50000)
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

        # Загрузка системы (ρ)
        self.rho_label = QLabel()
        self.rho_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rho_label.setStyleSheet("""
            background-color: #1a2a4f; border-radius: 8px; padding: 10px;
            font-size: 13px; color: #00d4ff;
        """)
        self.update_rho_display()
        layout.addWidget(self.rho_label)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Кнопка запуска
        self.run_button = QPushButton("🚀 ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setMinimumHeight(45)
        layout.addWidget(self.run_button)

        # Вывод характеристик
        stats_group = QGroupBox("Результаты")
        stats_layout = QVBoxLayout()
        self.stats_display = QLabel()
        self.stats_display.setStyleSheet("""
            background-color: #1a1f3a; border-radius: 8px; padding: 12px;
            font-family: monospace; font-size: 11px;
        """)
        self.stats_display.setWordWrap(True)
        stats_layout.addWidget(self.stats_display)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Визуализация
    # ------------------------------------------------------------------
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
                background-color: #1a2a4f; border-radius: 8px; padding: 8px;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        self.info_text = QLabel("Настройте параметры и запустите моделирование.")
        self.info_text.setStyleSheet("color: #c0d4f0; font-size: 11px;")
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_frame)

        return panel

    # ------------------------------------------------------------------
    # Слоты для связанных параметров
    # ------------------------------------------------------------------
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

    def on_nu_changed(self, value):
        nu = value / 100.0
        self.nu_spinbox.blockSignals(True)
        self.nu_spinbox.setValue(nu)
        self.nu_spinbox.blockSignals(False)
        self.nu_value_label.setText(f"{nu:.3f}")

    def on_nu_spinbox_changed(self, value):
        self.nu_slider.blockSignals(True)
        self.nu_slider.setValue(int(value * 100))
        self.nu_slider.blockSignals(False)
        self.nu_value_label.setText(f"{value:.3f}")

    def update_rho_display(self):
        lam = self.lambda_spinbox.value()
        mu = self.mu_spinbox.value()
        c = self.c_spinbox.value()
        rho = lam / (c * mu) if (c * mu) > 0 else 0
        self.rho_label.setText(f"Загрузка системы на один канал: ρ = {rho:.4f}")

    # ------------------------------------------------------------------
    # Запуск моделирования
    # ------------------------------------------------------------------
    def run_simulation(self):
        self.run_button.setEnabled(False)
        self.run_button.setText("⏳ ИДЁТ МОДЕЛИРОВАНИЕ...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Считываем параметры
        lam = self.lambda_spinbox.value()
        mu = self.mu_spinbox.value()
        nu = self.nu_spinbox.value()
        c = self.c_spinbox.value()
        K = self.K_spinbox.value()
        T_end = self.T_spinbox.value()
        warmup = self.warmup_spinbox.value()

        sim = MMcKImpatientSimulator(lam, mu, c, K, nu)

        def update_progress(value):
            self.progress_bar.setValue(value)
            QApplication.processEvents()

        def run_in_background():
            results = sim.simulate(T_end, warmup, update_progress)
            self.simulation_results = results
            self.display_results(results, lam, mu, nu, c, K)
            self.run_button.setEnabled(True)
            self.run_button.setText("🚀 ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ")
            self.progress_bar.setVisible(False)

        QTimer.singleShot(50, run_in_background)

    def display_results(self, results: SimulationResults,
                        lam, mu, nu, c, K):
        # График
        self.plot_canvas.plot_results(results.state_counts)

        # Текстовые показатели
        L = results.avg_queue_length
        U = results.server_utilization
        W = np.mean(results.wait_times) if results.wait_times else 0.0
        total_arrived = results.served_count + results.blocked_count + results.impatient_count
        p_block = results.blocked_count / total_arrived if total_arrived > 0 else 0
        p_impatient = results.impatient_count / total_arrived if total_arrived > 0 else 0

        stats_text = (
            f"<b>ОПЕРАЦИОННЫЕ ХАРАКТЕРИСТИКИ</b><br>"
            f"Среднее число заявок в системе L: {L:.4f}<br>"
            f"Коэффициент использования каналов U: {U:.4f}<br>"
            f"Среднее время ожидания (обслуженных) W: {W:.4f} сек<br>"
            f"<br>"
            f"<b>ВЕРОЯТНОСТНЫЕ ПОКАЗАТЕЛИ</b><br>"
            f"Всего прибыло заявок (после разогрева): {total_arrived}<br>"
            f"Обслужено: {results.served_count}<br>"
            f"Отказов (переполнение очереди): {results.blocked_count}  (P_block = {p_block:.4f})<br>"
            f"Уходов по нетерпеливости: {results.impatient_count}  (P_imp = {p_impatient:.4f})<br>"
        )

        self.stats_display.setText(stats_text)
        self.info_text.setText(
            f"λ={lam:.2f}, μ={mu:.2f}, ν={nu:.2f}, c={c}, K={K}. "
            f"L={L:.3f}, U={U:.3f}, W={W:.3f} с."
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
