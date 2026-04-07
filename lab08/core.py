# core.py
import sys
import numpy as np
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
from scipy import stats

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
    def __init__(self, parent=None, width=16, height=10, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0e27')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Основная область графика
        self.ax = self.fig.add_subplot(111, facecolor='#0f1433')
        
        # Отдельная область для легенды справа
        self.fig.subplots_adjust(right=0.75, left=0.1)
        self.legend_ax = self.fig.add_axes([0.78, 0.3, 0.18, 0.4])
        self.legend_ax.axis('off')
        
        self.setup_styles()
        
    def setup_styles(self):
        """Настройка стилей графика"""
        self.ax.spines['top'].set_color('#2a3a5a')
        self.ax.spines['right'].set_color('#2a3a5a')
        self.ax.spines['bottom'].set_color('#2a3a5a')
        self.ax.spines['left'].set_color('#2a3a5a')
        self.ax.tick_params(colors='#c0d4f0', labelsize=10)
        self.ax.xaxis.label.set_color('#00d4ff')
        self.ax.yaxis.label.set_color('#00d4ff')
        self.ax.title.set_color('#ffffff')
        
    def plot_results(self, data: np.ndarray, λT: float, lambda_val: float, T: float):
        """Построение графика с вынесенной легендой"""
        self.ax.clear()
        self.legend_ax.clear()
        self.legend_ax.axis('off')
        self.setup_styles()
        
        # Эмпирическая гистограмма
        max_val = max(data)
        bins = np.arange(-0.5, max_val + 1.5, 1)
        counts, bins, patches = self.ax.hist(data, bins=bins, density=True, 
                                             alpha=0.7, color='#00d4ff', 
                                             edgecolor='#0088cc', linewidth=1.5)
        
        # Теоретическое распределение Пуассона
        k_values = np.arange(0, max_val + 2)
        poisson_probs = stats.poisson.pmf(k_values, λT)
        self.ax.plot(k_values, poisson_probs, 'o-', color='#ff6b6b', 
                    markersize=8, linewidth=2.5,
                    markerfacecolor='#ff6b6b', markeredgecolor='white', markeredgewidth=1.5)
        
        # Заливка под кривой
        self.ax.fill_between(k_values, poisson_probs, alpha=0.2, color='#ff6b6b')
        
        # Сетка
        self.ax.grid(True, alpha=0.2, linestyle='--', color='#2a3a5a')
        self.ax.set_xlabel('Число запросов за интервал T', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Вероятность', fontsize=12, fontweight='bold')
        self.ax.set_title(f'Анализ пуассоновского потока запросов\nλ = {lambda_val} зап/сек | T = {T} сек | λT = {λT:.2f}', 
                         fontsize=13, fontweight='bold', pad=20)
        
        # Создаем элементы легенды на отдельной оси
        legend_elements = [
            matplotlib.patches.Patch(facecolor='#00d4ff', alpha=0.7, edgecolor='#0088cc', 
                                    label='Эмпирическое распределение'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='#ff6b6b', 
                                   markerfacecolor='#ff6b6b', markeredgecolor='white',
                                   linewidth=2.5, markersize=8, label=f'Теоретический Пуассон (λT={λT:.2f})')
        ]
        
        # Добавляем легенду справа
        self.legend_ax.legend(handles=legend_elements, loc='center', fontsize=11,
                             framealpha=0.9, facecolor='#1a1f3a', edgecolor='#00d4ff')
        
        # Аннотация с метриками
        mean_val = np.mean(data)
        var_val = np.var(data)
        textstr = f'Среднее: {mean_val:.3f}\nДисперсия: {var_val:.3f}\nОтношение: {var_val/mean_val:.3f}'
        props = dict(boxstyle='round', facecolor='#1a1f3a', alpha=0.8, edgecolor='#00d4ff', linewidth=1.5)
        self.ax.text(0.95, 0.95, textstr, transform=self.ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=props, color='#c0d4f0')
        
        self.draw()


class PoissonSimulator:
    def simulate_single_run(self, lambda_intensity: float, time_interval: float) -> int:
        """
        Симуляция одного эксперимента
        lambda_intensity: λ - интенсивность потока
        time_interval: T - время наблюдения
        """
        t = 0.0
        count = 0
        
        while True:
            dt = np.random.exponential(1 / lambda_intensity)
            t += dt
            if t > time_interval:
                break
            count += 1
            
        return count
    
    def simulate_many_runs(self, lambda_intensity: float, time_interval: float, 
                          n_runs: int, progress_callback=None) -> np.ndarray:
        results = np.zeros(n_runs, dtype=int)
        
        for i in range(n_runs):
            results[i] = self.simulate_single_run(lambda_intensity, time_interval)
            
            if progress_callback and (i + 1) % max(1, n_runs // 100) == 0:
                progress_callback((i + 1) / n_runs * 100)
                
        return results


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симулятор пуассоновского потока | Серверная аналитика")
        self.setGeometry(100, 100, 1400, 850)
        
        # Применяем стиль
        self.setStyleSheet(DARK_STYLE)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Левая панель (управление)
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Правая панель (визуализация)
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, stretch=2)
        
        # Инициализация симулятора
        self.simulator = PoissonSimulator()
        self.current_results = None
        
        # Установка значений по умолчанию
        self.update_lambdaT_display()
        
    def create_control_panel(self):
        """Создание панели управления"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Заголовок
        header = QLabel("⚙️ КОНФИГУРАЦИЯ")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff; padding: 10px;")
        layout.addWidget(header)
        
        # Разделитель
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2a3a5a; max-height: 2px;")
        layout.addWidget(sep)
        
        # Параметры потока
        flow_group = QGroupBox("Параметры потока")
        flow_layout = QVBoxLayout()
        
        # Lambda
        lambda_layout = QVBoxLayout()
        lambda_layout.addWidget(QLabel("Интенсивность λ (запросов/сек):"))
        self.lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.lambda_slider.setRange(1, 200)  # 0.1 to 20.0
        self.lambda_slider.setValue(50)  # 5.0
        self.lambda_slider.valueChanged.connect(self.on_lambda_changed)
        lambda_layout.addWidget(self.lambda_slider)
        
        lambda_value_layout = QHBoxLayout()
        self.lambda_value_label = QLabel("5.000")
        self.lambda_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lambda_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        self.lambda_spinbox = QDoubleSpinBox()
        self.lambda_spinbox.setRange(0.1, 20.0)
        self.lambda_spinbox.setValue(5.0)
        self.lambda_spinbox.setSingleStep(0.5)
        self.lambda_spinbox.valueChanged.connect(self.on_lambda_spinbox_changed)
        lambda_value_layout.addWidget(QLabel("Значение:"))
        lambda_value_layout.addWidget(self.lambda_spinbox)
        lambda_value_layout.addWidget(self.lambda_value_label)
        lambda_layout.addLayout(lambda_value_layout)
        
        flow_layout.addLayout(lambda_layout)
        
        # T
        T_layout = QVBoxLayout()
        T_layout.addWidget(QLabel("Интервал наблюдения T (сек):"))
        self.T_slider = QSlider(Qt.Orientation.Horizontal)
        self.T_slider.setRange(1, 100)  # 0.1 to 10.0
        self.T_slider.setValue(20)  # 2.0
        self.T_slider.valueChanged.connect(self.on_T_changed)
        T_layout.addWidget(self.T_slider)
        
        T_value_layout = QHBoxLayout()
        self.T_value_label = QLabel("2.000")
        self.T_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.T_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        self.T_spinbox = QDoubleSpinBox()
        self.T_spinbox.setRange(0.1, 10.0)
        self.T_spinbox.setValue(2.0)
        self.T_spinbox.setSingleStep(0.5)
        self.T_spinbox.valueChanged.connect(self.on_T_spinbox_changed)
        T_value_layout.addWidget(QLabel("Значение:"))
        T_value_layout.addWidget(self.T_spinbox)
        T_value_layout.addWidget(self.T_value_label)
        T_layout.addLayout(T_value_layout)
        
        flow_layout.addLayout(T_layout)
        
        # λT display
        self.lambdaT_display = QLabel()
        self.lambdaT_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lambdaT_display.setStyleSheet("""
            background-color: #1a2a4f;
            border-radius: 8px;
            padding: 10px;
            font-size: 13px;
            color: #00d4ff;
        """)
        flow_layout.addWidget(self.lambdaT_display)
        
        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)
        
        # Параметры эксперимента
        exp_group = QGroupBox("Параметры эксперимента")
        exp_layout = QVBoxLayout()
        
        exp_layout.addWidget(QLabel("Количество экспериментов N:"))
        self.n_runs_spinbox = QSpinBox()
        self.n_runs_spinbox.setRange(100, 100000)
        self.n_runs_spinbox.setValue(10000)
        self.n_runs_spinbox.setSingleStep(1000)
        exp_layout.addWidget(self.n_runs_spinbox)
        
        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Кнопка запуска
        self.run_button = QPushButton("🚀 ЗАПУСТИТЬ СИМУЛЯЦИЮ")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setMinimumHeight(45)
        layout.addWidget(self.run_button)
        
        # Статистика
        stats_group = QGroupBox("Статистика")
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
        """Создание панели визуализации"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Заголовок
        header = QLabel("📊 ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff; padding: 10px;")
        layout.addWidget(header)
        
        # Разделитель
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2a3a5a; max-height: 2px;")
        layout.addWidget(sep)
        
        # График
        self.plot_canvas = ModernPlotCanvas(width=10, height=7)
        layout.addWidget(self.plot_canvas)
        
        # Информационная панель
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #1a2a4f;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        
        self.info_text = QLabel("Готов к симуляции. Настройте параметры и нажмите 'Запустить симуляцию'")
        self.info_text.setStyleSheet("color: #c0d4f0; font-size: 11px;")
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_frame)
        
        return panel
    
    def on_lambda_changed(self, value):
        """Изменение слайдера λ"""
        lambda_val = value / 10.0
        self.lambda_spinbox.blockSignals(True)
        self.lambda_spinbox.setValue(lambda_val)
        self.lambda_spinbox.blockSignals(False)
        self.lambda_value_label.setText(f"{lambda_val:.3f}")
        self.update_lambdaT_display()
    
    def on_lambda_spinbox_changed(self, value):
        """Изменение спинбокса λ"""
        self.lambda_slider.blockSignals(True)
        self.lambda_slider.setValue(int(value * 10))
        self.lambda_slider.blockSignals(False)
        self.lambda_value_label.setText(f"{value:.3f}")
        self.update_lambdaT_display()
    
    def on_T_changed(self, value):
        """Изменение слайдера T"""
        T_val = value / 10.0
        self.T_spinbox.blockSignals(True)
        self.T_spinbox.setValue(T_val)
        self.T_spinbox.blockSignals(False)
        self.T_value_label.setText(f"{T_val:.3f}")
        self.update_lambdaT_display()
    
    def on_T_spinbox_changed(self, value):
        """Изменение спинбокса T"""
        self.T_slider.blockSignals(True)
        self.T_slider.setValue(int(value * 10))
        self.T_slider.blockSignals(False)
        self.T_value_label.setText(f"{value:.3f}")
        self.update_lambdaT_display()
    
    def update_lambdaT_display(self):
        """Обновление отображения λT"""
        lambda_val = self.lambda_spinbox.value()
        T_val = self.T_spinbox.value()
        lambdaT = lambda_val * T_val
        self.lambdaT_display.setText(f"Ожидаемое число запросов λT = {lambdaT:.4f}")
    
    def run_simulation(self):
        """Запуск симуляции"""
        # Блокируем кнопку
        self.run_button.setEnabled(False)
        self.run_button.setText("⏳ СИМУЛЯЦИЯ ВЫПОЛНЯЕТСЯ...")
        
        # Показываем прогресс
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Получаем параметры
        lambda_val = self.lambda_spinbox.value()
        T_val = self.T_spinbox.value()
        n_runs = self.n_runs_spinbox.value()
        lambdaT = lambda_val * T_val
        
        # Функция обновления прогресса
        def update_progress(progress):
            self.progress_bar.setValue(int(progress))
            QApplication.processEvents()
        
        # Запускаем симуляцию
        def run_in_background():
            results = self.simulator.simulate_many_runs(lambda_val, T_val, n_runs, update_progress)
            self.current_results = results
            self.display_results(results, lambda_val, T_val, lambdaT, n_runs)
            
            # Восстанавливаем UI
            self.run_button.setEnabled(True)
            self.run_button.setText("ЗАПУСТИТЬ СИМУЛЯЦИЮ")
            self.progress_bar.setVisible(False)
        
        QTimer.singleShot(100, run_in_background)
    
    def display_results(self, results: np.ndarray, lambda_val: float, T_val: float, lambdaT: float, n_runs: int):
        """Отображение результатов"""
        # Вычисляем статистику
        mean_val = np.mean(results)
        var_val = np.var(results)
        std_val = np.std(results)
        ratio = var_val / mean_val if mean_val > 0 else 0
        
        # Обновляем график
        self.plot_canvas.plot_results(results, lambdaT, lambda_val, T_val)
        
        # Обновляем статистику
        stats_text = f"""
        <b>ЭМПИРИЧЕСКИЕ ХАРАКТЕРИСТИКИ</b><br>
        <br>
        Среднее значение: {mean_val:.4f}<br>
        Дисперсия: {var_val:.4f}<br>
        Стандартное отклонение: {std_val:.4f}<br>
        Отношение дисперсии к среднему: {ratio:.4f}<br>
        <br>
        <b>ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ</b><br>
        <br>
        Ожидаемое λT: {lambdaT:.4f}<br>
        <br>
        """
        
        # Добавляем вывод
        if abs(ratio - 1) < 0.05:
            stats_text += '<span style="color:#00ff88;">✓ ПУАССОНОВСКИЙ ПОТОК ПОДТВЕРЖДЁН</span><br>'
            stats_text += '<span style="font-size: 10px;">Дисперсия ≈ среднему, распределение соответствует теории</span>'
        else:
            stats_text += '<span style="color:#ffaa00;">⚠ ОТКЛОНЕНИЕ ОТ ПУАССОНОВСКОГО ПОТОКА</span><br>'
            stats_text += f'<span style="font-size: 10px;">Отношение дисперсии к среднему = {ratio:.3f} (теоретически = 1)</span>'
        
        self.stats_display.setText(stats_text)
        
        # Обновляем информационную строку
        self.info_text.setText(f"Симуляция завершена | {n_runs} экспериментов | "
                              f"Среднее: {mean_val:.3f} | Дисперсия: {var_val:.3f} | "
                              f"λT = {lambdaT:.3f}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()