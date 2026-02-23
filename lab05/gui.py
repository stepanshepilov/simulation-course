import random
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QPoint, QEasingCurve
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen, QFont, QRadialGradient

YES_NO = ["Yes", "No"]

MAGIC_8_BALL = [
    "It is certain.",
    "It is decidedly so.",
    "Without a doubt.",
    "Yes — definitely.",
    "You may rely on it.",
    
    "As I see it, yes.",
    "Most likely.",
    "Outlook good.",
    "Yes.",
    "Signs point to yes.",
    
    "Reply hazy, try again.",
    "Ask again later.",
    "Better not tell you now.",
    "Cannot predict now.",
    "Concentrate and ask again.",
    
    "Don't count on it.",
    "My reply is no.",
    "My sources say no.",
    "Outlook not so good.",
    "Very doubtful.",
]

def rand_yesno():
    return random.choice(YES_NO)

def rand_magic():
    return random.choice(MAGIC_8_BALL)

class MagicBallWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setMaximumSize(300, 300)
        
        self.prediction = "Press\nthe button"
        
        self.shake_offset = QPoint(0, 0)
        self.shake_animation = QPropertyAnimation(self, b"pos")
        self.shake_animation.setDuration(300)
        self.shake_animation.setLoopCount(3)
        self.shake_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.reveal_prediction)
        
        self.is_shaking = False
        
        self.original_pos = None
        
    def shake(self, prediction_func):
        if not self.is_shaking:
            self.is_shaking = True
            self.original_pos = self.pos()
            
            self.next_prediction = prediction_func()
            
            self.shake_animation.setStartValue(self.original_pos)
            
            keyframes = []
            for i in range(6):
                if i % 2 == 0:
                    keyframes.append(self.original_pos + QPoint(10, 5))
                else:
                    keyframes.append(self.original_pos + QPoint(-10, -5))
            
            self.shake_animation.setKeyValueAt(0.2, keyframes[0])
            self.shake_animation.setKeyValueAt(0.4, keyframes[1])
            self.shake_animation.setKeyValueAt(0.6, keyframes[2])
            self.shake_animation.setKeyValueAt(0.8, keyframes[3])
            self.shake_animation.setEndValue(self.original_pos)
            
            self.shake_animation.finished.connect(self.shake_finished)
            self.shake_animation.start()
            
    def shake_finished(self):
        self.is_shaking = False
        self.timer.start(200)
        
    def reveal_prediction(self):
        self.prediction = self.next_prediction
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = self.rect().center()
        radius = min(self.width(), self.height()) // 2 - 20
        
        gradient = QRadialGradient(center.x() - 20, center.y() - 20, radius * 1.5)
        gradient.setColorAt(0, QColor(30, 30, 50))
        gradient.setColorAt(0.7, QColor(10, 10, 20))
        gradient.setColorAt(1, QColor(0, 0, 0))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(100, 100, 150), 2))
        painter.drawEllipse(center, radius, radius)
        
        text_gradient = QRadialGradient(center.x(), center.y() - 20, 80)
        text_gradient.setColorAt(0, QColor(240, 240, 255))
        text_gradient.setColorAt(1, QColor(200, 200, 220))
        
        painter.setBrush(QBrush(text_gradient))
        painter.setPen(QPen(QColor(0, 0, 50), 2))
        painter.drawEllipse(center.x() - 70, center.y() - 50, 140, 100)
        
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 50), 3))
        painter.drawArc(center.x() - radius + 20, center.y() - radius + 20, 
                       radius - 20, radius - 20, 30 * 16, 60 * 16)
        
        painter.setPen(QColor(0, 0, 80))
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        text = self.prediction
        lines = text.split('\n') if '\n' in text else [text[i:i+15] for i in range(0, len(text), 15)]
        
        y_offset = -20 - (len(lines) - 1) * 10
        for line in lines:
            text_rect = painter.boundingRect(
                center.x() - 65, center.y() + y_offset, 130, 30,
                Qt.AlignmentFlag.AlignCenter, line
            )
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, line)
            y_offset += 20

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 5 — Magic 8-Ball")
        self.setMinimumSize(400, 500)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                border: 2px solid #5a5a5a;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #7a7a7a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        self.ball_widget = MagicBallWidget()
        layout.addWidget(self.ball_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        btn_yes = QPushButton("Yes / No")
        btn_yes.clicked.connect(lambda: self.ball_widget.shake(rand_yesno))
        button_layout.addWidget(btn_yes)
        
        btn_magic = QPushButton("Magic 8-Ball")
        btn_magic.clicked.connect(lambda: self.ball_widget.shake(rand_magic))
        button_layout.addWidget(btn_magic)
        
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear_prediction)
        button_layout.addWidget(btn_clear)
        
        layout.addLayout(button_layout)
        
        subtitle = QLabel("Ask a question and shake the ball")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(subtitle)
        
    def clear_prediction(self):
        self.ball_widget.prediction = "Ask\na question"
        self.ball_widget.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
