# controls.py

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import Qt

class Controls(QWidget):
    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.on_slider_change)

        self.label = QLabel("Current Value: 50")

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_parameters)

        self.save_button = QPushButton("Save Parameters")
        self.save_button.clicked.connect(self.save_parameters)

        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def on_slider_change(self, value):
        self.label.setText(f"Current Value: {value}")
        self.update_callback(value)

    def reset_parameters(self):
        self.slider.setValue(50)
        self.label.setText("Current Value: 50")
        self.update_callback(50)

    def save_parameters(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'w') as f:
                f.write(f"Parameter Value: {self.slider.value()}\n")