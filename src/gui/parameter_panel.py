from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt

class ParameterPanel(QWidget):
    def __init__(self, parent=None):
        super(ParameterPanel, self).__init__(parent)
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Adjustable Coefficients")
        layout.addWidget(self.label)

        # Example adjustable parameters
        self.beta_slider = self.create_slider("Beta Coefficient", 0.0, 1.0, 0.35)
        self.gamma_slider = self.create_slider("Gamma Coefficient", 0.0, 1.0, 0.7)

        # Reset and Save buttons
        button_layout = QHBoxLayout()
        reset_button = QPushButton("Reset")
        save_button = QPushButton("Save")
        
        button_layout.addWidget(reset_button)
        button_layout.addWidget(save_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connect signals
        reset_button.clicked.connect(self.reset_parameters)
        save_button.clicked.connect(self.save_parameters)

    def create_slider(self, label_text, min_value, max_value, default_value):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value * 100)
        slider.setMaximum(max_value * 100)
        slider.setValue(default_value * 100)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)

        layout.addWidget(label)
        layout.addWidget(slider)
        self.layout().addLayout(layout)

        return slider

    def reset_parameters(self):
        self.beta_slider.setValue(35)  # Reset to default 0.35
        self.gamma_slider.setValue(70)  # Reset to default 0.7

    def save_parameters(self):
        beta_value = self.beta_slider.value() / 100.0
        gamma_value = self.gamma_slider.value() / 100.0
        print(f"Saved Parameters: Beta = {beta_value}, Gamma = {gamma_value}")
        # Here you would typically save these values to a config or pass them to the processing logic.