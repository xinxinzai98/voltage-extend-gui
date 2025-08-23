from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from controls import Controls
from plot_widget import PlotWidget
from widgets.parameter_panel import ParameterPanel
import sys

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voltage Extension GUI")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.parameter_panel = ParameterPanel()
        self.layout.addWidget(self.parameter_panel)

        self.plot_widget = PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.controls = Controls()
        self.layout.addWidget(self.controls)

        self.controls.start_button.clicked.connect(self.start_processing)
        self.controls.reset_button.clicked.connect(self.reset_parameters)

    def start_processing(self):
        # Logic to start processing and update the plot
        pass

    def reset_parameters(self):
        self.parameter_panel.reset_parameters()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())