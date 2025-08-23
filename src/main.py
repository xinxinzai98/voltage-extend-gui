import sys
import traceback
from PyQt5 import QtWidgets

try:
    from gui.main_window import MainWindow
except Exception as e:
    print("导入 MainWindow 失败:", e)
    traceback.print_exc()
    MainWindow = None

def main():
    app = QtWidgets.QApplication(sys.argv)
    if MainWindow is None:
        QtWidgets.QMessageBox.critical(None, "启动失败", "无法导入 MainWindow，详情请查看终端输出。")
        sys.exit(1)
    try:
        w = MainWindow()
        w.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("运行 MainWindow 时出错:", e)
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(None, "运行错误", f"{e}\n详情请查看终端输出。")
        sys.exit(1)

if __name__ == "__main__":
    main()