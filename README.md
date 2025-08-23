# Voltage Extend GUI

This project provides a graphical user interface (GUI) for extending voltage data and visualizing the results in real-time. The application allows users to adjust parameters dynamically and see how these changes affect the data processing outcomes.

## Project Structure

- **src/**: Contains the main application code.
  - **main.py**: Entry point for the application.
  - **process.py**: Data processing functions for extending voltage data and handling outliers.
  - **config.py**: Configuration settings and constants.
  - **gui/**: Contains GUI-related files.
    - **app.py**: Initializes the main application window.
    - **main_window.py**: Defines the layout of the main window.
    - **controls.py**: Implements controls like sliders and buttons for parameter adjustment.
    - **plot_widget.py**: Integrates matplotlib for real-time plotting.
    - **ui_main.ui**: Optional Qt Designer UI file for visual layout.
  - **widgets/**: Custom widgets for the application.
    - **parameter_panel.py**: Displays and modifies adjustable coefficients.
  - **io/**: Handles data loading and saving operations.
    - **data_loader.py**: Reads input data files and saves processed results.
  
- **tests/**: Contains unit tests for the application.
  - **test_extend.py**: Tests for the extension functionality.

- **requirements.txt**: Lists the dependencies required for the project.

- **.gitignore**: Specifies files and directories to ignore in version control.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd voltage-extend-gui
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/main.py
   ```

## Usage

- Use the sliders in the GUI to adjust parameters for the voltage extension process.
- The results will be visualized in real-time, allowing for immediate feedback on the effects of parameter changes.
- Save the processed data using the provided controls.

# Voltage Extend GUI - 使用说明（中文）

本项目提供一个交互式桌面应用，用于对电压序列数据做“外延”预测并实时可视化。界面允许动态调整外延算法中的各类系数（如收缩系数、抖动系数、低频混合比等），修改后会立即触发处理与绘图更新，便于对算法效果进行交互式调试与比较。

主要功能
- 加载 Excel/CSV 原始数据（第一列为 Time（小时），第二列为 Voltage）
- 异常值检测与插值（参见 [`detect_outliers`](src/process.py) / [`process_data`](src/process.py)）
- 基于周期块的高低位建模与外延（核心为 [`extend_data`](src/process.py)）
- 参数面板实时调整参数，界面自动调用外延并更新绘图（参见 [`MainWindow`](src/gui/main_window.py)）
- 导出处理后的数据与外延结果，保存图像

快速上手
1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
2. 运行应用：
   ```
   python src/main.py
   ```
3. 在主界面点击 “Load Data” 选择数据文件。默认会尝试加载 `data-layang_clean.xlsx`（若存在）。
4. 在左侧面板调整参数（收缩系数、抖动缩放、平滑窗口等），每次修改会触发重新计算并更新右侧绘图。
5. 使用“导出数据”可以将处理结果保存为 Excel；“保存图像”可以导出当前图像。

主要文件说明
- src/process.py：数据处理与外延算法实现，核心函数为 [`extend_data`](src/process.py)。
- src/gui/main_window.py：主界面实现，参数控件与信号连接，以及调用 [`extend_data`](src/process.py) 更新图与数据。
- src/gui/plot_widget.py：Matplotlib 嵌入窗口，负责实时绘图与交互（缩放/平移/框选）。
- src/io/data_loader.py：数据读写辅助（兼容 xlsx/csv）。
- tests/test_extend.py：单元测试示例。

实时调参与调试建议
- 在界面调整参数后，关注右侧绘图的“预测高位 / 预测低位”曲线形态，判断收缩系数（shrink_alpha）与 max_growth 对差值振幅的影响。
- 若需要在代码层面验证或扩展逻辑，查看并修改 [`extend_data`](src/process.py) 的参数默认值或内部子函数（例如 `_fit_predict_cycles`、`_build_residual_pools` 等）。
- 绘图风格、颜色与线宽可在主窗口面板中直接调整，便于对比不同参数下的曲线视觉差异。

依赖与环境
请参见 `requirements.txt`（在本目录下）安装必要库。建议使用 Python 3.8+ 虚拟环境。

贡献
欢迎提交 issue 或 PR。若需添加新参数或改进可视化交互，请在 GUI 部分（`src/gui/`）实现并确保参数变化会调用 `on_param_changed` 来触发更新。

许可证
本项目默认 MIT（如需更改请在仓库根目录添加 LICENSE 文件）。