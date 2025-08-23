from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QDoubleSpinBox, QSpinBox
import pandas as pd
import process as process_module
from gui.plot_widget import PlotWidget

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        import traceback, sys
        try:
            print("DEBUG: MainWindow.__init__ 开始构建 UI", file=sys.stdout)
            super().__init__()
            self.setWindowTitle("Voltage Extend GUI")
            self.resize(1200, 800)

            # 中央 widget
            central = QWidget()
            self.setCentralWidget(central)
            hbox = QHBoxLayout(central)

            # 左侧参数面板
            ctrl = QWidget()
            ctrl_layout = QVBoxLayout(ctrl)
            ctrl_layout.setContentsMargins(8,8,8,8)
            ctrl_layout.setSpacing(6)

            # 文件加载
            self.load_btn = QPushButton("Load Data")
            # clicked 会传入一个 bool 参数，使用 lambda 防止传参到 load_data
            self.load_btn.clicked.connect(lambda: self.load_data())
            ctrl_layout.addWidget(self.load_btn)

            # 参数控件简洁列表（label + spinbox）
            self.controls = {}
            def add_param(name, lbl, val, step=0.01, minv=0.0, maxv=10.0, decimals=3):
                l = QLabel(lbl)
                sb = QDoubleSpinBox()
                sb.setRange(minv, maxv)
                sb.setSingleStep(step)
                sb.setDecimals(decimals)
                sb.setValue(val)
                sb.valueChanged.connect(self.on_param_changed)
                ctrl_layout.addWidget(l)
                ctrl_layout.addWidget(sb)
                self.controls[name] = sb

            add_param('shrink_alpha', '收缩系数（越小越保守）', 0.35, step=0.05, minv=0.0, maxv=1.0, decimals=2)
            add_param('max_growth', '差值最大增长倍数', 1.05, step=0.01, minv=0.5, maxv=3.0, decimals=2)
            add_param('noise_scale', '段内噪声缩放', 0.4, step=0.05, minv=0.0, maxv=2.0, decimals=2)
            add_param('jitter_scale', '周期抖动缩放', 0.6, step=0.05, minv=0.0, maxv=2.0, decimals=2)
            add_param('phi', 'AR(1) 系数 φ', 0.8, step=0.05, minv=0.0, maxv=0.99, decimals=2)
            add_param('beta', '低频非线性混合比 β', 0.35, step=0.05, minv=0.0, maxv=1.0, decimals=2)
            add_param('desired_high_at_target', '目标高位（target处）', 1.9, step=0.01, minv=1.0, maxv=3.0, decimals=2)
            add_param('ramp_weight', '末端拉升权重', 0.9, step=0.05, minv=0.0, maxv=1.0, decimals=2)
            # 新增：gamma（diff 混合权重） 与 diff_end_multiplier（终点 diff 倍数）
            add_param('gamma', 'diff 历史回归混合权重 γ', 0.7, step=0.05, minv=0.0, maxv=1.0, decimals=2)
            add_param('diff_end_multiplier', '终点高低差倍数（终点 diff = 起始 diff * 倍数）', 1.0, step=0.05, minv=0.1, maxv=5.0, decimals=2)
            add_param('smooth_w', '残差平滑窗口', 3, step=1, minv=1, maxv=21, decimals=0)

            # 颜色/线型/显示 控件：颜色按钮 + 显示复选 + 线型下拉
            from PyQt5.QtWidgets import QComboBox, QCheckBox, QColorDialog
            self.color_selectors = {}
            def add_style_block(name, label, default_color):
                l = QLabel(label)
                row = QHBoxLayout()
                color_btn = QPushButton()
                color_btn.setFixedSize(24, 20)
                # 设置初始颜色样式表
                color_btn.setStyleSheet(f"background-color: {default_color}")
                vis_chk = QCheckBox("显示")
                vis_chk.setChecked(True)
                ls_box = QComboBox()
                ls_box.addItems(['-', '--', '-.', ':', 'None'])
                # 回调打开颜色选择器
                def pick_color():
                    c = QColorDialog.getColor()
                    if c.isValid():
                        color_btn.setStyleSheet(f"background-color: {c.name()}")
                        self.on_param_changed()
                color_btn.clicked.connect(pick_color)
                # 任何选项变化触发更新
                vis_chk.stateChanged.connect(self.on_param_changed)
                ls_box.currentIndexChanged.connect(self.on_param_changed)
                row.addWidget(color_btn)
                row.addWidget(vis_chk)
                row.addWidget(ls_box)
                ctrl_layout.addWidget(l)
                ctrl_layout.addLayout(row)
                self.color_selectors[name] = {'btn': color_btn, 'vis': vis_chk, 'ls': ls_box}

            add_style_block('original', '原始点样式', 'blue')
            add_style_block('interp',    '插值线样式', 'green')
            add_style_block('ext',       '外延线样式', 'red')

            # 预测高/低线样式（颜色/显示/线型）
            add_style_block('pred_high', '预测高位线', 'orange')
            add_style_block('pred_low',  '预测低位线', 'purple')

            # 线宽控制（原始/插值/外延/预测高/低）
            # 使用文件顶部已导入的 QDoubleSpinBox，移除此处重复导入以避免闭包错误
            lw_box = QHBoxLayout()
            lw_box.addWidget(QLabel("线宽 (原/插/外/高/低)"))
            self.linewidth_controls = {}
            for key, default in [('original',0.8), ('interp',1.0), ('ext',1.2), ('pred_high',1.0), ('pred_low',1.0)]:
                sb = QDoubleSpinBox()
                sb.setRange(0.1, 10.0)
                sb.setSingleStep(0.1)
                sb.setDecimals(2)
                sb.setValue(default)
                sb.valueChanged.connect(self.on_param_changed)
                lw_box.addWidget(sb)
                self.linewidth_controls[key] = sb
            ctrl_layout.addLayout(lw_box)

            # 预测线显示开关 & 导出/保存图像按钮
            self.pred_toggle_chk = QCheckBox("显示预测高/低值")
            self.pred_toggle_chk.setChecked(True)
            self.pred_toggle_chk.stateChanged.connect(self.on_param_changed)
            ctrl_layout.addWidget(self.pred_toggle_chk)

            # 导出与保存图像按钮
            btn_row = QHBoxLayout()
            self.export_btn = QPushButton("导出数据")
            self.save_img_btn = QPushButton("保存图像")
            btn_row.addWidget(self.export_btn)
            btn_row.addWidget(self.save_img_btn)
            ctrl_layout.addLayout(btn_row)
            self.export_btn.clicked.connect(self.on_export_clicked)
            self.save_img_btn.clicked.connect(self.on_save_image_clicked)

            # 缩放与视图控制按钮（放大镜为切换矩形缩放工具；缩小为缩小中心区域；右键/按钮重置视图）
            zoom_box = QHBoxLayout()
            self.zoom_tool_btn = QPushButton("🔍 放大框选")
            self.zoom_out_btn = QPushButton("缩小 (中心)")
            self.reset_view_btn = QPushButton("重置视图")
            zoom_box.addWidget(self.zoom_tool_btn)
            zoom_box.addWidget(self.zoom_out_btn)
            zoom_box.addWidget(self.reset_view_btn)
            ctrl_layout.addLayout(zoom_box)

            # 连接：放大框选切换、缩小（按比例）、重置（恢复）
            self.zoom_tool_btn.clicked.connect(lambda: self.plot.toggle_rect_zoom())
            self.zoom_out_btn.clicked.connect(lambda: self.plot.zoom(1.25))
            self.reset_view_btn.clicked.connect(lambda: self.plot.reset_view())

            # 为参数控件添加鼠标悬浮提示（示例，可按需扩展）
            tips = {
                'shrink_alpha': '收缩系数：越小越保守（拉近预测到最后观测）',
                'gamma': 'diff 与历史回归的混合权重：0 表示只用线性插值的 diff，1 表示只用历史回归 diff',
                'diff_end_multiplier': '外延终点相对于起始平均 diff 的倍数（>1 表示逐步放大振幅）',
                'max_growth': '差值最大增长倍数：限制高低差增长幅度',
                'noise_scale': '段内噪声缩放：增大段内噪声幅度',
                'jitter_scale': '周期抖动缩放：增大周期级均值抖动',
                'phi': 'AR(1) 系数 φ：越接近1越连续平滑',
                'beta': '低频非线性混合比 β：越大越偏离直线趋势',
                'desired_high_at_target': '目标高位：希望在 target_hour 附近的高位值',
                'ramp_weight': '末端拉升权重：控制 ramp 注入比例',
                'smooth_w': '残差平滑窗口：段内噪声短期平滑窗口长度'
            }
            for name, widget in self.controls.items():
                if name in tips:
                    widget.setToolTip(tips[name])

            # 状态栏显示当前文件路径
            # 把控制面板和绘图部件加入中央布局
            self.plot = PlotWidget()
            hbox.addWidget(ctrl, 0)
            hbox.addWidget(self.plot, 1)
            self.statusBar().showMessage("未加载任何文件")

            # 模型数据占位
            self.raw_df = None
            self.processed_df = None
            self.ext_df = None
            self.final_df = None

            # 加载示例路径（如果存在）
            default_path = "data-layang_clean.xlsx"
            if QtCore.QFile.exists(default_path):
                self.load_data(default_path)

            print("DEBUG: MainWindow UI 构建完成", file=sys.stdout)
        except Exception as e:
            print("ERROR: MainWindow 初始化失败:", e, file=sys.stderr)
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, "初始化错误", f"{e}\n请查看终端输出以获取堆栈信息")
            # 重新抛出以便 upstream 捕获并退出
            raise

    def load_data(self, path=None):
        # 防护：若被信号误传入 bool（例如 clicked 会传 bool），忽略它
        if isinstance(path, bool):
            path = None
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open data file", ".", "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)")
            if not path:
                return
        try:
            # 直接用 pandas 读取，兼容 xlsx / csv
            if path.lower().endswith('.csv'):
                df = pd.read_csv(path)
            else:
                # 读取 xlsx；需要 openpyxl 安装
                df = pd.read_excel(path, engine='openpyxl')
            # 如果文件带有表头或额外行，尝试容错处理列名
            if 'Time' not in df.columns or 'Voltage' not in df.columns:
                # 若文件类似于原始格式（第一行为标题），尝试重命名前两列
                df = df.iloc[:, :2]
                df.columns = ['Time', 'Voltage']
            # 确保时间为数值（小时）
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
            df = df.dropna(subset=['Time','Voltage']).reset_index(drop=True)

            # 后续使用已有 process 模块做异常检测与插值
            outlier_mask = process_module.detect_outliers(df, 'Voltage')
            processed = process_module.process_data(df, outlier_mask)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"读取/处理数据失败: {e}")
            return
        self.raw_df = df
        self.processed_df = processed
        self.current_path = path
        self.statusBar().showMessage(f"Loaded {path}")
        self.on_param_changed()

    def get_params(self):
        params = {}
        for k, v in self.controls.items():
            try:
                params[k] = int(v.value()) if k == 'smooth_w' else float(v.value())
            except Exception:
                # 兜底：若控件异常，使用当前控件文本或默认
                try:
                    params[k] = float(v.text())
                except Exception:
                    params[k] = 3 if k == 'smooth_w' else 0.0
        params['smooth_w'] = int(params.get('smooth_w', 3))
        # 防护 target_spin 可能未初始化
        target_spin = getattr(self, 'target_spin', None)
        params['target_hour'] = int(target_spin.value()) if target_spin is not None else 2000
        return params

    def _collect_styles(self):
        """收集颜色/可见/线型/线宽设置，返回字典传给 plot.update_plot"""
        styles = {}
        for k, v in self.color_selectors.items():
            # 颜色从按钮样式表读取（fallback）
            style = v['btn'].styleSheet()
            import re
            m = re.search(r'background-color:\s*(#[0-9A-Fa-f]{6})', style)
            col = m.group(1) if m else ('blue' if k=='original' else ('green' if k=='interp' else 'red'))
            visible = bool(v['vis'].isChecked())
            linestyle = v['ls'].currentText()
            lw = float(self.linewidth_controls.get(k, self.linewidth_controls.get('ext')).value()) if hasattr(self, 'linewidth_controls') else 1.0
            styles[k] = {'color': col, 'visible': visible, 'linestyle': linestyle, 'linewidth': lw}
        # 确保预测线也有线宽（若用户未创建对应控件）
        for pk in ('pred_high','pred_low'):
            if pk not in styles:
                styles[pk] = {'color': ('orange' if pk=='pred_high' else 'purple'), 'visible': True, 'linestyle': '--', 'linewidth': float(self.linewidth_controls.get(pk, 1.0).value()) if hasattr(self,'linewidth_controls') else 1.0}
        return styles

    def on_export_clicked(self):
        """导出 processed 和 final 数据到 xlsx 使用 process.save_results 若存在"""
        if not hasattr(self, 'final_df') or self.final_df is None:
            QtWidgets.QMessageBox.information(self, "提示", "尚未生成外延数据，无法导出")
            return
        try:
            if hasattr(process_module, 'save_results'):
                process_module.save_results(self.processed_df, self.final_df, self.current_path if hasattr(self, 'current_path') else "data")
                QtWidgets.QMessageBox.information(self, "完成", "数据已导出")
            else:
                base, _ = os.path.splitext(self.current_path or "data.xlsx")
                self.processed_df[['Time','Voltage','is_outlier','Voltage_interpolated']].to_excel(f"{base}_processed.xlsx", index=False)
                self.final_df[['Time','Voltage','is_extended']].to_excel(f"{base}_final.xlsx", index=False)
                QtWidgets.QMessageBox.information(self, "完成", f"已保存到 {base}_processed.xlsx / {base}_final.xlsx")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(e))

    def on_save_image_clicked(self):
        if not hasattr(self, 'plot') or self.plot is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", ".", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")
        if not path:
            return
        try:
            self.plot.save_figure(path)
            QtWidgets.QMessageBox.information(self, "完成", f"图像已保存到 {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "保存失败", str(e))

    def on_param_changed(self):
        if self.processed_df is None:
            return
        params = self.get_params()
        try:
            final_df, ext_df = process_module.extend_data(self.processed_df,
                                                          target_hour=params.pop('target_hour'),
                                                          ref_hours=300,
                                                          **params)
            self.final_df = final_df
            self.ext_df = ext_df

            # 收集颜色/样式/显示设置并传给绘图部件
            styles = self._collect_styles()
            # 将样式以 colors 参数传给 plot.update_plot（plot_widget 目前接受 colors 参数）
            self.plot.update_plot(self.processed_df, final_df, ext_df, colors=styles)
        except Exception as e:
            import traceback, sys
            tb = traceback.format_exc()
            print("on_param_changed 错误:", tb, file=sys.stderr)
            self.statusBar().showMessage(f"更新失败: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "更新失败", f"{e}\n\n详细堆栈已打印到终端：\n{tb}")

    def reset_defaults(self):
        defaults = {
            'shrink_alpha':0.35, 'max_growth':1.05, 'noise_scale':0.4,
            'jitter_scale':0.6, 'phi':0.8, 'beta':0.35, 'desired_high_at_target':1.9,
            'ramp_weight':0.9, 'smooth_w':3
        }
        for k,v in defaults.items():
            if k in self.controls:
                self.controls[k].setValue(v)
        self.target_spin.setValue(2000)
        self.on_param_changed()