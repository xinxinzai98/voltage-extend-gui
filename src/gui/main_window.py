import os
import sys
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QSpinBox, QLineEdit, QCheckBox
# 新增导入
from PyQt5.QtCore import Qt
# 引入 plot widget（文件已存在）
from gui.plot_widget import PlotWidget

# 尝试导入 process 模块（容错，兼容不同模块路径）
try:
    import process as process_module
except Exception:
    try:
        # 将 src 目录加入 sys.path 后重试
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if base not in sys.path:
            sys.path.insert(0, base)
        import process as process_module
    except Exception:
        process_module = None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voltage Extend GUI (简化版)")
        self.resize(1100, 700)

        # 状态占位
        self.processed_df = None
        self.final_df = None
        self.ext_df = None
        self.current_file_path = None

        # 占位容器，避免属性缺失导致的 AttributeError
        self.controls = {}
        self.color_selectors = {}
        self.linewidth_controls = {}

        # 中央布局：左侧控制、右侧绘图
        central = QWidget()
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # 左侧控制面板（精简）
        ctrl_container = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_container)

        self.load_btn = QPushButton("加载数据")
        self.load_btn.clicked.connect(self.on_load_clicked)
        ctrl_layout.addWidget(self.load_btn)

        th_box = QHBoxLayout()
        th_box.addWidget(QLabel("预测到 (小时)"))
        self.target_hour_spin = QSpinBox()
        self.target_hour_spin.setRange(1, 100000)
        self.target_hour_spin.setValue(2000)
        self.target_hour_spin.valueChanged.connect(self.on_param_changed)
        th_box.addWidget(self.target_hour_spin)
        ctrl_layout.addLayout(th_box)

        rs_box = QHBoxLayout()
        rs_box.addWidget(QLabel("参考区间 起始"))
        self.ref_start_edit = QLineEdit()
        self.ref_start_edit.setPlaceholderText("可选，起始 Time（例如 1600.0）")
        self.ref_start_edit.editingFinished.connect(self.on_param_changed)
        self.ref_start_edit.setToolTip("参考区间的起始时间（留空表示使用最早参考点）")
        rs_box.addWidget(self.ref_start_edit)
        ctrl_layout.addLayout(rs_box)

        re_box = QHBoxLayout()
        re_box.addWidget(QLabel("参考区间 结束"))
        self.ref_end_edit = QLineEdit()
        self.ref_end_edit.setPlaceholderText("可选，结束 Time（留空表示最后时间）")
        self.ref_end_edit.editingFinished.connect(self.on_param_changed)
        self.ref_end_edit.setToolTip("参考区间的结束 Time（留空表示使用最后时间）")
        re_box.addWidget(self.ref_end_edit)
        ctrl_layout.addLayout(re_box)

        # 采样区间（单独于参考区间）
        ss_box = QHBoxLayout()
        ss_box.addWidget(QLabel("采样区间 起始"))
        self.sample_start_edit = QLineEdit()
        self.sample_start_edit.setPlaceholderText("可选，采样起始 Time（留空表示从头）")
        self.sample_start_edit.setToolTip("采样区间起始 Time，用于收集样本段")
        ss_box.addWidget(self.sample_start_edit)
        ctrl_layout.addLayout(ss_box)

        se_box = QHBoxLayout()
        se_box.addWidget(QLabel("采样区间 结束"))
        self.sample_end_edit = QLineEdit()
        self.sample_end_edit.setPlaceholderText("可选，采样结束 Time（留空表示到尾）")
        self.sample_end_edit.setToolTip("采样区间结束 Time，用于收集样本段")
        se_box.addWidget(self.sample_end_edit)
        ctrl_layout.addLayout(se_box)

        # 单段测试控件：是否只使用单个采样段，以及索引选择
        single_box = QHBoxLayout()
        self.single_segment_checkbox = QCheckBox("仅使用单个采样段")
        self.single_segment_checkbox.setToolTip("勾选后会仅使用由索引选择器指定的采样段来合成（便于测试）")
        single_box.addWidget(self.single_segment_checkbox)
        single_box.addWidget(QLabel("索引"))
        self.sample_index_spin = QSpinBox()
        self.sample_index_spin.setRange(0, 0)
        self.sample_index_spin.setValue(0)
        self.sample_index_spin.setToolTip("要单独使用的采样段索引（从0开始），采样后会根据实际段数调整范围")
        single_box.addWidget(self.sample_index_spin)
        ctrl_layout.addLayout(single_box)

        # 把三个调参控件（MAD 限幅倍数、时间容差、绝对幅度因子）删除，
        # 仅保留“首周期严格截断”复选框，避免与 heavy 依赖产生的交互问题
        self.strict_trunc_checkbox = QCheckBox("首周期严格截断（不调整相位）")
        self.strict_trunc_checkbox.setChecked(True)
        self.strict_trunc_checkbox.setToolTip("勾选后首周期按原始已持续时长截断，避免相位重排导致后续周期错位")
        ctrl_layout.addWidget(self.strict_trunc_checkbox)

        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("导出数据")
        self.save_img_btn = QPushButton("保存图像")
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.save_img_btn)
        # 采样与合成外延按钮
        self.collect_samples_btn = QPushButton("采样并保存段")
        self.collect_samples_btn.setToolTip("按当前参考起止（ref_start/ref_end）在原始数据中按周期切片并收集样本段，用于后续随机合成外延")
        btn_row.addWidget(self.collect_samples_btn)

        self.synthesize_btn = QPushButton("生成外延")
        self.synthesize_btn.setToolTip("使用已采集的样本段随机合成逐样本外延（会使用当前预测的 pred_high/pred_low 作为目标）")
        btn_row.addWidget(self.synthesize_btn)

        # 在已有按钮行加入“生成预测线”按钮（与其它按钮同一行 btn_row 中）
        self.predict_btn = QPushButton("生成预测线")
        self.predict_btn.setToolTip("根据当前参考区间计算 pred_high / pred_low 并绘制预测线（手动触发）")
        btn_row.addWidget(self.predict_btn)
        self.predict_btn.clicked.connect(self.on_generate_predictions_clicked)

        ctrl_layout.addLayout(btn_row)
        self.export_btn.clicked.connect(self.on_export_clicked)
        self.save_img_btn.clicked.connect(self.on_save_image_clicked)
        self.collect_samples_btn.clicked.connect(self.on_collect_samples_clicked)
        self.synthesize_btn.clicked.connect(self.on_synthesize_clicked)

        ctrl_layout.addStretch(1)
        main_layout.addWidget(ctrl_container, 0)

        # 右侧绘图
        self.plot = PlotWidget()
        main_layout.addWidget(self.plot, 1)

    # ---------- 文件/数据操作 ----------
    def on_load_clicked(self):
        try:
            self.load_data()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(e))

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", ".", "Excel/CSV Files (*.xlsx *.csv);;All Files (*)")
        if not path:
            return
        self.current_file_path = path
        if process_module is None:
            raise RuntimeError("无法导入 process 模块，请确保 src/process.py 可被导入")

        # 进度对话框（可取消）
        dlg = QtWidgets.QProgressDialog("正在加载数据...", "取消", 0, 100, self)
        dlg.setWindowTitle("加载中")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(200)
        dlg.setValue(0)
        try:
            # 步骤 1: 读取文件
            dlg.setLabelText("读取文件...")
            QtWidgets.QApplication.processEvents()
            df = process_module.read_data(path)
            dlg.setValue(20)
            if dlg.wasCanceled():
                return

            if df is None or len(df) == 0:
                raise RuntimeError("读取到空数据，请检查文件")

            # 步骤 2: 清洗（若可用）
            dlg.setLabelText("清洗原始数据...")
            QtWidgets.QApplication.processEvents()
            try:
                if hasattr(process_module, 'clean_raw_data'):
                    df_clean = process_module.clean_raw_data(df, col='Voltage')
                else:
                    df_clean = df
            except Exception:
                df_clean = df
            dlg.setValue(60)
            if dlg.wasCanceled():
                return

            # 步骤 3: 处理（不做额外插值）
            dlg.setLabelText("处理数据...")
            QtWidgets.QApplication.processEvents()
            try:
                self.processed_df = process_module.process_data(df_clean)
            except Exception as e:
                # 回退为原始已清洗数据（保证后续不崩溃）
                self.processed_df = df_clean
            dlg.setValue(85)
            if dlg.wasCanceled():
                return

            # 步骤 4: 更新界面（计算预测/绘图等）
            dlg.setLabelText("更新界面...")
            QtWidgets.QApplication.processEvents()
            try:
                self.on_param_changed()
            except Exception:
                pass
            dlg.setValue(100)
        finally:
            dlg.close()

    def on_export_clicked(self):
        if self.processed_df is None:
            QtWidgets.QMessageBox.information(self, "导出", "无数据可导出")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出数据", ".", "Excel Files (*.xlsx)")
        if not path:
            return
        base = path[:-5] if path.lower().endswith('.xlsx') else path
        try:
            self.processed_df.to_excel(f"{base}_processed.xlsx", index=False)
            if self.final_df is not None:
                self.final_df.to_excel(f"{base}_final.xlsx", index=False)
            QtWidgets.QMessageBox.information(self, "导出", "导出完成")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(e))

    def on_save_image_clicked(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", ".", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if not path:
            return
        try:
            if hasattr(self.plot, 'save_figure'):
                self.plot.save_figure(path)
            else:
                pix = self.plot.grab()
                pix.save(path)
            QtWidgets.QMessageBox.information(self, "保存图像", "已保存")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "保存失败", str(e))

    # ---------- 参数 / 绘图更新 ----------
    def get_params(self):
        params = {}
        try:
            params['target_hour'] = int(self.target_hour_spin.value())
        except Exception:
            params['target_hour'] = 2000
        try:
            txt = self.ref_start_edit.text().strip()
            params['ref_start'] = float(txt) if txt != '' else None
        except Exception:
            params['ref_start'] = None
        try:
            txt = self.ref_end_edit.text().strip()
            params['ref_end'] = float(txt) if txt != '' else None
        except Exception:
            params['ref_end'] = None
        return params

    def on_param_changed(self):
        """
        参数变更仅更新本地显示（不自动计算预测线）。
        用户需点击“生成预测线”按钮来触发 extend_data 计算并绘制。
        """
        if getattr(self, 'processed_df', None) is None:
            return
        params = self.get_params()

        # 不再自动生成预测线：清除/保留 final_df/ext_df 状态以避免自动覆盖
        # 将 final_df 设为 processed_df（仅用于绘图显示原始数据）
        self.final_df = self.processed_df
        self.ext_df = None

        try:
            self.plot.update_plot(self.processed_df, self.final_df, None, colors={}, ref_start=params.get('ref_start'), ref_end=params.get('ref_end'))
        except Exception:
            pass

    def on_collect_samples_clicked(self):
        """回调：使用 process.collect_sample_segments 采集样本段并保存在 self.sample_segments"""
        try:
            if getattr(self, 'processed_df', None) is None:
                QtWidgets.QMessageBox.information(self, "提示", "请先加载数据")
                return
            # 优先使用采样区间控件（sample_start/sample_end），若为空则回退到参考区间 ref_start/ref_end，再回退到全部
            try:
                txt = self.sample_start_edit.text().strip()
                sample_start = float(txt) if txt != '' else None
            except Exception:
                sample_start = None
            try:
                txt = self.sample_end_edit.text().strip()
                sample_end = float(txt) if txt != '' else None
            except Exception:
                sample_end = None
            if sample_start is None and sample_end is None:
                # 回退到参考区间（若用户填写了）
                try:
                    txt = self.ref_start_edit.text().strip()
                    sample_start = float(txt) if txt != '' else None
                except Exception:
                    sample_start = None
                try:
                    txt = self.ref_end_edit.text().strip()
                    sample_end = float(txt) if txt != '' else None
                except Exception:
                    sample_end = None
            # 不再从 GUI 读取 cap_k / tol_frac，使用 process 模块的默认行为
            segs = process_module.collect_sample_segments(self.processed_df, sample_start=sample_start, sample_end=sample_end)
            self.sample_segments = segs
            # 更新索引选择器范围
            max_idx = max(0, len(segs) - 1)
            try:
                self.sample_index_spin.setRange(0, max_idx)
                self.sample_index_spin.setValue(0)
            except Exception:
                pass
            QtWidgets.QMessageBox.information(self, "采样完成", f"已采集 {len(segs)} 个样本段（保存在 sample_segments）")
        except Exception as e:
             import traceback, sys
             print("on_collect_samples_clicked 错误:", traceback.format_exc(), file=sys.stderr)
             QtWidgets.QMessageBox.critical(self, "采样失败", str(e))

    def on_synthesize_clicked(self):
        """回调：用已采样段合成逐样本外延并更新绘图（支持首周期相位对齐与单周期按点相位着色）"""
        try:
            if getattr(self, 'processed_df', None) is None:
                QtWidgets.QMessageBox.information(self, "提示", "请先加载数据")
                return
            if not hasattr(self, 'sample_segments') or not self.sample_segments:
                QtWidgets.QMessageBox.information(self, "提示", "请先点击“采样并保存段”以收集样本段")
                return
            if getattr(self, 'ext_df', None) is None or len(self.ext_df) == 0:
                QtWidgets.QMessageBox.information(self, "提示", "请先生成周期级预测（extend_data），以得到 pred_high/pred_low 目标")
                return

            # 选择样本段集合（索引用于选择要测试的段）
            if getattr(self, 'single_segment_checkbox', None) and self.single_segment_checkbox.isChecked():
                idx = int(self.sample_index_spin.value())
                idx = max(0, min(idx, len(self.sample_segments) - 1))
                sample_segments_use = [self.sample_segments[idx]]
            else:
                sample_segments_use = self.sample_segments

            # 判断原始最后一个点属于 high 还是 low：使用全局 KMeans 与 processed_df 的 Voltage_interpolated/Voltage
            last_val = float(self.processed_df['Voltage_interpolated'].iat[-1] if 'Voltage_interpolated' in self.processed_df.columns else self.processed_df['Voltage'].iat[-1])
            vals_all = (self.processed_df['Voltage_interpolated'].values if 'Voltage_interpolated' in self.processed_df.columns else self.processed_df['Voltage'].values)
            start_phase = None
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=2, random_state=0).fit(vals_all.reshape(-1,1))
                centers = km.cluster_centers_.ravel()
                high_label = int(np.argmax(centers))
                last_lab = km.predict([[last_val]])[0]
                start_phase = 'high' if int(last_lab) == high_label else 'low'
            except Exception:
                # 退化：用全局中位数判断
                start_phase = 'high' if last_val >= float(np.median(vals_all)) else 'low'

            # 构建 ext_cycles_df 并把周期时间移到 last_time 后
            ext_cycles_df = self.ext_df.copy()
            last_time = float(self.processed_df['Time'].iloc[-1])
            cycle_hours = 8.0

            if getattr(self, 'single_segment_checkbox', None) and self.single_segment_checkbox.isChecked():
                ext_cycles_df = ext_cycles_df.iloc[:1].copy()
                new_mid_times = np.array([last_time + cycle_hours])
            else:
                n_cycles = len(ext_cycles_df)
                new_mid_times = last_time + cycle_hours * np.arange(1, n_cycles + 1)
            ext_cycles_df['Time'] = new_mid_times

            # 调用合成（不再从 GUI 读取 cap_k/abs_cap_mult，而使用 process 模块默认值；仍传 strict_truncate）
            strict_truncate = bool(self.strict_trunc_checkbox.isChecked())
            ext_samples = process_module.synthesize_extension_from_samples(
                self.processed_df,
                ext_cycles_df,
                sample_segments_use,
                last_time=last_time,
                interval=None,
                cycle_hours=cycle_hours,
                rng_seed=None,
                stretch_limit=0.25,
                residual_scale=1.0,
                start_phase=start_phase,
                strict_truncate=strict_truncate
            )
            self.ext_df_samples = ext_samples

            # 单周期测试：按目标 mid（pred_high/pred_low）把外延点分为 high/low，用于按点着色显示
            high_low_marks = None
            if getattr(self, 'single_segment_checkbox', None) and self.single_segment_checkbox.isChecked():
                if len(ext_samples) > 0:
                    # 获取当前目标的 mid（ext_cycles_df 第一行）
                    th = float(ext_cycles_df['pred_high'].iloc[0]) if 'pred_high' in ext_cycles_df.columns else None
                    tl = float(ext_cycles_df['pred_low'].iloc[0])  if 'pred_low'  in ext_cycles_df.columns else None
                    if th is not None and tl is not None:
                        mid = 0.5*(th + tl)
                        times = ext_samples['Time'].values
                        vals = ext_samples['Voltage'].values
                        high_mask = vals >= mid
                        high_times = times[high_mask]; high_vals = vals[high_mask]
                        low_times  = times[~high_mask]; low_vals = vals[~high_mask]
                        high_low_marks = {'high_pts': (high_times, high_vals), 'low_pts': (low_times, low_vals)}
            # 更新 final_df: 合并并绘图（传入高低点集合用于着色）
            try:
                self.final_df = pd.concat([self.processed_df, ext_samples], ignore_index=True).sort_values('Time').reset_index(drop=True)
            except Exception:
                self.final_df = self.processed_df

            try:
                self.plot.update_plot(self.processed_df, self.final_df, ext_samples, colors={}, ref_start=self.get_params().get('ref_start'), ref_end=self.get_params().get('ref_end'), high_low_marks=high_low_marks)
            except Exception:
                pass

            QtWidgets.QMessageBox.information(self, "生成完成", f"生成外延样本点：{len(ext_samples)} 条")
        except Exception as e:
            import traceback, sys
            print("on_synthesize_clicked 错误:", traceback.format_exc(), file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "生成失败", str(e))

    def on_generate_predictions_clicked(self):
        """手动触发：根据当前参考区间计算预测（pred_high/pred_low）并更新绘图。"""
        try:
            if getattr(self, 'processed_df', None) is None:
                QtWidgets.QMessageBox.information(self, "提示", "请先加载数据")
                return
            if process_module is None:
                QtWidgets.QMessageBox.critical(self, "错误", "无法导入 process 模块")
                return

            params = self.get_params()
            # 调用 process.extend_data 生成预测（可能较慢，必要时考虑放到后台线程）
            try:
                final_df, ext_df = process_module.extend_data(self.processed_df, **params)
            except Exception as e:
                raise RuntimeError(f"生成预测失败: {e}")

            self.final_df = final_df
            self.ext_df = ext_df

            # 绘图并提示
            try:
                self.plot.update_plot(self.processed_df, self.final_df, self.ext_df, colors={}, ref_start=params.get('ref_start'), ref_end=params.get('ref_end'))
            except Exception:
                pass

            QtWidgets.QMessageBox.information(self, "完成", f"已生成 {0 if ext_df is None else len(ext_df)} 条周期预测")
        except Exception as e:
            import traceback, sys
            print("on_generate_predictions_clicked 错误:", traceback.format_exc(), file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "失败", str(e))

# end of file
