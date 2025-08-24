from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib
import matplotlib.font_manager as fm
import numpy as np

# 尝试设置一个支持中文的字体列表（macOS 常见字体优先）
_CHINESE_CANDIDATES = [
    "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "Microsoft YaHei", "SimHei"
]
_available = []
for name in _CHINESE_CANDIDATES:
    try:
        prop = fm.FontProperties(family=name)
        f = fm.findfont(prop, fallback_to_default=False)
        if f:
            _available.append(name)
    except Exception:
        continue

if _available:
    matplotlib.rcParams['font.sans-serif'] = _available + matplotlib.rcParams.get('font.sans-serif', [])
    matplotlib.rcParams['font.family'] = 'sans-serif'
# 确保负号正常显示
matplotlib.rcParams['axes.unicode_minus'] = False


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # 交互状态
        self._is_panning = False
        self._pan_start = None
        self._orig_xlim = None
        self._orig_ylim = None
        self._initial_view = None  # 保存初始视图以右键重置

        # 连接鼠标事件
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_button_press(self, event):
        # 中键（2）按下开启平移
        try:
            if event.button == 2:  # middle
                self._is_panning = True
                self._pan_start = (event.x, event.y)
                self._orig_xlim = self.ax.get_xlim()
                self._orig_ylim = self.ax.get_ylim()
            elif event.button == 3:  # right -> 重置视图到初始
                if self._initial_view is not None:
                    self.ax.set_xlim(self._initial_view[0])
                    self.ax.set_ylim(self._initial_view[1])
                    self.canvas.draw_idle()
        except Exception:
            pass

    def _on_button_release(self, event):
        if event.button == 2:
            self._is_panning = False
            self._pan_start = None

    def _on_motion(self, event):
        if not self._is_panning or event.x is None or event.y is None:
            return
        try:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            # 像素到数据坐标映射
            x0, x1 = self._orig_xlim
            y0, y1 = self._orig_ylim
            w = self.canvas.width()
            h = self.canvas.height()
            if w == 0 or h == 0:
                return
            dx_data = -dx * (x1 - x0) / w
            dy_data = dy * (y1 - y0) / h
            self.ax.set_xlim(x0 + dx_data, x1 + dx_data)
            self.ax.set_ylim(y0 + dy_data, y1 + dy_data)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_scroll(self, event):
        """以鼠标位置为中心缩放，滚轮向上放大（scale<1），向下缩小（scale>1）"""
        try:
            base_scale = 1.1
            if event.button == 'up':
                scale = 1 / base_scale
            else:
                scale = base_scale
            xdata = event.xdata
            ydata = event.ydata
            if xdata is None or ydata is None:
                return
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale
            relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
            rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
            new_xmin = xdata - relx * new_width
            new_xmax = xdata + (1 - relx) * new_width
            new_ymin = ydata - rely * new_height
            new_ymax = ydata + (1 - rely) * new_height
            self.ax.set_xlim(new_xmin, new_xmax)
            self.ax.set_ylim(new_ymin, new_ymax)
            self.canvas.draw_idle()
        except Exception:
            pass

    def update_plot(self, processed_df, final_df, ext_df=None, colors=None, ref_start=None, ref_end=None, high_low_marks=None):
        """
        绘图更新：兼容周期预测(pred_high/pred_low)与逐样本外延(Voltage)。
        high_low_marks: None or {'high_pts':(times,vals), 'low_pts':(times,vals)}
        """
        import numpy as _np
        self.ax.clear()

        # 原始/插值数据
        if processed_df is not None and len(processed_df) > 0:
            t = processed_df['Time'].values
            v = processed_df['Voltage'].values if 'Voltage' in processed_df.columns else None
            vi = processed_df['Voltage_interpolated'].values if 'Voltage_interpolated' in processed_df.columns else None
            if v is not None:
                style = colors.get('original', {}) if colors else {}
                self.ax.plot(t, v, '.', color=style.get('color', 'blue'), label='原始', alpha=0.6, markersize=3)
            if vi is not None:
                style = colors.get('interp', {}) if colors else {}
                self.ax.plot(t, vi, '-', color=style.get('color', 'green'), label='插值', linewidth=style.get('linewidth', 1.0), alpha=0.9)

        # 外延 / 预测线：兼容多种格式
        if ext_df is not None and len(ext_df) > 0:
            if {'pred_high', 'pred_low'}.issubset(ext_df.columns):
                th = ext_df['Time'].values
                ph = ext_df['pred_high'].values
                pl = ext_df['pred_low'].values
                sh = (colors.get('pred_high', {}) if colors else {})
                sl = (colors.get('pred_low', {}) if colors else {})
                if sh.get('visible', True):
                    self.ax.plot(th, ph, sh.get('linestyle', '-'), color=sh.get('color', 'orange'), linewidth=sh.get('linewidth', 1.0), label='预测 高均值')
                if sl.get('visible', True):
                    self.ax.plot(th, pl, sl.get('linestyle', '-'), color=sl.get('color', 'purple'), linewidth=sl.get('linewidth', 1.0), label='预测 低均值')
            elif 'Voltage' in ext_df.columns:
                te = ext_df['Time'].values
                ve = ext_df['Voltage'].values
                se = (colors.get('ext', {}) if colors else {})
                if se.get('visible', True):
                    self.ax.plot(te, ve, '-', color=se.get('color', 'red'), linewidth=se.get('linewidth', 1.2), label='外延样本')

        # 绘制参考区间竖线（红色虚线）
        try:
            y_min, y_max = self.ax.get_ylim()
            if ref_start is not None:
                self.ax.vlines([ref_start], ymin=y_min, ymax=y_max, colors='red', linestyles='--', linewidth=1.0, alpha=0.9)
            if ref_end is not None:
                self.ax.vlines([ref_end], ymin=y_min, ymax=y_max, colors='red', linestyles='--', linewidth=1.0, alpha=0.9)
        except Exception:
            if ref_start is not None:
                self.ax.axvline(ref_start, color='red', linestyle='--', linewidth=1.0, alpha=0.9)
            if ref_end is not None:
                self.ax.axvline(ref_end, color='red', linestyle='--', linewidth=1.0, alpha=0.9)

        # 单周期测试：按点着色显示 high/low（来自 high_low_marks）
        if high_low_marks:
            try:
                if 'high_pts' in high_low_marks and high_low_marks['high_pts'] is not None:
                    th, vh = high_low_marks['high_pts']
                    self.ax.scatter(th, vh, c='orange', s=20, marker='o', label='外延 高点(按点)')
                if 'low_pts' in high_low_marks and high_low_marks['low_pts'] is not None:
                    tl, vl = high_low_marks['low_pts']
                    self.ax.scatter(tl, vl, c='purple', s=20, marker='o', label='外延 低点(按点)')
            except Exception:
                pass

        # 生成图例
        try:
            self.ax.set_xlabel('Time (hours)')
            self.ax.set_ylabel('Voltage')
            # 保存初始视图（仅第一次有数据时）
            if self._initial_view is None:
                try:
                    self._initial_view = (self.ax.get_xlim(), self.ax.get_ylim())
                except Exception:
                    self._initial_view = None
            self.ax.legend(loc='upper right', fontsize='small')
        except Exception:
            self.ax.legend()

        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()