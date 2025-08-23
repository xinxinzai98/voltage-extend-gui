from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from matplotlib.widgets import RectangleSelector

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.fig = Figure(figsize=(8,6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (hours)')
        self.ax.set_ylabel('Voltage')

        # 交互状态：矩形选择（实线细框） + 鼠标中键平移 + 鼠标滚轮以光标为中心缩放
        try:
            self._rect_selector = RectangleSelector(
                self.ax,
                self.on_rect_select,
                useblit=True,
                button=[1],
                interactive=True,
                # 实线、细边框以提示框选范围
                rectprops={'edgecolor': 'black', 'facecolor': 'none', 'linestyle': '-', 'linewidth': 0.8}
            )
            try:
                self._rect_selector.set_active(False)
            except Exception:
                pass
        except Exception:
            # 如果当前 matplotlib 版本不兼容 RectangleSelector 的某些参数，回退到最简构造
            try:
                self._rect_selector = RectangleSelector(self.ax, self.on_rect_select)
                try:
                    self._rect_selector.set_active(False)
                except Exception:
                    pass
            except Exception:
                self._rect_selector = None
        self._pan = False
        self._pan_start = None

        # 连接鼠标事件：中键平移、右键重置
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        # 鼠标滚轮缩放（以鼠标指针处为中心）
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        # 保存上次自动缩放范围
        self._last_xlim = None
        self._last_ylim = None

    def update_plot(self, processed_df, final_df, ext_df, colors=None):
        import pandas as pd
        self.ax.clear()
        valid = processed_df[~processed_df['is_outlier']]

        # 支持两种传参名：colors（向后兼容）或 styles（含 linewidth/visible/linestyle）
        styles = colors if colors is not None else {}
        def get_style(key, default_color, default_ls='-'):
            item = styles.get(key, {})
            return {
                'color': item.get('color', default_color),
                'linestyle': item.get('linestyle', default_ls),
                'visible': item.get('visible', True),
                'linewidth': float(item.get('linewidth', 1.0))
            }

        s_orig = get_style('original', 'blue', default_ls='')
        s_interp = get_style('interp', 'green', default_ls='-')
        s_ext = get_style('ext', 'red', default_ls='-')
        s_ph = get_style('pred_high', 'orange', default_ls='--')
        s_pl = get_style('pred_low', 'purple', default_ls='--')

        # 原始点与插值线与外延线
        if s_orig['visible']:
            self.ax.plot(valid['Time'], valid['Voltage'], 'o', color=s_orig['color'], label='Original', markersize=3, alpha=0.8, linewidth=s_orig['linewidth'])
        if s_interp['visible']:
            ls = '' if s_interp['linestyle'] in ('', 'None') else s_interp['linestyle']
            self.ax.plot(valid['Time'], valid['Voltage_interpolated'], ls, color=s_interp['color'], label='Interpolated', alpha=0.9, linewidth=s_interp['linewidth'])
        if ext_df is not None and len(ext_df) > 0 and s_ext['visible']:
            ls = '' if s_ext['linestyle'] in ('', 'None') else s_ext['linestyle']
            self.ax.plot(ext_df['Time'], ext_df['Voltage'], ls, color=s_ext['color'], label='Extended', linewidth=s_ext['linewidth'], alpha=0.9)

        # 绘制预测高/低线：优先使用 final_df 是否包含预测列，否则用外延数据局部极值近似
        def compute_local_envelope(times, vals, win_pts=10):
            n = len(vals)
            high = [None]*n
            low = [None]*n
            for i in range(n):
                a = max(0, i-win_pts)
                b = min(n, i+win_pts+1)
                seg = vals[a:b]
                if len(seg) == 0:
                    high[i] = low[i] = vals[i]
                else:
                    high[i] = max(seg)
                    low[i] = min(seg)
            return np.array(high), np.array(low)

        # 优先检查 final_df/ext_df 是否包含专用列 'pred_high'/'pred_low'
        ph_x = ph_y = pl_x = pl_y = None
        if final_df is not None and {'pred_high','pred_low'}.issubset(final_df.columns):
            ph_x = final_df['Time'].values
            ph_y = final_df['pred_high'].values
            pl_x = final_df['Time'].values
            pl_y = final_df['pred_low'].values
        elif ext_df is not None and len(ext_df) > 0:
            t_ext = np.asarray(ext_df['Time'].values)
            v_ext = np.asarray(ext_df['Voltage'].values)
            # 使用局部窗口点数与外延长度相关
            win = max(3, int(len(v_ext)/80))
            high_env, low_env = compute_local_envelope(t_ext, v_ext, win_pts=win)
            ph_x, ph_y = t_ext, high_env
            pl_x, pl_y = t_ext, low_env

        # 绘制预测线且保持周期间平均高低差不被图形绘制变形（这里只绘制包络线）
        if ph_x is not None and pl_x is not None:
            if s_ph['visible']:
                self.ax.plot(ph_x, ph_y, linestyle=s_ph['linestyle'] if s_ph['linestyle']!='None' else '-', color=s_ph['color'], linewidth=s_ph['linewidth'], label='Predicted High')
            if s_pl['visible']:
                self.ax.plot(pl_x, pl_y, linestyle=s_pl['linestyle'] if s_pl['linestyle']!='None' else '-', color=s_pl['color'], linewidth=s_pl['linewidth'], label='Predicted Low')

        # 保存当前自动缩放范围
        try:
            self._last_xlim = self.ax.get_xlim()
            self._last_ylim = self.ax.get_ylim()
        except Exception:
            self._last_xlim = None
            self._last_ylim = None

        self.ax.set_ylim(1.4, 2.1)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    # 矩形缩放回调（在 RectangleSelector 激活状态下触发）
    def on_rect_select(self, eclick, erelease):
        try:
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if None in (x1, x2, y1, y2):
                return
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                return
            self.ax.set_xlim((xmin, xmax))
            self.ax.set_ylim((ymin, ymax))
            # 关闭矩形选择工具（一次性使用）
            self._rect_selector.set_active(False)
            self.canvas.draw()
        except Exception:
            pass

    # 切换矩形缩放工具
    def toggle_rect_zoom(self):
        active = self._rect_selector.active
        self._rect_selector.set_active(not active)

    # 缩放函数（以当前中心缩放）
    def zoom(self, scale_factor):
        try:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            xc = 0.5*(x0+x1)
            yc = 0.5*(y0+y1)
            new_half_x = 0.5*(x1-x0) * scale_factor
            new_half_y = 0.5*(y1-y0) * scale_factor
            self.ax.set_xlim(xc - new_half_x, xc + new_half_x)
            self.ax.set_ylim(yc - new_half_y, yc + new_half_y)
            self.canvas.draw()
        except Exception:
            pass

    # 鼠标事件：中键按下开始平移，右键单击复位视图
    def _on_mouse_press(self, event):
        if event.button == 2:  # 中键开始平移
            self._pan = True
            self._pan_start = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
        elif event.button == 3:  # 右键重置视图
            self.reset_view()

    def _on_mouse_release(self, event):
        if event.button == 2:
            self._pan = False
            self._pan_start = None

    def _on_mouse_move(self, event):
        if self._pan and self._pan_start and event.xdata is not None and event.ydata is not None:
            x0, y0, (xlim0, xlim1), (ylim0, ylim1) = (self._pan_start[0], self._pan_start[1], self._pan_start[2], self._pan_start[3])
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            # 以像素位移估算数据坐标位移
            try:
                inv = self.ax.transData.inverted()
                p0 = inv.transform((self._pan_start[0], self._pan_start[1]))
                p1 = inv.transform((event.x, event.y))
                dx_data = p0[0] - p1[0]
                dy_data = p0[1] - p1[1]
                self.ax.set_xlim(self._pan_start[2][0] + dx_data, self._pan_start[2][1] + dx_data)
                self.ax.set_ylim(self._pan_start[3][0] + dy_data, self._pan_start[3][1] + dy_data)
                self.canvas.draw()
            except Exception:
                pass

    def _on_scroll(self, event):
        """鼠标滚轮缩放：以 event.xdata/event.ydata 为中心逐步放大/缩小"""
        try:
            if event.inaxes != self.ax:
                return
            # 向上滚动放大，向下滚动缩小（兼容不同 matplotlib，event.button 可能为 'up'/'down' 或 None）
            direction = getattr(event, 'button', None)
            if direction is None:
                # 部分后端使用 step 属性（正数为上）
                step = getattr(event, 'step', 1)
                zoom_in = step > 0
            else:
                zoom_in = (direction == 'up')

            factor = 0.9 if zoom_in else 1.1
            xdata = event.xdata
            ydata = event.ydata
            if xdata is None or ydata is None:
                return
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            # 缩放相对于鼠标位置保持该点位置相对不变
            left = xdata - (xdata - x0) * factor
            right = xdata + (x1 - xdata) * factor
            bottom = ydata - (ydata - y0) * factor
            top = ydata + (y1 - ydata) * factor
            self.ax.set_xlim(left, right)
            self.ax.set_ylim(bottom, top)
            self.canvas.draw()
        except Exception:
            pass

    def reset_view(self):
        try:
            if self._last_xlim is not None and self._last_ylim is not None:
                self.ax.set_xlim(self._last_xlim)
                self.ax.set_ylim(self._last_ylim)
            else:
                self.ax.relim()
                self.ax.autoscale_view()
            self.canvas.draw()
        except Exception:
            pass

    def save_figure(self, path):
        """保存当前 fig 到文件（支持 png/pdf 等）"""
        try:
            self.fig.savefig(path, dpi=200, bbox_inches='tight')
        except Exception as e:
            raise