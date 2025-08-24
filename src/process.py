import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os
from typing import Optional

def read_data(file_path):
    """
    简化读取：支持 xlsx 或 csv，直接返回包含 Time, Voltage, Voltage_interpolated 的 DataFrame。
    不做异常检测与插值，假定外部已经标准化数据。
    """
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        # 对于 xlsx，读取第一个 sheet
        df = pd.read_excel(file_path)
    # 保证有两列：Time, Voltage（若列名不同，尝试前两列）
    if 'Time' not in df.columns or 'Voltage' not in df.columns:
        df = df.iloc[:, :2].copy()
        df.columns = ['Time', 'Voltage']
    # 确保数值类型
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
    # 不做异常检测，直接把 Voltage 作为 Voltage_interpolated 返回
    df = df.dropna(subset=['Time']).reset_index(drop=True)
    df['Voltage_interpolated'] = df['Voltage'].copy()
    return df

def detect_outliers(df, col):
    """
    占位实现：不做异常检测，返回全 False（与新工作流兼容）。
    若后续需要可替换为外部标准化输出的异常掩码。
    """
    return np.zeros(len(df), dtype=bool)

def process_data(df, outlier_mask=None):
    """
    简化处理：标记 is_outlier、保留 Voltage_processed（等于 Voltage），
    并保证 Voltage_interpolated 字段存在。
    """
    if outlier_mask is None:
        outlier_mask = np.zeros(len(df), dtype=bool)
    df = df.copy()
    df['is_outlier'] = outlier_mask
    df['Voltage_processed'] = df['Voltage'].copy()
    if 'Voltage_interpolated' not in df.columns:
        df['Voltage_interpolated'] = df['Voltage_processed'].copy()
    else:
        # 如果存在插值列，保证缺失处用原始值填充
        df['Voltage_interpolated'] = df['Voltage_interpolated'].fillna(df['Voltage_processed'])
    return df

def analyze_trend(df):
    times = df['Time'].values
    data = df['Voltage_interpolated'].values
    window = max(11, min(101, int(len(data) * 0.05)))
    if window % 2 == 0: window += 1
    try:
        trend = savgol_filter(data, window, 2)
    except:
        trend = np.convolve(data, np.ones(window)/window, mode='same')
    fluct = data - trend
    return trend, fluct, np.std(fluct)

def extend_data(df,
                target_hour: float = 2000.0,
                ref_hours: float = 300.0,
                ref_start: Optional[float] = None,
                ref_end: Optional[float] = None,
                seed: Optional[int] = None, **kwargs):
    """
    简化外延预测：只返回每周期中点的高/低均值预测曲线（pred_high/pred_low）。
    返回 (final_df, ext_df)：
      - final_df: 原始数据拷贝（未生成逐样本外延）
      - ext_df: DataFrame，列 ['Time','pred_high','pred_low']，Time 为周期中点（未来）

    参数：
      - ref_hours: 若未指定 ref_start/ref_end，则以最后时间向前 ref_hours 小时作为参考区间
      - ref_start/ref_end: 可显式指定参考区间（Time 的值）
      - target_hour: 预测到的最大时间（小时）
      - seed: 可选随机种子（目前仅为接口保留）
    """
    # 基本保护与拷贝
    df2 = df.copy().reset_index(drop=True)
    if len(df2) < 2:
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])

    last_time = float(df2['Time'].iloc[-1])

    # 参考区间选择
    if ref_start is not None and ref_end is not None:
        ref_df = df2[(df2['Time'] >= ref_start) & (df2['Time'] <= ref_end)].copy().reset_index(drop=True)
    else:
        ref_start_time = last_time - float(ref_hours)
        ref_df = df2[df2['Time'] >= ref_start_time].copy().reset_index(drop=True)
    if len(ref_df) < 4:
        # 参考数据不足，返回空预测
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])

    # 估算采样间隔与每 4 小时平台样本数（用于分块对齐）
    times = ref_df['Time'].values
    if len(times) < 2:
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])
    interval = float(np.mean(np.diff(times)))
    cycle_len = max(1, int(round(4.0 / interval)))
    cycle_hours = 8.0

    # 确保 ref_df 长度为平台单元的整数倍（按 cycle_len 裁剪前端）
    n = len(ref_df)
    trim = n % cycle_len
    if trim != 0 and n > cycle_len:
        ref_df = ref_df.iloc[trim:].reset_index(drop=True)

    voltages = ref_df['Voltage_interpolated'].values if 'Voltage_interpolated' in ref_df.columns else ref_df['Voltage'].values

    # 聚类为高/低并分块
    kmeans = KMeans(n_clusters=2, random_state=0).fit(voltages.reshape(-1, 1))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.ravel()
    high_label = int(np.argmax(centers))
    platform = np.where(labels == high_label, 1, 0)
    ref_df['platform'] = platform
    # 构造块
    changes = np.where(np.diff(platform) != 0)[0] + 1
    idx = np.concatenate(([0], changes, [len(platform)]))
    blocks = [(platform[s], s, e) for s, e in zip(idx[:-1], idx[1:])]

    # 清洗块：长度、均值与内部方差阈值过滤（去除异常周期）
    tol = 0.12  # 宽松一点，允许略有抖动
    min_len = max(1, int(cycle_len * (1 - tol)))
    max_len = max(1, int(cycle_len * (1 + tol)))
    grp = ref_df.groupby('platform')['Voltage_interpolated'].mean()
    low_ref = grp.get(0, np.mean(voltages))
    high_ref = grp.get(1, np.mean(voltages))
    clean_blocks = []
    for p, s, e in blocks:
        seg = voltages[s:e]
        seg_len = e - s
        if seg_len < min_len or seg_len > max_len:
            continue
        seg_std = np.std(seg)
        seg_mean = np.mean(seg)
        # 根据类型判断是否接近总体参考均值且波动不大
        if p == 1 and abs(seg_mean - high_ref) < 0.08 and seg_std < 0.06:
            clean_blocks.append((p, s, e))
        if p == 0 and abs(seg_mean - low_ref) < 0.08 and seg_std < 0.06:
            clean_blocks.append((p, s, e))

    # 配对高低为周期（相邻两块一对）
    paired_times = []
    high_means = []
    low_means = []
    i = 0
    while i + 1 < len(clean_blocks):
        p1, s1, e1 = clean_blocks[i]
        p2, s2, e2 = clean_blocks[i+1]
        if p1 == p2:
            i += 1
            continue
        if p1 == 1:
            high_seg = ref_df['Voltage_interpolated'].values[s1:e1]
            low_seg = ref_df['Voltage_interpolated'].values[s2:e2]
        else:
            high_seg = ref_df['Voltage_interpolated'].values[s2:e2]
            low_seg = ref_df['Voltage_interpolated'].values[s1:e1]
        paired_times.append( (ref_df['Time'].values[s1:e1].mean() + ref_df['Time'].values[s2:e2].mean())/2.0 )
        high_means.append(np.mean(high_seg))
        low_means.append(np.mean(low_seg))
        i += 2

    paired_times = np.array(paired_times)
    high_means = np.array(high_means)
    low_means = np.array(low_means)

    if len(paired_times) < 2:
        # 样本不足，返回空预测
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])

    # 对 high_means / low_means 做线性回归并以最后观测为锚点外推
    lr_h = LinearRegression().fit(paired_times.reshape(-1,1), high_means)
    lr_l = LinearRegression().fit(paired_times.reshape(-1,1), low_means)
    last_mid = paired_times[-1]
    last_h = high_means[-1]
    last_l = low_means[-1]
    slope_h = float(lr_h.coef_[0])
    slope_l = float(lr_l.coef_[0])

    # 构造预测周期中点（从下一个周期中点开始）
    ext_times = np.arange(last_mid + cycle_hours, target_hour + cycle_hours, cycle_hours)
    if len(ext_times) == 0:
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])

    pred_high = last_h + slope_h * (ext_times - last_mid)
    pred_low  = last_l + slope_l * (ext_times - last_mid)

    # 输出 ext_df：每周期中点的预测高/低均值
    ext_df = pd.DataFrame({
        'Time': ext_times,
        'pred_high': pred_high,
        'pred_low' : pred_low
    })

    return df2, ext_df

def save_results(processed_df, final_df, file_path):
    """保留导出函数（不做绘图），GUI 可调用此函数导出 xlsx。"""
    base = os.path.splitext(file_path)[0]
    processed_df[['Time','Voltage','is_outlier','Voltage_interpolated']].to_excel(f"{base}_processed.xlsx", index=False)
    final_df[['Time','Voltage','is_extended']].to_excel(f"{base}_final.xlsx", index=False)
    return f"{base}_processed.xlsx", f"{base}_final.xlsx"

def collect_sample_segments(df,
                            sample_start: Optional[float] = None,
                            sample_end: Optional[float] = None,
                            cycle_hours: float = 8.0,
                            cap_k: float = 3.0,
                            tol_frac: float = 0.20):
    """
    采样段收集：cap_k 与 tol_frac 可由 GUI 配置（cap_k: MAD 限幅倍数，tol_frac: 时间容差比例）
    """
    if df is None or len(df) == 0:
        return []

    d = df.copy().reset_index(drop=True)
    if sample_start is not None:
        d = d[d['Time'] >= sample_start].reset_index(drop=True)
    if sample_end is not None:
        d = d[d['Time'] <= sample_end].reset_index(drop=True)
    if len(d) < 2:
        return []

    times = d['Time'].values
    vals = d['Voltage_interpolated'].values if 'Voltage_interpolated' in d.columns else d['Voltage'].values
    interval = float(np.median(np.diff(times)))
    if interval <= 0 or np.isnan(interval):
        return []

    # 全局聚类判断高/低标签，保证不同片段间标签一致
    try:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(vals.reshape(-1,1))
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.ravel()
        high_label = int(np.argmax(centers))
    except Exception:
        med = np.median(vals)
        labels = (vals >= med).astype(int)
        high_label = 1

    # 找到连续块
    changes = np.where(np.diff(labels) != 0)[0] + 1
    idx = np.concatenate(([0], changes, [len(labels)]))
    blocks = [(labels[s], s, e) for s, e in zip(idx[:-1], idx[1:])]

    def _cap_outliers(arr, times_arr, cap_k=3.0):
        """基于 MAD/STD 限幅把离群点拉近中位数（而不是删除）。"""
        if len(arr) <= 2:
            return arr, times_arr
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-8:
            std = np.std(arr)
            thr = max(3.0 * std, 1e-6)
        else:
            thr = max(cap_k * mad, 1e-6)
        # 把每个点限制到 [med-thr, med+thr]
        capped = np.clip(arr, med - thr, med + thr)
        return capped, times_arr

    segs = []
    i = 0
    expected_each = cycle_hours / 2.0  # 4.0 小时

    while i + 1 < len(blocks):
        p1, s1, e1 = blocks[i]
        p2, s2, e2 = blocks[i+1]
        if p1 == p2:
            i += 1
            continue
        if p1 == high_label:
            high_s, high_e = s1, e1
            low_s, low_e = s2, e2
        else:
            high_s, high_e = s2, e2
            low_s, low_e = s1, e1

        high_times = times[high_s:high_e]
        low_times  = times[low_s:low_e]
        high_vals = vals[high_s:high_e].copy()
        low_vals  = vals[low_s:low_e].copy()

        # 点级异常修正（caps 而非删除）
        high_vals_c, high_times_c = _cap_outliers(high_vals, high_times, cap_k=cap_k)
        low_vals_c, low_times_c   = _cap_outliers(low_vals, low_times, cap_k=cap_k)

        # 计算段实际持续时长（包含采样间隔）
        if len(high_times_c) >= 1:
            high_dur = (high_times_c[-1] - high_times_c[0]) + interval if len(high_times_c) > 1 else interval
        else:
            high_dur = 0.0
        if len(low_times_c) >= 1:
            low_dur = (low_times_c[-1] - low_times_c[0]) + interval if len(low_times_c) > 1 else interval
        else:
            low_dur = 0.0

        # 容差放宽：允许任一段在容差范围内即可保留（避免因测点偏移导致全部丢弃）
        ok_high = abs(high_dur - expected_each) <= tol_frac * expected_each
        ok_low  = abs(low_dur  - expected_each) <= tol_frac * expected_each
        if not (ok_high or ok_low):
            i += 2
            continue

        total_len = len(high_vals_c) + len(low_vals_c)
        if total_len < 2:
            i += 2
            continue

        combined_vals = np.concatenate([high_vals_c, low_vals_c])
        combined_times = np.concatenate([high_times_c, low_times_c])
        high_mean = float(np.mean(high_vals_c)) if len(high_vals_c) > 0 else float(np.max(combined_vals))
        low_mean  = float(np.mean(low_vals_c))  if len(low_vals_c)  > 0 else float(np.min(combined_vals))

        mid = 0.5*(high_mean + low_mean)
        diff = high_mean - low_mean
        segs.append({
            'times': combined_times,
            'values': combined_vals,
            'high_vals': high_vals_c,
            'low_vals': low_vals_c,
            'high_mean': high_mean,
            'low_mean': low_mean,
            'mid': float(mid),
            'diff': float(diff)
        })
        i += 2

    return segs


def synthesize_extension_from_samples(processed_df,
                                      ext_cycles_df,
                                      sample_segments,
                                      last_time: Optional[float] = None,
                                      interval: Optional[float] = None,
                                      cycle_hours: float = 8.0,
                                      rng_seed: Optional[int] = None,
                                      stretch_limit: float = 0.25,
                                      residual_scale: float = 1.0,
                                      start_phase: Optional[str] = None,
                                      cap_k: float = 10.0,
                                      abs_cap_mult: float = 20.0,
                                      strict_truncate: bool = False):
    """
    合成逐样本外延（改进）：
      - 默认 cap_k=10, abs_cap_mult=20（更宽松）
      - strict_truncate: 若为 True，则首周期严格按 remaining 截断（不会通过相位重排来对齐）
    """
    import numpy as _np
    import pandas as _pd

    if ext_cycles_df is None or len(ext_cycles_df) == 0 or len(sample_segments) == 0:
        return _pd.DataFrame(columns=['Time', 'Voltage', 'is_extended'])

    if last_time is None:
        last_time = float(processed_df['Time'].iloc[-1])
    if interval is None:
        times = processed_df['Time'].values
        interval = float(_np.median(_np.diff(times))) if len(times) > 1 else 1.0
        if interval <= 0:
            interval = 1.0

    rng = _np.random.default_rng(rng_seed)

    expected_each = cycle_hours / 2.0  # 4h
    desired_len_full = max(1, int(round(cycle_hours / interval)))

    # ---- 推断或使用 start_phase，并估算已持续时间 elapsed ----
    vals = processed_df['Voltage_interpolated'].values if 'Voltage_interpolated' in processed_df.columns else processed_df['Voltage'].values
    last_val = float(vals[-1])
    inferred_phase = 'high'
    elapsed = 0.0
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, random_state=0).fit(vals.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
        high_label = int(_np.argmax(centers))
        last_lab = int(km.predict([[last_val]])[0])
        inferred_phase = 'high' if last_lab == high_label else 'low'
        labels = km.predict(vals.reshape(-1, 1))
        # 反向扫描找到当前 run 的起始索引
        idx = len(labels) - 1
        while idx > 0 and labels[idx-1] == labels[idx]:
            idx -= 1
        start_of_run_time = float(processed_df['Time'].iat[idx])
        elapsed = max(0.0, last_time - start_of_run_time + interval)
    except Exception:
        med = float(_np.median(vals))
        inferred_phase = 'high' if last_val >= med else 'low'
        elapsed = 0.0

    # 优先使用传入的 start_phase
    start_phase_used = str(start_phase) if start_phase is not None else inferred_phase

    remaining = max(0.0, expected_each - elapsed)
    remaining = min(remaining, expected_each)
    first_cycle_duration = remaining + expected_each
    first_cycle_n = max(1, int(round(first_cycle_duration / interval)))

    # 全局波动尺度（用于绝对截断）
    global_diff_scale = max(1e-6, float(_np.std(vals)))

    all_times = []
    all_values = []

    current_start = last_time  # 第一个外延点为 last_time + interval

    # helper: 限幅并缩放到目标 mid（使用模块内 clip_and_shift_deviations）
    def _clip_and_shift(resampled, sample_mid, sample_diff, target_high, target_low):
        try:
            return clip_and_shift_deviations(resampled, sample_mid, sample_diff, target_high, target_low,
                                             global_diff_scale, cap_k=cap_k, abs_cap_mult=abs_cap_mult, stretch_limit=stretch_limit)
        except Exception:
            # 兜底本地实现（保持与 helper 相同逻辑）
            target_diff = abs(float(target_high - target_low))
            if abs(sample_diff) < 1e-8:
                max_dev_base = max(0.5 * target_diff, 2.0 * global_diff_scale)
            else:
                max_dev_base = max(cap_k * (abs(sample_diff) * 0.25), 0.4 * target_diff, 2.0 * global_diff_scale)

            dev = resampled - sample_mid
            dev_clipped = _np.clip(dev, -max_dev_base, max_dev_base)

            if abs(sample_diff) < 1e-8:
                scale = 1.0
            else:
                raw_scale = (float(target_high) - float(target_low)) / sample_diff
                scale = float(_np.clip(raw_scale, 1.0 - stretch_limit, 1.0 + stretch_limit))

            new_vals = sample_mid + dev_clipped * scale
            target_mid = 0.5 * (float(target_high) + float(target_low))
            new_vals = new_vals - sample_mid + target_mid

            abs_cap = max(0.6 * target_diff if target_diff > 0 else 0.0, abs_cap_mult * float(max(1e-6, global_diff_scale)))
            new_vals = _np.clip(new_vals, target_mid - abs_cap, target_mid + abs_cap)
            return new_vals

    # 生成循环：首周期有两种模式：
    # - strict_truncate=True: 首周期按 remaining 严格截断（n_current = int(round(remaining/interval))），然后拼接对端完整；不做相位重排。
    # - strict_truncate=False: 采用按时间比例分配点数的方式（保留原行为）
    for cycle_idx, row in enumerate(ext_cycles_df.itertuples(index=False)):
        if cycle_idx == 0:
            # 当前平台是否 high？
            current_is_high = (start_phase_used == 'high')
            if strict_truncate:
                # 严格用剩余时长决定首段点数（尽可能精确）
                n_current = max(0, int(round(remaining / interval)))
                # 首周期总点数 first_cycle_n；剩余点数给对端
                n_other = max(0, first_cycle_n - n_current)
                if current_is_high:
                    high_len_target = max(0, n_current)
                    low_len_target = max(0, n_other)
                    use_high_first_flag = True
                else:
                    high_len_target = max(0, n_other)
                    low_len_target = max(0, n_current)
                    use_high_first_flag = False
                n_points = high_len_target + low_len_target
                # 若 n_points==0，退回至少 1 点
                if n_points == 0:
                    n_points = max(1, first_cycle_n)
                    # fallback: 把高位留至少1点
                    high_len_target = max(1, high_len_target)
                    low_len_target = n_points - high_len_target
            else:
                # 原来的按比例分配方式（保持兼容）
                n_points = first_cycle_n
                current_dur = remaining
                other_dur = expected_each
                total_first = current_dur + other_dur if (current_dur + other_dur) > 0 else expected_each*2
                ratio_current = current_dur / total_first if total_first>0 else 0.5
                n_current = max(1, int(round(n_points * ratio_current)))
                n_other = max(0, n_points - n_current)
                if current_is_high:
                    high_len_target = n_current
                    low_len_target = n_other
                    use_high_first_flag = True
                else:
                    high_len_target = n_other
                    low_len_target = n_current
                    use_high_first_flag = False
        else:
            # 后续周期均为完整周期（高先低后）
            n_points = desired_len_full
            seg = sample_segments[rng.integers(0, len(sample_segments))]
            high_vals = _np.asarray(seg['high_vals']).astype(float)
            low_vals = _np.asarray(seg['low_vals']).astype(float)
            total_orig = len(high_vals) + len(low_vals) if (len(high_vals) + len(low_vals)) > 0 else 1
            high_len_target = max(1, int(round(n_points * (len(high_vals) / total_orig))))
            low_len_target = max(0, n_points - high_len_target)
            use_high_first_flag = True

        # 确保有选定的 sample seg
        if cycle_idx == 0:
            seg = sample_segments[rng.integers(0, len(sample_segments))]
            high_vals = _np.asarray(seg['high_vals']).astype(float)
            low_vals = _np.asarray(seg['low_vals']).astype(float)

        # 重采样并构造 resampled（保持 high then low 存储顺序，use_high_first_flag 控制拼接顺序）
        def _resample_arr(arr, target_n):
            if target_n <= 0:
                return _np.array([])
            if len(arr) == 0:
                return _np.full(target_n, 0.0)
            orig_x = _np.linspace(0.0, 1.0, num=len(arr))
            target_x = _np.linspace(0.0, 1.0, num=target_n)
            return _np.interp(target_x, orig_x, arr)

        res_high = _resample_arr(high_vals, high_len_target)
        res_low = _resample_arr(low_vals, low_len_target)

        if use_high_first_flag:
            resampled = _np.concatenate([res_high, res_low]) if (len(res_high) + len(res_low)) > 0 else _np.array([])
        else:
            resampled = _np.concatenate([res_low, res_high]) if (len(res_high) + len(res_low)) > 0 else _np.array([])

        # 填充/截断到 n_points
        if len(resampled) < n_points:
            resampled = _np.pad(resampled, (0, n_points - len(resampled)), mode='edge')
        elif len(resampled) > n_points:
            resampled = resampled[:n_points]

        sample_high_mean = float(seg.get('high_mean', _np.mean(high_vals) if len(high_vals) > 0 else 0.0))
        sample_low_mean = float(seg.get('low_mean', _np.mean(low_vals) if len(low_vals) > 0 else 0.0))
        sample_mid = 0.5 * (sample_high_mean + sample_low_mean)
        sample_diff = float(seg.get('diff', sample_high_mean - sample_low_mean))

        target_high = float(getattr(row, 'pred_high', sample_high_mean))
        target_low = float(getattr(row, 'pred_low', sample_low_mean))

        # 使用 helper 对单点偏差进行限幅并缩放到目标 mid
        new_vals = _clip_and_shift(resampled, sample_mid, sample_diff, target_high, target_low)

        # 时间轴：首个外延点为 last_time + interval，随后连续
        cycle_times = current_start + _np.arange(1, n_points + 1) * interval
        all_times.extend(cycle_times.tolist())
        all_values.extend(new_vals.tolist())

        # 连续拼接
        current_start = current_start + n_points * interval

    ext_df_samples = _pd.DataFrame({'Time': _np.array(all_times), 'Voltage': _np.array(all_values), 'is_extended': True})
    ext_df_samples = ext_df_samples.sort_values('Time').reset_index(drop=True)
    return ext_df_samples

def compute_first_cycle_params(processed_df, interval: float, cycle_hours: float = 8.0, start_phase_override: Optional[str] = None):
    """
    计算首个外延周期所需参数：
      - 判断最后点属于 high/low（返回 start_phase 'high'/'low'）
      - 反向扫描估算该平台已持续时间 elapsed（小时）
      - 计算 remaining = max(0, 4h - elapsed)
      - 返回 dict 包含: start_phase (str), elapsed (float), remaining (float),
        first_cycle_duration (float), first_cycle_n (int)
    任何失败均以安全默认返回（elapsed=0, start_phase='high'或按中位数判断）。
    """
    import numpy as _np
    try:
        vals = processed_df['Voltage_interpolated'].values if 'Voltage_interpolated' in processed_df.columns else processed_df['Voltage'].values
        last_time = float(processed_df['Time'].iat[-1])
        expected_each = cycle_hours / 2.0
        # 尝试用 KMeans 判断 label
        try:
            km = KMeans(n_clusters=2, random_state=0).fit(vals.reshape(-1,1))
            centers = km.cluster_centers_.ravel()
            high_label = int(_np.argmax(centers))
            last_val = float(processed_df['Voltage_interpolated'].iat[-1] if 'Voltage_interpolated' in processed_df.columns else processed_df['Voltage'].iat[-1])
            last_lab = int(km.predict([[last_val]])[0])
            inferred_high = (last_lab == high_label)
            labels = km.predict(vals.reshape(-1,1))
            # 反向扫描该 label 连续块开始时间
            idx = len(labels) - 1
            while idx > 0 and labels[idx-1] == labels[idx]:
                idx -= 1
            start_of_run_time = float(processed_df['Time'].iat[idx])
            elapsed = max(0.0, last_time - start_of_run_time + interval)
            start_phase = 'high' if inferred_high else 'low'
        except Exception:
            med = float(_np.median(vals))
            last_val = float(processed_df['Voltage_interpolated'].iat[-1] if 'Voltage_interpolated' in processed_df.columns else processed_df['Voltage'].iat[-1])
            start_phase = 'high' if last_val >= med else 'low'
            elapsed = 0.0
        # 覆盖 start_phase 如果用户传入 override
        if start_phase_override is not None:
            start_phase = str(start_phase_override)
        remaining = max(0.0, expected_each - elapsed)
        remaining = min(remaining, expected_each)
        first_cycle_duration = remaining + expected_each
        first_cycle_n = max(1, int(round(first_cycle_duration / interval)))
        return {
            'start_phase': start_phase,
            'elapsed': float(elapsed),
            'remaining': float(remaining),
            'first_cycle_duration': float(first_cycle_duration),
            'first_cycle_n': int(first_cycle_n)
        }
    except Exception:
        # 保守默认
        expected_each = cycle_hours / 2.0
        first_cycle_n = max(1, int(round(expected_each * 2.0 / (interval if interval>0 else 1.0))))
        return {'start_phase':'high','elapsed':0.0,'remaining':expected_each,'first_cycle_duration':expected_each*2.0,'first_cycle_n':first_cycle_n}


def clip_and_shift_deviations(resampled, sample_mid, sample_diff, target_high, target_low, global_diff_scale,
                              cap_k=3.0, abs_cap_mult=3.0, stretch_limit=0.25):
    """
    将 resampled（以 sample_mid 为中心的数组）进行两步处理：
      1) 对单点偏差按 cap_k(MAD/std) 限幅（避免极端点保持原值）
      2) 按样本 diff -> 目标 diff 做缩放（限幅 stretch_limit），并平移到 target_mid
      3) 对最终值进行绝对幅度截断，避免孤立极端点（abs_cap_mult * global_diff_scale 或基于 target_diff）
    返回 new_vals numpy 数组。
    """
    import numpy as _np
    try:
        target_diff = abs(float(target_high - target_low))
        # 估算原始偏差幅度基准
        if abs(sample_diff) < 1e-8:
            max_dev_base = max(0.5 * target_diff, 2.0 * global_diff_scale)
        else:
            # 使用 cap_k 控制单点限幅灵敏度
            max_dev_base = max(cap_k * (abs(sample_diff) * 0.25), 0.4 * target_diff, 2.0 * global_diff_scale)

        dev = resampled - sample_mid
        dev_clipped = _np.clip(dev, -max_dev_base, max_dev_base)

        # 缩放到目标差值
        if abs(sample_diff) < 1e-8:
            scale = 1.0
        else:
            raw_scale = (float(target_high) - float(target_low)) / sample_diff
            scale = float(_np.clip(raw_scale, 1.0 - stretch_limit, 1.0 + stretch_limit))

        new_vals = sample_mid + dev_clipped * scale
        # 平移到 target_mid
        target_mid = 0.5 * (float(target_high) + float(target_low))
        new_vals = new_vals - sample_mid + target_mid

        # 绝对幅度限幅：abs_cap_mult * global_diff_scale 或 0.6 * target_diff，两者取最大以保留必要差值
        abs_cap = max(0.6 * target_diff if target_diff>0 else 0.0, abs_cap_mult * float(max(1e-6, global_diff_scale)))
        new_vals = _np.clip(new_vals, target_mid - abs_cap, target_mid + abs_cap)
        return new_vals
    except Exception:
        return resampled - 0.0  # 退回原始（安全）