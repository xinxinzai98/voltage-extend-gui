import os
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

# 下面两个依赖较重，按需导入以加快模块加载
# from scipy.signal import savgol_filter
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LinearRegression

# ---------- 公共 helper ----------
def _to_array(series_or_arr) -> np.ndarray:
    a = np.asarray(series_or_arr)
    if a.ndim == 0:
        return a.reshape(1,)
    return a

def _resample_arr(arr: np.ndarray, target_n: int) -> np.ndarray:
    """均匀重采样（线性插值），空数组返回指定常数填充。"""
    if target_n <= 0:
        return np.array([])
    arr = _to_array(arr).astype(float)
    if arr.size == 0:
        return np.full(target_n, 0.0, dtype=float)
    if arr.size == 1:
        return np.full(target_n, float(arr[0]), dtype=float)
    orig_x = np.linspace(0.0, 1.0, num=arr.size)
    target_x = np.linspace(0.0, 1.0, num=target_n)
    return np.interp(target_x, orig_x, arr).astype(float)

def _fit_kmeans_if_needed(vals: np.ndarray, n_clusters: int = 2, random_state: int = 0):
    """按需导入 sklearn 并训练 KMeans；失败则返回 None。"""
    try:
        from sklearn.cluster import KMeans
        vals2 = _to_array(vals).reshape(-1, 1)
        km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(vals2)
        return km
    except Exception:
        return None

# ---------- I/O 与基础处理 ----------
def read_data(file_path: str) -> pd.DataFrame:
    """读取 csv/xlsx，确保包含 Time, Voltage, Voltage_interpolated 列。"""
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    if 'Time' not in df.columns or 'Voltage' not in df.columns:
        df = df.iloc[:, :2].copy()
        df.columns = ['Time', 'Voltage']
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
    df = df.dropna(subset=['Time']).reset_index(drop=True)
    if 'Voltage_interpolated' not in df.columns:
        df['Voltage_interpolated'] = df['Voltage'].copy()
    else:
        df['Voltage_interpolated'] = df['Voltage_interpolated'].fillna(df['Voltage'])
    return df

def detect_outliers(df: pd.DataFrame, col: str) -> np.ndarray:
    """基于 IQR 的离群点检测（兼容 dataclean.py 的行为）。
    返回与 df 等长的 bool 掩码，True 表示被判为异常（包括等于 0 或 NaN 的点）。
    """
    data = np.asarray(df[col].values, dtype=float)
    # 标记 NaN 为异常
    nan_mask = np.isnan(data)
    if data.size == 0:
        return np.zeros(0, dtype=bool)
    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (data < lower) | (data > upper) | (data == 0) | nan_mask
    return mask.astype(bool)


def shift_fill_nan(df: pd.DataFrame, col: str, mask: np.ndarray) -> pd.DataFrame:
    """把 mask 标记的点先设为 NaN，然后按 dataclean.py 中的“顺移填补”逻辑填值：
    即把后面第一个非 NaN 的值向前移动到遇到的第一个 NaN 处，原先非 NaN 的位置置为 NaN。
    该函数返回处理后的副本，不修改原 df。
    """
    df_clean = df.copy().reset_index(drop=True)
    vals = df_clean[col].astype(float).values  # 直接操作 numpy 数组
    # 标记异常为 NaN
    vals = vals.copy()
    mask_arr = np.asarray(mask, dtype=bool)
    vals[mask_arr] = np.nan

    n = len(vals)
    i = 0
    while i < n:
        if np.isnan(vals[i]):
            # 找后面第一个非NaN
            j = i + 1
            while j < n and np.isnan(vals[j]):
                j += 1
            if j < n:
                # 把后面的值搬到前面，后面的位置置为 NaN（与 dataclean.py 行为一致）
                vals[i] = vals[j]
                vals[j] = np.nan
        i += 1
    df_clean[col] = vals
    return df_clean


def clean_raw_data(df: pd.DataFrame, col: str = 'Voltage') -> pd.DataFrame:
    """集成 dataclean 的流程：检测异常 -> 顺移填补 -> 丢弃仍为 NaN 的点。
    返回一个新的 DataFrame（已重置索引），并包含处理统计信息作为属性（.attrs）。
    """
    if df is None or len(df) == 0:
        return df.copy()
    mask = detect_outliers(df, col)
    df_filled = shift_fill_nan(df, col, mask)
    df_clean = df_filled.dropna(subset=[col]).reset_index(drop=True)
    # 保存一些简短统计供调试（不影响运行）
    try:
        df_clean.attrs['dataclean_in_count'] = int(len(df))
        df_clean.attrs['dataclean_outliers'] = int(mask.sum())
        df_clean.attrs['dataclean_out_count'] = int(len(df_clean))
    except Exception:
        pass
    return df_clean


def process_data(df: pd.DataFrame, outlier_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    最小化处理：不做插值处理（不生成平滑或填补值）。
    仅标记 outlier、保留原始 Voltage 作为 Voltage_processed，并把 Voltage_interpolated
    直接指向处理后的原始值，方便现有下游代码继续使用 Voltage_interpolated 字段而不改变值。
    """
    if df is None:
        return df
    if outlier_mask is None:
        outlier_mask = np.zeros(len(df), dtype=bool)
    df2 = df.copy().reset_index(drop=True)
    df2['is_outlier'] = np.asarray(outlier_mask, dtype=bool)
    # 保留原始电压作为 processed 值（不进行插值/平滑）
    df2['Voltage_processed'] = df2['Voltage'].copy()
    # 为兼容现有代码，保留 Voltage_interpolated 列，但其内容等同于原始处理值（无插值）
    df2['Voltage_interpolated'] = df2['Voltage_processed']
    return df2

def analyze_trend(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    """返回 trend, fluctuation, std(fluctuation)。尽量使用 numpy 运算。"""
    data = _to_array(df['Voltage_interpolated'].values)
    n = data.size
    window = max(11, min(101, max(11, int(n * 0.05))))
    if window % 2 == 0:
        window += 1
    try:
        from scipy.signal import savgol_filter
        trend = savgol_filter(data, window, 2)
    except Exception:
        trend = np.convolve(data, np.ones(window)/window, mode='same')
    fluct = data - trend
    return trend, fluct, float(np.std(fluct))

# ---------- extend_data（少改，保留主要逻辑但用 helper） ----------
def extend_data(df: pd.DataFrame,
                target_hour: float = 2000.0,
                ref_hours: float = 300.0,
                ref_start: Optional[float] = None,
                ref_end: Optional[float] = None,
                seed: Optional[int] = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.copy().reset_index(drop=True)
    if len(df2) < 2:
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])

    last_time = float(df2['Time'].iloc[-1])
    if ref_start is not None and ref_end is not None:
        ref_df = df2[(df2['Time'] >= ref_start) & (df2['Time'] <= ref_end)].reset_index(drop=True)
    else:
        ref_start_time = last_time - float(ref_hours)
        ref_df = df2[df2['Time'] >= ref_start_time].reset_index(drop=True)
    if len(ref_df) < 4:
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])

    times = ref_df['Time'].values
    interval = float(np.mean(np.diff(times)))
    cycle_len = max(1, int(round(4.0 / interval)))
    cycle_hours = 8.0

    # 裁剪到整周期
    n = len(ref_df)
    trim = n % cycle_len
    if trim != 0 and n > cycle_len:
        ref_df = ref_df.iloc[trim:].reset_index(drop=True)

    voltages = _to_array(ref_df['Voltage_interpolated'].values)

    km = _fit_kmeans_if_needed(voltages)
    if km is not None:
        labels = km.labels_
        centers = km.cluster_centers_.ravel()
        high_label = int(np.argmax(centers))
    else:
        med = float(np.median(voltages))
        labels = (voltages >= med).astype(int)
        high_label = 1

    platform = np.where(labels == high_label, 1, 0)
    ref_df = ref_df.assign(platform=platform)

    # 构造块并清洗（少量容差）
    changes = np.where(np.diff(platform) != 0)[0] + 1
    idx = np.concatenate(([0], changes, [len(platform)]))
    blocks = [(platform[s], s, e) for s, e in zip(idx[:-1], idx[1:])]

    tol = 0.12
    min_len = max(1, int(cycle_len * (1 - tol)))
    max_len = max(1, int(cycle_len * (1 + tol)))
    grp = ref_df.groupby('platform')['Voltage_interpolated'].mean()
    low_ref = grp.get(0, float(np.mean(voltages)))
    high_ref = grp.get(1, float(np.mean(voltages)))

    clean_blocks = []
    for p, s, e in blocks:
        seg = voltages[s:e]
        seg_len = e - s
        if seg_len < min_len or seg_len > max_len:
            continue
        seg_std = float(np.std(seg))
        seg_mean = float(np.mean(seg))
        if p == 1 and abs(seg_mean - high_ref) < 0.08 and seg_std < 0.06:
            clean_blocks.append((p, s, e))
        if p == 0 and abs(seg_mean - low_ref) < 0.08 and seg_std < 0.06:
            clean_blocks.append((p, s, e))

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
        high_means.append(float(np.mean(high_seg)))
        low_means.append(float(np.mean(low_seg)))
        i += 2

    paired_times = np.array(paired_times)
    high_means = np.array(high_means)
    low_means = np.array(low_means)

    if len(paired_times) < 2:
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])

    try:
        from sklearn.linear_model import LinearRegression
        lr_h = LinearRegression().fit(paired_times.reshape(-1,1), high_means)
        lr_l = LinearRegression().fit(paired_times.reshape(-1,1), low_means)
        slope_h = float(lr_h.coef_[0])
        slope_l = float(lr_l.coef_[0])
    except Exception:
        slope_h = 0.0
        slope_l = 0.0

    last_mid = paired_times[-1]
    last_h = high_means[-1]
    last_l = low_means[-1]

    ext_times = np.arange(last_mid + cycle_hours, target_hour + cycle_hours, cycle_hours)
    if len(ext_times) == 0:
        return df2, pd.DataFrame(columns=['Time', 'pred_high', 'pred_low'])

    pred_high = last_h + slope_h * (ext_times - last_mid)
    pred_low  = last_l + slope_l * (ext_times - last_mid)

    ext_df = pd.DataFrame({
        'Time': ext_times,
        'pred_high': pred_high,
        'pred_low' : pred_low
    })

    return df2, ext_df

# ---------- 采样段收集与外延合成（重用 helpers） ----------
def collect_sample_segments(df: pd.DataFrame,
                            sample_start: Optional[float] = None,
                            sample_end: Optional[float] = None,
                            cycle_hours: float = 8.0,
                            cap_k: float = 3.0,
                            tol_frac: float = 0.20) -> List[Dict[str, Any]]:
    """
    收集样本段：已取消对片内点的限幅（不再对 high/low 做 MAD/阈值截断），
    仅按分段和时间容差筛选片段并返回原始片段数据（方便保留原始波动）。
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
    vals = _to_array(d['Voltage_interpolated'].values)
    interval = float(np.median(np.diff(times)))
    if interval <= 0 or np.isnan(interval):
        return []

    km = _fit_kmeans_if_needed(vals)
    if km is not None:
        labels = km.predict(vals.reshape(-1,1))
        centers = km.cluster_centers_.ravel()
        high_label = int(np.argmax(centers))
    else:
        med = float(np.median(vals))
        labels = (vals >= med).astype(int)
        high_label = 1

    changes = np.where(np.diff(labels) != 0)[0] + 1
    idx = np.concatenate(([0], changes, [len(labels)]))
    blocks = [(labels[s], s, e) for s, e in zip(idx[:-1], idx[1:])]

    segs = []
    i = 0
    expected_each = cycle_hours / 2.0

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

        # 直接使用原始片段数据（取消内部限幅）
        high_vals = _to_array(vals[high_s:high_e])
        low_vals  = _to_array(vals[low_s:low_e])

        if high_vals.size > 1:
            high_dur = (times[high_e-1] - times[high_s]) + interval
        else:
            high_dur = interval if high_vals.size==1 else 0.0
        if low_vals.size > 1:
            low_dur = (times[low_e-1] - times[low_s]) + interval
        else:
            low_dur = interval if low_vals.size==1 else 0.0

        ok_high = abs(high_dur - expected_each) <= tol_frac * expected_each
        ok_low  = abs(low_dur  - expected_each) <= tol_frac * expected_each
        if not (ok_high or ok_low):
            i += 2
            continue

        total_len = high_vals.size + low_vals.size
        if total_len < 2:
            i += 2
            continue

        combined_vals = np.concatenate([high_vals, low_vals])
        combined_times = np.concatenate([times[high_s:high_e], times[low_s:low_e]])
        high_mean = float(np.mean(high_vals)) if high_vals.size>0 else float(np.max(combined_vals))
        low_mean  = float(np.mean(low_vals))  if low_vals.size>0 else float(np.min(combined_vals))
        mid = 0.5*(high_mean + low_mean)
        diff = high_mean - low_mean

        segs.append({
            'times': combined_times,
            'values': combined_vals,
            'high_vals': high_vals,
            'low_vals': low_vals,
            'high_mean': high_mean,
            'low_mean': low_mean,
            'mid': float(mid),
            'diff': float(diff)
        })
        i += 2

    return segs

def synthesize_extension_from_samples(processed_df: pd.DataFrame,
                                      ext_cycles_df: pd.DataFrame,
                                      sample_segments: List[Dict[str, Any]],
                                      last_time: Optional[float] = None,
                                      interval: Optional[float] = None,
                                      cycle_hours: float = 8.0,
                                      rng_seed: Optional[int] = None,
                                      stretch_limit: float = 0.25,
                                      residual_scale: float = 1.0,
                                      start_phase: Optional[str] = None,
                                      cap_k: float = 10.0,
                                      abs_cap_mult: float = 20.0,
                                      strict_truncate: bool = False) -> pd.DataFrame:
    if ext_cycles_df is None or len(ext_cycles_df) == 0 or len(sample_segments) == 0:
        return pd.DataFrame(columns=['Time', 'Voltage', 'is_extended'])

    if last_time is None:
        last_time = float(processed_df['Time'].iloc[-1])
    if interval is None:
        times = processed_df['Time'].values
        interval = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        if interval <= 0:
            interval = 1.0

    rng = np.random.default_rng(rng_seed)
    expected_each = cycle_hours / 2.0
    desired_len_full = max(1, int(round(cycle_hours / interval)))

    vals = _to_array(processed_df['Voltage_interpolated'].values)
    last_val = float(vals[-1])
    inferred_phase = 'high'
    elapsed = 0.0
    km = _fit_kmeans_if_needed(vals)
    if km is not None:
        try:
            centers = km.cluster_centers_.ravel()
            high_label = int(np.argmax(centers))
            last_lab = int(km.predict([[last_val]])[0])
            inferred_phase = 'high' if last_lab == high_label else 'low'
            labels = km.predict(vals.reshape(-1,1))
            idx = len(labels) - 1
            while idx > 0 and labels[idx-1] == labels[idx]:
                idx -= 1
            start_of_run_time = float(processed_df['Time'].iat[idx])
            elapsed = max(0.0, last_time - start_of_run_time + interval)
        except Exception:
            inferred_phase = 'high' if last_val >= float(np.median(vals)) else 'low'
            elapsed = 0.0
    else:
        inferred_phase = 'high' if last_val >= float(np.median(vals)) else 'low'
        elapsed = 0.0

    start_phase_used = str(start_phase) if start_phase is not None else inferred_phase
    remaining = max(0.0, expected_each - elapsed)
    remaining = min(remaining, expected_each)
    first_cycle_duration = remaining + expected_each
    first_cycle_n = max(1, int(round(first_cycle_duration / interval)))

    global_diff_scale = max(1e-6, float(np.std(vals)))

    all_times: List[float] = []
    all_values: List[float] = []

    current_start = last_time

    def _clip_and_shift_local(resampled, sample_mid, sample_diff, target_high, target_low):
        # 内部限幅实现，行为与之前的 helper 保持一致
        target_diff = abs(float(target_high - target_low))
        if abs(sample_diff) < 1e-8:
            max_dev_base = max(0.5 * target_diff, 2.0 * global_diff_scale)
        else:
            max_dev_base = max(cap_k * (abs(sample_diff) * 0.25), 0.4 * target_diff, 2.0 * global_diff_scale)
        dev = resampled - sample_mid
        dev_clipped = np.clip(dev, -max_dev_base, max_dev_base)
        if abs(sample_diff) < 1e-8:
            scale = 1.0
        else:
            raw_scale = (float(target_high) - float(target_low)) / sample_diff
            scale = float(np.clip(raw_scale, 1.0 - stretch_limit, 1.0 + stretch_limit))
        new_vals = sample_mid + dev_clipped * scale
        target_mid = 0.5 * (float(target_high) + float(target_low))
        new_vals = new_vals - sample_mid + target_mid
        abs_cap = max(0.6 * target_diff if target_diff > 0 else 0.0, abs_cap_mult * float(max(1e-6, global_diff_scale)))
        return np.clip(new_vals, target_mid - abs_cap, target_mid + abs_cap)

    # 循环生成逐样本序列
    for cycle_idx, row in enumerate(ext_cycles_df.itertuples(index=False)):
        if cycle_idx == 0:
            current_is_high = (start_phase_used == 'high')
            if strict_truncate:
                n_current = max(0, int(round(remaining / interval)))
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
                if n_points == 0:
                    n_points = max(1, first_cycle_n)
                    high_len_target = max(1, high_len_target)
                    low_len_target = n_points - high_len_target
            else:
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
            n_points = desired_len_full
            seg = sample_segments[rng.integers(0, len(sample_segments))]
            high_vals = _to_array(seg['high_vals'])
            low_vals = _to_array(seg['low_vals'])
            total_orig = max(1, high_vals.size + low_vals.size)
            high_len_target = max(1, int(round(n_points * (high_vals.size / total_orig))))
            low_len_target = max(0, n_points - high_len_target)
            use_high_first_flag = True

        if cycle_idx == 0:
            seg = sample_segments[rng.integers(0, len(sample_segments))]
            high_vals = _to_array(seg['high_vals'])
            low_vals = _to_array(seg['low_vals'])

        res_high = _resample_arr(high_vals, high_len_target)
        res_low = _resample_arr(low_vals, low_len_target)

        if use_high_first_flag:
            resampled = np.concatenate([res_high, res_low]) if (res_high.size + res_low.size) > 0 else np.array([])
        else:
            resampled = np.concatenate([res_low, res_high]) if (res_high.size + res_low.size) > 0 else np.array([])

        if resampled.size < n_points:
            resampled = np.pad(resampled, (0, n_points - resampled.size), mode='edge')
        elif resampled.size > n_points:
            resampled = resampled[:n_points]

        sample_high_mean = float(seg.get('high_mean', float(np.mean(high_vals)) if high_vals.size>0 else 0.0))
        sample_low_mean = float(seg.get('low_mean',  float(np.mean(low_vals))  if low_vals.size>0 else 0.0))
        sample_mid = 0.5 * (sample_high_mean + sample_low_mean)
        sample_diff = float(seg.get('diff', sample_high_mean - sample_low_mean))

        target_high = float(getattr(row, 'pred_high', sample_high_mean))
        target_low  = float(getattr(row, 'pred_low', sample_low_mean))

        new_vals = _clip_and_shift_local(resampled, sample_mid, sample_diff, target_high, target_low)

        cycle_times = current_start + np.arange(1, n_points + 1) * interval
        all_times.extend(cycle_times.tolist())
        all_values.extend(new_vals.tolist())
        current_start = current_start + n_points * interval

    ext_df_samples = pd.DataFrame({'Time': np.array(all_times), 'Voltage': np.array(all_values), 'is_extended': True})
    ext_df_samples = ext_df_samples.sort_values('Time').reset_index(drop=True)
    return ext_df_samples