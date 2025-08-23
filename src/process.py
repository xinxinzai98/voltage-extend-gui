import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os

def read_data(file_path):
    # 跳过第一行项目名
    df = pd.read_excel(file_path, skiprows=1)
    df.columns = ['Time', 'Voltage']
    return df

def detect_outliers(df, col):
    data = df[col].values
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (data < lower) | (data > upper) | (data == 0)  # 0值也视为异常
    return mask

def process_data(df, outlier_mask):
    df['is_outlier'] = outlier_mask
    df['Voltage_processed'] = df['Voltage'].copy()
    df.loc[outlier_mask, 'Voltage_processed'] = np.nan
    # 插值
    valid = ~df['Voltage_processed'].isna()
    interp = np.interp(df['Time'], df['Time'][valid], df['Voltage_processed'][valid])
    df['Voltage_interpolated'] = interp
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

def extend_data(df, target_hour=2000, ref_hours=300,
                shrink_alpha=0.35, max_growth=1.05, noise_scale=0.4,
                jitter_scale=0.6, phi=0.8, beta=0.35,
                desired_high_at_target=1.9, ramp_weight=0.9,
                smooth_w=3, gamma=0.7, diff_end_multiplier=1.0):
    last_valid_idx = df.index[-1]
    last_valid_time = df.loc[last_valid_idx, 'Time']
    df2 = df.copy()
    df2['is_extended'] = False

    # 基于全表计算采样间隔与每4小时样本数
    full_times = df2['Time'].values
    if len(full_times) < 2:
        print("数据点太少，无法确定间隔。")
        return df2, pd.DataFrame()
    interval = np.mean(np.diff(full_times))
    cycle_len = max(1, int(round(4.0 / interval)))  # 每个 4 小时平台对应样本数
    full_cycle_samples = cycle_len * 2
    cycle_hours = 8.0  # 一个完整高/低周期小时数

    # 参考区间：最近 ref_hours 小时 -> 若样本不足回退到末尾若干条
    start_time = last_valid_time - ref_hours
    cand = df2[df2['Time'] >= start_time].copy()
    if len(cand) < max(10, cycle_len):
        n = min(len(df2), 200)
        cand = df2.iloc[-n:].copy()

    # 裁成整数个 4 小时平台段单位（保证整周期）
    n_cand = len(cand)
    # 先按 4 小时平台单位裁（而非整周期）以便后续识别高低交替
    trim = n_cand % cycle_len
    if trim != 0 and n_cand > cycle_len:
        cand = cand.iloc[trim:].copy()

    ref_df = cand.copy()
    times = ref_df['Time'].values
    voltages = ref_df['Voltage_interpolated'].values
    if len(times) < max(10, cycle_len):
        print("参考数据仍然不足，跳过外延。")
        return df2, pd.DataFrame()

    # 平台识别（KMeans -> high=1, low=0）
    kmeans = KMeans(n_clusters=2, random_state=0).fit(voltages.reshape(-1, 1))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.ravel()
    high_label = int(np.argmax(centers))
    platform = np.where(labels == high_label, 1, 0)
    ref_df = ref_df.reset_index(drop=True)
    ref_df['platform'] = platform

    # 分段（相邻平台变化处切分）
    changes = np.where(np.diff(platform) != 0)[0] + 1
    block_idx = np.concatenate(([0], changes, [len(platform)]))
    blocks = [(platform[s], s, e) for s, e in zip(block_idx[:-1], block_idx[1:])]

    # 严格筛除异常周期块（长度 / 均值 / 波动）
    tol = 0.05  # 周期长度容忍率（±5%）
    min_len = max(1, int(cycle_len * (1 - tol)))
    max_len = max(1, int(cycle_len * (1 + tol)))
    mean_tol = 0.03  # 平台均值允许偏差（可调）
    std_tol = 0.04   # 平台内部波动上限（可调）

    # 先计算整体高低均值用于参考
    grp = ref_df.groupby('platform')['Voltage_interpolated'].mean()
    low_mean = grp.get(0, np.mean(voltages))
    high_mean = grp.get(1, np.mean(voltages))

    clean_blocks = []
    for ptype, s, e in blocks:
        seg = ref_df['Voltage_interpolated'].values[s:e]
        seg_len = e - s
        if not (min_len <= seg_len <= max_len):
            continue
        seg_mean = seg.mean()
        seg_std = seg.std()
        if ptype == 1 and abs(seg_mean - high_mean) < mean_tol and seg_std < std_tol:
            clean_blocks.append((ptype, s, e))
        elif ptype == 0 and abs(seg_mean - low_mean) < mean_tol and seg_std < std_tol:
            clean_blocks.append((ptype, s, e))

    # 如果没有合格块，回退到去掉3*sigma异常点的整体策略
    if len(clean_blocks) == 0:
        med = np.median(voltages)
        sigma = np.std(voltages)
        mask = np.abs(voltages - med) <= (3 * sigma)
        clean_times = times[mask]
        clean_voltages = voltages[mask]
        if len(clean_times) < max(10, cycle_len):
            print("回退后数据仍不足，跳过外延。")
            return df2, pd.DataFrame()
        # 尝试用整段数据估计高低均值序列：简单分成完整周期块
        # 这里退化为用每个完整周期内的高/低均值（若可行）
        # 继续下面的流程使用 clean_* 变量
        # 将 ref_df 替换为干净子集（按mask）
        tmp_df = ref_df.iloc[mask.nonzero()[0]].reset_index(drop=True)
        ref_df = tmp_df
        # 重新构建 blocks（简单切分为固定长度块）
        values = ref_df['Voltage_interpolated'].values
        n_blocks = len(values) // cycle_len
        blocks = []
        for i in range(n_blocks):
            s = i * cycle_len
            e = s + cycle_len
            # label by comparing mean to overall mean
            p = 1 if values[s:e].mean() > values.mean() else 0
            blocks.append((p, s, e))
        clean_blocks = blocks

    # 将清洗后的块按相邻两块配对成完整周期（保证一对包含高与低）
    paired_cycle_times = []
    high_means = []
    low_means = []
    i = 0
    while i + 1 < len(clean_blocks):
        p1, s1, e1 = clean_blocks[i]
        p2, s2, e2 = clean_blocks[i+1]
        # 只接受高/低交替的配对，否则跳过一个块
        if p1 == p2:
            i += 1
            continue
        # 把高作为 high_block，低作为 low_block
        if p1 == 1:
            high_seg = ref_df['Voltage_interpolated'].values[s1:e1]
            low_seg = ref_df['Voltage_interpolated'].values[s2:e2]
            mid_t = (ref_df['Time'].values[s1:e1].mean() + ref_df['Time'].values[s2:e2].mean()) / 2.0
        else:
            high_seg = ref_df['Voltage_interpolated'].values[s2:e2]
            low_seg = ref_df['Voltage_interpolated'].values[s1:e1]
            mid_t = (ref_df['Time'].values[s1:e1].mean() + ref_df['Time'].values[s2:e2].mean()) / 2.0

        high_means.append(np.mean(high_seg))
        low_means.append(np.mean(low_seg))
        paired_cycle_times.append(mid_t)
        i += 2

    paired_cycle_times = np.array(paired_cycle_times)
    high_means = np.array(high_means)
    low_means = np.array(low_means)

    if len(paired_cycle_times) < 2:
        # 样本过少，降级为用整体高低均值预测常数线
        pred_high_const = np.mean(high_means) if len(high_means) > 0 else high_mean
        pred_low_const = np.mean(low_means) if len(low_means) > 0 else low_mean
        use_constant = True
    else:
        use_constant = False
        # 用线性回归分别拟合高/低均值随时间的变化（但斜率用于外推，左端点锚定到最后观测的高/低均值）
        lr_high = LinearRegression().fit(paired_cycle_times.reshape(-1,1), high_means)
        lr_low = LinearRegression().fit(paired_cycle_times.reshape(-1,1), low_means)

    # 为外延生成逐样本时间点
    ext_times = np.arange(last_valid_time + interval, target_hour + interval, interval)
    if len(ext_times) == 0:
        return df2, pd.DataFrame()

    # 计算每个未来周期的中点时间，并预测每周期的 high/low 均值
    last_cycle_mid = paired_cycle_times[-1] if len(paired_cycle_times) > 0 else last_valid_time
    ext_cycle_count = int(np.ceil((ext_times[-1] - last_cycle_mid) / cycle_hours)) + 1
    ext_cycle_mid_times = last_cycle_mid + np.arange(1, ext_cycle_count+1) * cycle_hours

    # 获取最后观测的周期均值（用于锚定线性外推的左端点）
    last_obs_high_ref = high_means[-1] if len(high_means) > 0 else high_mean
    last_obs_low_ref  = low_means[-1]  if len(low_means)  > 0 else low_mean

    if use_constant:
        linear_high = np.full(len(ext_cycle_mid_times), pred_high_const)
        linear_low  = np.full(len(ext_cycle_mid_times), pred_low_const)
    else:
        # 使用回归的斜率，但以最后观测周期均值为基点进行线性外推（左端点为 last_obs_*）
        slope_high = float(lr_high.coef_[0]) if hasattr(lr_high, 'coef_') else 0.0
        slope_low  = float(lr_low.coef_[0])  if hasattr(lr_low, 'coef_')  else 0.0
        linear_high = last_obs_high_ref + slope_high * (ext_cycle_mid_times - last_cycle_mid)
        linear_low  = last_obs_low_ref  + slope_low  * (ext_cycle_mid_times - last_cycle_mid)

    # 增加低频非线性分量：用二次多项式拟合历史 high_means/low_means（若样本不足降阶）
    def poly_fit_predict(x, y, x_pred, max_deg=2):
        deg = min(max_deg, max(1, len(x)-1))
        coeff = np.polyfit(x, y, deg)
        return np.poly1d(coeff)(x_pred)

    if len(paired_cycle_times) >= 3:
        poly_high_pred = poly_fit_predict(paired_cycle_times, high_means, ext_cycle_mid_times, max_deg=2)
        poly_low_pred  = poly_fit_predict(paired_cycle_times, low_means,  ext_cycle_mid_times, max_deg=2)
    else:
        poly_high_pred = linear_high.copy()
        poly_low_pred  = linear_low.copy()

    # --- 保持 mid（高低中位）允许低频非线性变化，但 diff（高-低）由历史 diff 序列建模以保持同步变化 ---
    # mid 由 linear 与 poly 混合（保持之前设计）
    beta = 0.35
    mid_linear = 0.5 * (linear_high + linear_low)
    mid_poly   = 0.5 * (poly_high_pred + poly_low_pred)
    beta_mid = beta
    pred_mid_per_cycle = (1 - beta_mid) * mid_linear + beta_mid * mid_poly

    # 建模历史 diff = high_means - low_means（若样本足够则拟合线性趋势，否则取常数）
    hist_diff = high_means - low_means if (len(high_means) == len(low_means) and len(high_means) > 0) else np.array([])
    if len(hist_diff) >= 2:
        lr_diff = LinearRegression().fit(paired_cycle_times.reshape(-1,1), hist_diff)
        diff_pred = lr_diff.predict(ext_cycle_mid_times.reshape(-1,1))
    else:
        last_diff_obs = (high_means[-1] - low_means[-1]) if len(high_means)>0 and len(low_means)>0 else (high_mean - low_mean)
        diff_pred = np.full(len(ext_cycle_mid_times), last_diff_obs)

    # 初始 diff 由 poly/linear 高低差计算（作为参考），再与 diff_pred 混合以保持历史规律
    ref_diff = (poly_high_pred - poly_low_pred) * beta + (linear_high - linear_low) * (1 - beta)
    # 混合比例 gamma 控制使用历史 diff_pred 的权重（越大越遵循历史 diff 变化）
    gamma = 0.7
    pred_diff_per_cycle = (1 - gamma) * ref_diff + gamma * diff_pred

    # --- 保证预测 diff 不小于历史倒数 N 个周期的平均 diff（按绝对值比较） ---
    N_recent = min(10, len(high_means))
    if N_recent > 0:
        recent_diffs = np.abs(high_means[-N_recent:] - low_means[-N_recent:])
        recent_avg_diff = float(np.mean(recent_diffs))
        # 对每个周期确保振幅不被缩小到低于 recent_avg_diff
        sign_vec = np.sign(pred_diff_per_cycle)
        abs_pred = np.abs(pred_diff_per_cycle)
        abs_adjusted = np.maximum(abs_pred, recent_avg_diff)
        pred_diff_per_cycle = sign_vec * abs_adjusted
    # 如果没有历史可参考，则保持原有 pred_diff_per_cycle

    # --- 继续后续处理：收缩、限幅、重建高低等（保持原逻辑） ---
    # --- 定义最近观测的高/低值（必须在后续使用前定义） ---
    last_obs_high = high_means[-1] if len(high_means) > 0 else high_mean
    last_obs_low  = low_means[-1]  if len(low_means)  > 0 else low_mean

    # 保守收缩：把 mid / diff 都拉向最后观测值（防止突变）
    last_obs_mid = 0.5 * (last_obs_high + last_obs_low)
    shrink_alpha_mid = 0.35
    pred_mid_per_cycle = last_obs_mid + shrink_alpha_mid * (pred_mid_per_cycle - last_obs_mid)

    last_diff = max(1e-6, (last_obs_high - last_obs_low))
    max_growth = 1.05
    allowed = last_diff * max_growth
    # 对 diff 做限幅，防止过大变化
    scale = np.minimum(1.0, np.abs(allowed / (pred_diff_per_cycle + 1e-12)))
    pred_diff_per_cycle = pred_diff_per_cycle * scale

    # 最终重建高低值，保证同步变化（mid ± diff/2）
    pred_high_per_cycle = pred_mid_per_cycle + 0.5 * pred_diff_per_cycle
    pred_low_per_cycle  = pred_mid_per_cycle - 0.5 * pred_diff_per_cycle

    # --- 收缩与差值上限处理（保持保守） ---
    last_obs_high = high_means[-1] if len(high_means) > 0 else high_mean
    last_obs_low  = low_means[-1]  if len(low_means) > 0  else low_mean
    shrink_alpha = 0.35
    pred_high_per_cycle = last_obs_high + shrink_alpha * (pred_high_per_cycle - last_obs_high)
    pred_low_per_cycle  = last_obs_low  + shrink_alpha * (pred_low_per_cycle  - last_obs_low)

    last_diff = max(1e-6, (last_obs_high - last_obs_low))
    max_growth = 1.05
    allowed = last_diff * max_growth
    cur_diff = pred_high_per_cycle - pred_low_per_cycle
    scale = np.minimum(1.0, np.abs(allowed / (cur_diff + 1e-12)))
    pred_mid = (pred_high_per_cycle + pred_low_per_cycle) / 2.0
    adjusted_half = 0.5 * np.abs(cur_diff) * scale
    pred_high_per_cycle = pred_mid + adjusted_half
    pred_low_per_cycle  = pred_mid - adjusted_half

    # --- 轻微提升高低曲线斜率（按周期平滑 ramp，加到 mid 上，使在 target 时高值接近目标） ---
    try:
        # 将目标高位从 2.0 调为 1.9，使 2000h 附近高值更保守
        desired_high_at_target = 1.9
        if len(pred_high_per_cycle) > 0 and 'ext_cycle_mid_times' in locals():
            current_end = pred_high_per_cycle[-1]
            delta = desired_high_at_target - current_end
            span = ext_cycle_mid_times[-1] - ext_cycle_mid_times[0] if len(ext_cycle_mid_times) > 1 else 1.0
            if span == 0:
                span = 1.0
            # 保持大部分差值来自原预测，仅注入一部分 ramp 来拉高末端（比例可调）
            ramp_weight = 0.9
            ramp = (ext_cycle_mid_times - ext_cycle_mid_times[0]) / span * (delta * ramp_weight)
            # 为保持高低差同步，同样把 ramp 加到低位（不会改变 diff 形状）
            pred_high_per_cycle = pred_high_per_cycle + ramp
            pred_low_per_cycle  = pred_low_per_cycle  + ramp
    except Exception:
        pass

    # --- 从参考区间学习残差池（保证 high_resid / low_resid 在后续使用前存在） ---
    high_resid = np.array([])
    low_resid = np.array([])
    if 'clean_blocks' in locals() and len(clean_blocks) > 0:
        for ptype, s, e in clean_blocks:
            seg = ref_df['Voltage_interpolated'].values[s:e]
            if len(seg) == 0:
                continue
            if ptype == 1:
                high_resid = np.concatenate((high_resid, seg - seg.mean()))
            else:
                low_resid = np.concatenate((low_resid, seg - seg.mean()))

    # 回退策略：若分块残差不足，尝试用平台内所有点的偏差；仍不足则用小幅正态噪声填充
    if high_resid.size == 0:
        if 'platform' in ref_df.columns and np.any(ref_df['platform'] == 1):
            vals = ref_df.loc[ref_df['platform'] == 1, 'Voltage_interpolated'].values
            if vals.size > 0:
                high_resid = vals - np.mean(vals)
        if high_resid.size == 0:
            high_resid = np.random.normal(0, max(1e-4, np.std(voltages) * 0.01), size=500)

    if low_resid.size == 0:
        if 'platform' in ref_df.columns and np.any(ref_df['platform'] == 0):
            vals = ref_df.loc[ref_df['platform'] == 0, 'Voltage_interpolated'].values
            if vals.size > 0:
                low_resid = vals - np.mean(vals)
        if low_resid.size == 0:
            low_resid = np.random.normal(0, max(1e-4, np.std(voltages) * 0.01), size=500)

    # 去极端值并保证为 numpy 数组
    low_h, up_h = np.percentile(high_resid, [1, 99])
    high_resid = np.clip(np.asarray(high_resid), low_h, up_h)
    low_l, up_l = np.percentile(low_resid, [1, 99])
    low_resid = np.clip(np.asarray(low_resid), low_l, up_l)

    # 周期级抖动：用 AR(1) 生成平滑的序列（比 iid 噪声更有连贯性）
    hist_std_high = np.std(high_means) if len(high_means) > 1 else max(1e-3, np.std(voltages)*0.01)
    hist_std_low  = np.std(low_means)  if len(low_means)  > 1 else max(1e-3, np.std(voltages)*0.01)
    # 增大周期级抖动幅度，使高/低曲线在百小时尺度上能看到更明显波动（靠近原始噪声）
    jitter_scale = 0.6   # 从 0.35 -> 0.6（可调）
    phi = 0.8  # AR(1) 系数，越接近1越平滑
    n_cycles = len(pred_high_per_cycle)
    def gen_ar1(n, sigma, phi):
        if n <= 0: return np.array([])
        eps_sd = sigma * np.sqrt(1 - phi**2)
        arr = np.zeros(n)
        arr[0] = np.random.normal(0, sigma)
        for t in range(1, n):
            arr[t] = phi * arr[t-1] + np.random.normal(0, eps_sd)
        return arr

    cycle_jitter_high = gen_ar1(n_cycles, hist_std_high * jitter_scale, phi)
    cycle_jitter_low  = gen_ar1(n_cycles, hist_std_low  * jitter_scale, phi)

    pred_high_per_cycle = pred_high_per_cycle + cycle_jitter_high
    pred_low_per_cycle  = pred_low_per_cycle  + cycle_jitter_low

    # 生成逐样本外延：每段噪声由参考残差 bootstrap 并做短窗口平滑，保留“自然纹理”
    ext_data = []
    ext_time_list = []
    t_ptr = last_valid_time
    cycle_idx = 0

    # 计算最后点的平台，保证先接入与最后点不同的平台（交替）
    last_val = df2['Voltage_interpolated'].values[-1]
    last_label = kmeans.predict([[last_val]])[0]
    last_platform = 1 if last_label == high_label else 0
    start_platform = 1 - last_platform

    # 平滑窗口（odd），用于对采样残差做轻度平滑以产生更自然噪声
    smooth_w = 3
    if smooth_w % 2 == 0:
        smooth_w += 1

    # 生成按周期样本，直到覆盖 ext_times
    while t_ptr < ext_times[-1]:
        for seg_ptype in (start_platform, 1 - start_platform):
            # 获取本周期对应的高/低基值
            if cycle_idx >= len(pred_high_per_cycle):
                ph = pred_high_per_cycle[-1] if len(pred_high_per_cycle)>0 else high_mean
                pl = pred_low_per_cycle[-1]  if len(pred_low_per_cycle)>0  else low_mean
            else:
                ph = pred_high_per_cycle[cycle_idx]
                pl = pred_low_per_cycle[cycle_idx]

            seg_times = np.arange(t_ptr + interval, t_ptr + (cycle_len + 1) * interval, interval)[:cycle_len]
            if len(seg_times) == 0:
                continue

            # sample residuals from empirical pool and apply light smoothing
            if seg_ptype == 1:
                pool = high_resid
            else:
                pool = low_resid

            # bootstrap residuals
            sampled = np.random.choice(pool, size=len(seg_times), replace=True)
            # 轻度平滑（如果序列较短则跳过大窗口）
            if len(sampled) >= smooth_w:
                kernel = np.ones(smooth_w) / smooth_w
                sampled = np.convolve(sampled, kernel, mode='same')
            # 限制单点极端值（避免异常）——按历史残差的上下百分位截断
            lower_p, upper_p = np.percentile(pool, [1, 99])
            sampled = np.clip(sampled, lower_p, upper_p)

            base_val = ph if seg_ptype == 1 else pl
            values = base_val + sampled

            ext_time_list.extend(seg_times)
            ext_data.extend(values)
            t_ptr = seg_times[-1]
        cycle_idx += 1
    # end while

    ext_times_full = np.array(ext_time_list)
    ext_data_full = np.array(ext_data)

    # 截断到目标时间范围
    mask = ext_times_full <= ext_times[-1]
    ext_times_full = ext_times_full[mask]
    ext_data_full = ext_data_full[mask]

    # 确保首个外延点与最后原始点平滑衔接
    if len(ext_data_full) > 0:
        ext_data_full[0] = last_val

    ext_df = pd.DataFrame({'Time': ext_times_full, 'Voltage': ext_data_full, 'is_extended': True})

    # final_df 保留原始加外延
    final_df = pd.concat([df2, ext_df], ignore_index=True)
    return final_df, ext_df

def save_results(processed_df, final_df, file_path):
    base = os.path.splitext(file_path)[0]
    processed_df[['Time','Voltage','is_outlier','Voltage_interpolated']].to_excel(f"{base}_processed.xlsx", index=False)
    final_df[['Time','Voltage','is_extended']].to_excel(f"{base}_final.xlsx", index=False)
    print(f"处理数据已保存到: {base}_processed.xlsx")
    print(f"外延数据已保存到: {base}_final.xlsx")

def visualize(processed_df, final_df, ext_df):
    plt.figure(figsize=(16,10))
    valid = processed_df[~processed_df['is_outlier']]
    plt.plot(valid['Time'], valid['Voltage'], 'bo-', label='Original Data', alpha=0.6, markersize=2)
    plt.plot(valid['Time'], valid['Voltage_interpolated'], 'g-', label='Interpolated Data', alpha=0.8)
    if len(ext_df) > 0:
        plt.plot(ext_df['Time'], ext_df['Voltage'], 'r-', label='Extended Data', alpha=0.9)
    plt.xlabel('Time (hours)')
    plt.ylabel('Voltage')
    plt.ylim(1.5, 2)  # 固定y轴范围为1.5~2
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    file_path = "data-layang_clean.xlsx"
    df = read_data(file_path)
    print(f"数据范围: {df['Time'].min()} ~ {df['Time'].max()} 小时")
    outlier_mask = detect_outliers(df, 'Voltage')
    processed_df = process_data(df, outlier_mask)
    final_df, ext_df = extend_data(processed_df, target_hour=2000)
    visualize(processed_df, final_df, ext_df)
    save_results(processed_df, final_df, file_path)
    print("处理完成！")
    print(f"原始数据点数: {len(df)}")
    print(f"异常值数量: {outlier_mask.sum()}")
    print(f"外延数据点数: {len(ext_df)}")
    print(f"最终数据点数: {len(final_df)}")

if __name__ == "__main__":
    main()