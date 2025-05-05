# -*- coding: utf-8 -*-
"""
完整的PPG波形特征提取与特征选择模块。
功能:
1. 从原始PPG信号提取75维形态学特征 (基于 Ma等人2023年《KD-Informer》论文).
2. 使用修改的ChiMerge算法(算法1)对连续特征进行离散化.
3. 实现BEFS特征选择方法(算法2), 包括包装法(Wrapper)和嵌入法(Embedded)两阶段以及后续的逐步后向消除(Backward Elimination).
依赖项: numpy, scipy, pywt, antropy, nolds, pandas, sklearn
"""

import numpy as np
from numpy import trapz
from scipy import signal, stats
import pywt
from antropy import sample_entropy, spectral_entropy, detrended_fluctuation
import nolds
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE

# ================== 数据加载与分段 ==================
def load_ppg_abp_from_folder(folder_path):
    """从指定文件夹中读取.mat文件的PPG和ABP数据。
    返回一个记录列表，每个元素为{'ppg': ppg_array, 'abp': abp_array}。"""
    from scipy.io import loadmat
    import os
    records = []
    # 列出文件夹中所有.mat文件，按照文件名中的数字顺序排序
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')],
                       key=lambda x: int(os.path.splitext(x)[0]) if x[:-4].isdigit() else x)
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        data = loadmat(file_path)
        if 'data' in data:
            # 如果.mat是一个结构体 'data' 包含 ppg 和 abp 字段
            ppg = data['data']['ppg'][0,0].squeeze()
            abp = data['data']['abp'][0,0].squeeze()
        else:
            # 如果.mat直接包含 ppg 和 abp 键
            ppg = data.get('ppg', None)
            abp = data.get('abp', None)
            if ppg is not None: ppg = ppg.squeeze()
            if abp is not None: abp = abp.squeeze()
        if ppg is None or abp is None:
            continue
        records.append({'ppg': ppg, 'abp': abp})
    return records

def segment_signal(ppg, abp, window_size, step_size):
    """将PPG和ABP序列按滑动窗口切分成多段。
    返回一个列表，每个元素是{'ppg_segment': ppg_seg, 'abp_segment': abp_seg}。
    参数:
        ppg: 1D numpy数组, 原始PPG信号
        abp: 1D numpy数组, 原始ABP信号(需与PPG长度一致)
        window_size: 窗口长度(采样点数)
        step_size: 滑动步长(采样点数)
    """
    n = len(ppg)
    samples = []
    start = 0
    while start + window_size <= n:
        end = start + window_size
        ppg_seg = ppg[start:end]
        abp_seg = abp[start:end]
        samples.append({'ppg_segment': ppg_seg, 'abp_segment': abp_seg})
        start += step_size
    return samples

# ================== 75维特征提取 ==================
def _band_power(sig, fs, fmin, fmax):
    """计算信号在频带 [fmin, fmax] 内的带宽功率 (Welch方法)。"""
    # 使用Welch功率谱密度估计计算频带功率
    nperseg = min(int(4 * fs), len(sig)) if len(sig) > 0 else int(4 * fs)
    if nperseg < 1:  # 确保nperseg为正整数
        nperseg = len(sig)
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
    # 掩码选取频率范围内的功率谱
    mask = (freqs >= fmin) & (freqs <= fmax)
    # 计算频带内功率 (积分PSD)
    return np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0

def _peak_cycle(ppg, fs):
    """从PPG波形中定位首个完整周期的谷-峰-谷点索引 (返回 (v0, p0, v1))。
    如果无法找到完整周期则返回 None。"""
    # 寻找峰和谷（取负值的峰作为谷）
    peaks, _ = signal.find_peaks(ppg, distance=0.3 * fs)
    troughs, _ = signal.find_peaks(-ppg, distance=0.3 * fs)
    if len(peaks) < 1 or len(troughs) < 2:
        return None
    # 第一个峰p0，以及在它之前的最后一个谷v0，之后的第一个谷v1
    p0 = peaks[0]
    v0_candidates = troughs[troughs < p0]
    if len(v0_candidates) == 0:
        return None
    v0 = v0_candidates.max()
    v1_candidates = troughs[troughs > p0]
    if len(v1_candidates) == 0:
        return None
    v1 = v1_candidates.min()
    return (v0, p0, v1) if v1 > v0 else None

def extract_ppg_features_75(ppg, fs=125):
    """
    提取PPG信号的75维形态学特征。
    输入:
        ppg: 1D numpy数组, PPG信号
        fs: 采样率 (Hz), 默认为125Hz
    返回:
        feat: 字典, 包含75个特征名和对应值
    """
    x = np.asarray(ppg, dtype=float)
    L = len(x)
    if L < fs * 2:
        raise ValueError("信号长度过短，不足2秒")
    feat = {}
    # 1. 统计/时域特征
    feat["Mean"]        = x.mean()
    feat["Median"]      = np.median(x)
    feat["Std"]         = x.std(ddof=1)
    feat["Var"]         = x.var(ddof=1)
    feat["IQR"]         = stats.iqr(x)
    feat["RMS"]         = np.sqrt(np.mean(x**2))
    feat["Skew"]        = stats.skew(x, bias=False)
    feat["Kurtosis"]    = stats.kurtosis(x, bias=False)
    feat["Min"]         = x.min()
    feat["Max"]         = x.max()
    feat["PP_amp"]      = feat["Max"] - feat["Min"]          # 峰-谷幅值
    feat["Sum"]         = x.sum()
    feat["AbsSum"]      = np.abs(x).sum()
    feat["SqSum"]       = (x**2).sum()
    feat["MeanAbsDiff"] = np.mean(np.abs(np.diff(x)))
    feat["ZeroCross"]   = np.sum((x[:-1] * x[1:]) < 0)       # 过零次数
    slope = np.gradient(x)
    feat["Slope_Mean"]  = slope.mean()
    feat["Slope_Std"]   = slope.std(ddof=1)
    # Shannon熵 (基于直方图密度)
    counts, _ = np.histogram(x, bins='auto', density=True)
    counts = counts[counts > 0]
    feat["ShannonEnt"]  = -np.sum(counts * np.log(counts))
    # Fisher偏度 (无偏估计)
    m = feat["Mean"]
    feat["x_skew"]      = np.sum((x - m)**3) / L / ((np.sum((x - m)**2) / L)**1.5) if np.sum((x - m)**2) != 0 else np.nan

    # 2. 周期相关特征
    cycle = _peak_cycle(x, fs)
    cycle_keys = ["MF","K_area","T_up","T_down","PW",
                  "Area_Sys","Area_Dia","Slope_Ratio","Symmetry_Idx","Modulation_Idx"]
    if cycle is None:
        for k in cycle_keys:
            feat[k] = np.nan
    else:
        v0, p0, v1 = cycle
        if v1 <= v0:
            for k in cycle_keys:
                feat[k] = np.nan
        else:
            feat["MF"] = x[p0] / (np.mean(x**2) + 1e-8)
            seg = x[v0:v1]
            if seg.size > 0:
                feat["K_area"] = (seg.mean() - seg.min()) / (seg.max() - seg.min() + 1e-8)
            else:
                feat["K_area"] = np.nan
            feat["T_up"]   = (p0 - v0) / fs               # 上升时间(s)
            feat["T_down"] = (v1 - p0) / fs               # 下降时间(s)
            feat["PW"]     = (v1 - v0) / fs               # 脉搏波周期(s)
            feat["Area_Sys"] = trapz(x[v0:p0], dx=1/fs) if p0 > v0 else np.nan
            feat["Area_Dia"] = trapz(x[p0:v1], dx=1/fs) if v1 > p0 else np.nan
            up_slope = (x[p0] - x[v0]) / (feat["T_up"] + 1e-8)
            down_slope = (x[p0] - x[v1]) / (feat["T_down"] + 1e-8)
            feat["Slope_Ratio"] = up_slope / (down_slope + 1e-8)
            feat["Symmetry_Idx"] = feat["T_up"] / (feat["T_down"] + 1e-8)
            feat["Modulation_Idx"] = (x[p0] - x.min()) / (x[p0] + x.min() + 1e-8)

    # 3. APG特征 (PPG的二阶差分峰谷)
    d2 = np.gradient(np.gradient(x))
    a_peaks, _ = signal.find_peaks(d2, distance=0.2 * fs)      # APG波峰
    a_troughs, _ = signal.find_peaks(-d2, distance=0.2 * fs)   # APG波谷
    def _pick(arr, idx):
        return arr[idx] if len(arr) > idx else np.nan
    for idx, name in enumerate(["a","b","c","d","e"]):
        if name in ["a", "b", "e"]:
            pos = _pick(a_peaks, idx)
        else:
            pos = _pick(a_troughs, idx-2)
        feat[f"APG_{name}"] = d2[pos] if not np.isnan(pos) else np.nan
    a_val = feat["APG_a"]
    for name in ["b","c","d","e"]:
        cur = feat[f"APG_{name}"]
        if a_val is None or (isinstance(a_val, float) and np.isnan(a_val)) or a_val == 0 or (isinstance(cur, float) and np.isnan(cur)):
            feat[f"APG_{name}/a"] = np.nan
        else:
            feat[f"APG_{name}/a"] = cur / a_val

    # 4. 频域特征
    feat["P_total"] = _band_power(x, fs, 0, fs/2)      # 总功率 (0 ~ Nyquist)
    feat["P_ULF"]   = _band_power(x, fs, 0, 0.003)     # 超低频功率
    feat["P_VLF"]   = _band_power(x, fs, 0.003, 0.04)  # 极低频功率
    feat["P_LF"]    = _band_power(x, fs, 0.04, 0.15)   # 低频功率
    feat["P_HF"]    = _band_power(x, fs, 0.15, 0.4)    # 高频功率
    feat["LF_HF"]   = feat["P_LF"] / (feat["P_HF"] + 1e-8)
    nperseg_welch = min(int(4 * fs), L)
    if nperseg_welch < 1:
        nperseg_welch = L
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg_welch)
    psd_sum = psd.sum() + 1e-12
    feat["SpecCentroid"]  = (freqs * psd).sum() / psd_sum
    feat["SpecBandwidth"] = np.sqrt(((freqs - feat["SpecCentroid"])**2 * psd).sum() / psd_sum)
    feat["SpecEntropy"]   = spectral_entropy(x, fs, method='welch', normalize=False)
    cum_power = np.cumsum(psd) / psd_sum
    feat["SpecRolloff"]   = freqs[np.searchsorted(cum_power, 0.95)]
    mask = (freqs[1:] > 0) & (psd[1:] > 0)
    feat["SpecSlope"] = stats.linregress(np.log(freqs[1:][mask]), np.log(psd[1:][mask]))[0] if np.any(mask) else np.nan
    feat["SpecFlatness"]  = stats.gmean(psd + 1e-12) / (psd_sum / len(psd))
    sp_peaks, _ = signal.find_peaks(psd)
    feat["SpecPeakNum"] = len(sp_peaks)
    nperseg_stft = min(int(fs), L) if L > 0 else int(fs)
    if nperseg_stft < 1:
        nperseg_stft = L
    f_stft, t_stft, Zxx = signal.stft(x, fs=fs, nperseg=nperseg_stft)
    E = np.abs(Zxx) ** 2
    feat["STFT_Energy_Mean"] = E.mean()
    feat["STFT_Energy_Var"]  = E.var(ddof=1)
    coeffs = pywt.wavedec(x, 'db4', level=3)
    for i, c in enumerate(coeffs[1:], start=1):
        feat[f"Wavelet_D{i}"] = np.sum(c**2)
    feat["Wavelet_A3"] = np.sum(coeffs[0]**2)

    # 5. 非线性/脉搏间期(RR)特征
    feat["ApEn"]    = sample_entropy(x)   # 近似熵 (用样本熵近似)
    feat["SampEn"]  = sample_entropy(x)   # 样本熵
    feat["RenyiEn"] = -np.log((counts**2).sum() + 1e-12)  # Rényi熵(二阶)
    feat["Hurst"]   = nolds.hurst_rs(x)
    feat["DFA_alpha"] = detrended_fluctuation(x)
    feat["Lyap"]    = nolds.lyap_r(x, emb_dim=10, lag=1, min_tsep=int(0.1*fs))
    feat["CorrDim"] = nolds.corr_dim(x, emb_dim=10, lag=1)
    peaks, _ = signal.find_peaks(x, distance=0.3 * fs)
    rr = np.diff(peaks) / fs
    if len(rr) < 4:
        for k in ["RR_mean","RR_SDNN","RR_RMSSD","pNN50","SD1","SD2","SD1_SD2","RR_LF","RR_HF","RR_LF_HF"]:
            feat[k] = np.nan
    else:
        feat["RR_mean"]  = rr.mean()
        feat["RR_SDNN"]  = rr.std(ddof=1)
        feat["RR_RMSSD"] = np.sqrt(np.mean(np.diff(rr) ** 2))
        feat["pNN50"]    = np.mean(np.abs(np.diff(rr)) > 0.05)
        sd1 = np.sqrt(0.5) * np.std(rr[1:] - rr[:-1], ddof=1)
        sd2 = np.sqrt(2 * (rr.std(ddof=1) ** 2) - sd1**2)
        feat["SD1"] = sd1
        feat["SD2"] = sd2
        feat["SD1_SD2"] = sd1 / (sd2 + 1e-8)
        d_rr = np.diff(rr)
        mean_diff = np.mean(d_rr)
        if mean_diff <= 0:
            feat["RR_LF"] = 0.0
            feat["RR_HF"] = 0.0
        else:
            fs_rr = 1.0 / mean_diff
            feat["RR_LF"] = _band_power(rr - rr.mean(), fs_rr, 0.04, 0.15)
            feat["RR_HF"] = _band_power(rr - rr.mean(), fs_rr, 0.15, 0.4)
        feat["RR_LF_HF"] = feat["RR_LF"] / (feat["RR_HF"] + 1e-8)
    # 确认特征数目
    assert len(feat) == 75, f"提取到{len(feat)}个特征, 应为75个"
    return feat

# ================== 特征离散化 (ChiMerge) ==================
def modified_chi_merge(values, labels, k=10, alpha=0.05):
    """对单个连续特征数组values执行ChiMerge离散化，返回最佳分箱切割点列表。"""
    values = np.asarray(values)
    labels = np.asarray(labels)
    # 1. 初始等宽分箱: 将特征值等距划分为k个区间
    bins = np.linspace(values.min(), values.max(), k + 1)
    # 确保包含边界
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    # intervals表示每个值所属的区间索引 (0 ~ k-1)
    intervals = np.digitize(values, bins) - 1
    # 2. 迭代合并相邻区间
    def compute_chi2_for_adjacent(intervals, labels, idx):
        # 计算当前区间idx和idx+1的卡方统计量 (2行 x m列, m为类别数)
        mask = (intervals == idx) | (intervals == idx + 1)
        if not np.any(mask):
            return np.inf
        sub_intervals = intervals[mask]
        sub_labels = labels[mask]
        # 构造2行类别频数表obs: 第0行对应区间idx, 第1行对应区间idx+1
        classes = np.unique(sub_labels)
        obs = np.zeros((2, len(classes)), dtype=int)
        for j, cls in enumerate(classes):
            obs[0, j] = np.sum((sub_intervals == idx)   & (sub_labels == cls))
            obs[1, j] = np.sum((sub_intervals == idx+1) & (sub_labels == cls))
        # 期望频数
        row_sum = obs.sum(axis=1, keepdims=True)
        col_sum = obs.sum(axis=0, keepdims=True)
        total = obs.sum()
        if total == 0:
            return np.inf
        expected = row_sum * col_sum / total
        # 卡方值计算 (添加极小值避免除以0)
        chi2 = np.sum((obs - expected) ** 2 / (expected + 1e-8))
        return chi2

    # 不断合并直到没有相邻区间可以合并
    while True:
        unique_intervals = np.unique(intervals)
        if len(unique_intervals) <= 1:
            break
        chi2_values = []
        for idx in range(int(unique_intervals.min()), int(unique_intervals.max())):
            if idx in unique_intervals and (idx + 1) in unique_intervals:
                chi2 = compute_chi2_for_adjacent(intervals, labels, idx)
                chi2_values.append((chi2, idx))
        if len(chi2_values) == 0:
            break
        chi2_values.sort(key=lambda x: x[0])
        min_chi2, merge_idx = chi2_values[0]
        if min_chi2 > alpha:
            break
        # 合并区间: 将interval = merge_idx+1的合并到merge_idx
        intervals[intervals == merge_idx + 1] = merge_idx
        intervals[intervals > merge_idx + 1] -= 1
        # 降低显著性水平 (逐步收紧条件)
        alpha *= 0.9

    # 3. 生成最终切割点列表 (每个合并区间的上边界)
    cut_points = []
    final_intervals = np.unique(intervals)
    final_intervals.sort()
    for iv in final_intervals[:-1]:
        max_val = values[intervals == iv].max()
        cut_points.append(max_val)
    cut_points = np.sort(np.array(cut_points))
    return cut_points

def discretize_features(features, labels, k=10, alpha=0.05):
    """对特征矩阵的每个连续特征执行ChiMerge离散化，返回每个特征的分箱边界字典。
    参数:
        features: pandas DataFrame 或 2D numpy数组, shape (n_samples, n_features)
        labels: 1D数组或列表, 每个样本对应的离散类别标签 (监督信息)
        k: 初始等距分箱数量
        alpha: 卡方阈值初始显著性水平
    返回:
        discretization_bins: {feature_name 或 索引: cut_points数组}
            每个特征对应的切割点(升序)。返回的切割点列表长度为 (最终区间数-1)。
            如返回空array表示该特征无需切分（所有值归为一类）。
    """
    if isinstance(features, pd.DataFrame):
        columns = features.columns
        X = features.values
    else:
        X = np.array(features)
        columns = np.arange(X.shape[1])
    labels = np.asarray(labels)
    discretization_bins = {}
    for idx, col in enumerate(columns):
        values = X[:, idx]
        cut_points = modified_chi_merge(values, labels, k=k, alpha=alpha)
        discretization_bins[col] = cut_points
    return discretization_bins

# ================== 特征选择 (BEFS方法) ==================
def baseline_set_generation(X, y, k=15):
    """BEFS算法基线特征集生成阶段:
    结合嵌入式和包装式方法选择候选特征集合。
    参数:
        X: numpy数组或 DataFrame，形状 (n_samples, n_features)
        y: 目标值数组 (长度n_samples)
        k: 每种方法选取的特征数 (嵌入法和包装法各选取前k个特征)
    返回:
        baseline_features: 基线特征集的索引列表
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)
    # 嵌入法: 随机森林回归计算特征重要性，选取最高的k个特征
    model = RandomForestRegressor(random_state=0)
    model.fit(X, y)
    importances = model.feature_importances_
    embedded_features = np.argsort(importances)[-k:]
    # 包装法: 递归特征消除 (RFE) 选择k个特征
    rfe = RFE(estimator=RandomForestRegressor(random_state=0), n_features_to_select=k)
    rfe.fit(X, y)
    wrapper_features = np.where(rfe.support_)[0]
    # 合并嵌入法和包装法结果
    baseline_set = set(embedded_features).union(set(wrapper_features))
    baseline_features = sorted(list(baseline_set))
    return baseline_features

def backward_elimination(X, y, feature_indices, threshold=0.01):
    """BEFS算法后向逐步消除阶段:
    基于给定的初始特征集合，对模型性能影响最小的特征逐步移除。
    参数:
        X: numpy数组或 DataFrame，形状 (n_samples, n_features)
        y: 目标值数组 (长度 n_samples)
        feature_indices: 初始特征集合的索引列表 (如 baseline_set_generation 输出)
        threshold: 性能提升阈值比例 (默认为0.01，即1%)
                   每次移除特征后，如果误差降低不到 threshold 比例，则停止移除。
    返回:
        current_features: 后向消除后剩余的特征索引列表
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)
    current_features = list(feature_indices)
    if len(current_features) == 0:
        return []
    model = RandomForestRegressor(random_state=0)
    model.fit(X[:, current_features], y)
    best_score = mean_squared_error(y, model.predict(X[:, current_features]))
    while len(current_features) > 1:
        best_candidate_score = None
        best_candidate_idx = None
        # 尝试移除集合中的每个特征，记录移除后的MSE
        for i in range(len(current_features)):
            temp_features = current_features[:i] + current_features[i+1:]
            model.fit(X[:, temp_features], y)
            mse = mean_squared_error(y, model.predict(X[:, temp_features]))
            if best_candidate_score is None or mse < best_candidate_score:
                best_candidate_score = mse
                best_candidate_idx = i
        # 计算误差改善率
        improvement = (best_score - best_candidate_score) / best_score if best_candidate_score is not None else 0
        if best_candidate_score is None or improvement < threshold:
            break  # 若无显著改善则停止
        # 移除最不重要特征并更新最佳误差
        best_score = best_candidate_score
        current_features.pop(best_candidate_idx)
    return current_features

def befs_feature_selection(X, sbp_values, dbp_values, k=20, threshold=0.01):
    """执行BEFS特征选择: 对SBP和DBP分别进行基线特征选取和后向消除, 并合并结果。
    参数:
        X: 特征矩阵 (numpy数组或 DataFrame, shape: n_samples x n_features)
        sbp_values: 收缩压 (SBP) 目标值数组
        dbp_values: 舒张压 (DBP) 目标值数组
        k: 基线阶段每种方法选取的特征数 (默认20)
        threshold: 后向消除阶段的性能阈值 (默认0.01)
    返回:
        final_features: 最终选定的特征索引列表 (升序)
        selected_sbp: SBP子集优化后的特征索引列表
        selected_dbp: DBP子集优化后的特征索引列表
    """
    # 基线特征子集选取
    baseline_sbp = baseline_set_generation(X, sbp_values, k=k)
    baseline_dbp = baseline_set_generation(X, dbp_values, k=k)
    # 后向逐步消除
    selected_sbp = backward_elimination(X, sbp_values, baseline_sbp, threshold=threshold)
    selected_dbp = backward_elimination(X, dbp_values, baseline_dbp, threshold=threshold)
    # 合并两个目标任务的特征集合
    final_set = set(selected_sbp) | set(selected_dbp)
    final_features = sorted(list(final_set))
    return final_features, selected_sbp, selected_dbp