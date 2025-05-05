import numpy as np

def mlcss_distance(y_pred, y_true, epsilon=None):
    """
    计算修改的最长公共子序列(MLCSS)距离:contentReference[oaicite:60]{index=60}。
    该距离衡量预测与真实波形序列的形状相似度（0表示完全匹配）。
    epsilon: 判定匹配的容差，默认取真实波形幅度范围的5%。
    """
    n = len(y_true)
    # 线性校正预测序列幅度以匹配真实序列尺度
    if np.std(y_pred) < 1e-6:
        scaled = y_pred - np.mean(y_pred) + np.mean(y_true)
    else:
        alpha = np.std(y_true) / (np.std(y_pred) + 1e-8)
        beta = np.mean(y_true) - alpha * np.mean(y_pred)
        scaled = alpha * y_pred + beta
    if epsilon is None:
        epsilon = 0.05 * (np.max(y_true) - np.min(y_true))
    # 计算LCSS长度（动态规划）
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            if abs(scaled[i-1] - y_true[j-1]) <= epsilon:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = dp[i-1][j] if dp[i-1][j] >= dp[i][j-1] else dp[i][j-1]
    lcss_length = dp[n][n]
    similarity = lcss_length / n
    distance = 1.0 - similarity
    return distance

def compute_metrics(predictions, truths):
    """
    计算波形序列的MAE, ME, SD和平均MLCSS距离。
    predictions, truths: numpy数组 (N_samples, seq_len)
    返回包含各指标的字典。
    """
    pred_flat = predictions.flatten()
    true_flat = truths.flatten()
    error = pred_flat - true_flat
    mae = np.mean(np.abs(error))
    me = np.mean(error)
    sd = np.std(error, ddof=0)  # 总体标准差
    # 计算每段的MLCSS距离并取平均
    mlcss_list = []
    for i in range(predictions.shape[0]):
        mlcss_list.append(mlcss_distance(predictions[i], truths[i]))
    mlcss_avg = float(np.mean(mlcss_list))
    return {"MAE": mae, "ME": me, "SD": sd, "MLCSS": mlcss_avg}

# 辅助函数：从BP波形序列提取SBP和DBP序列并计算相关性和误差
from scipy.stats import pearsonr

def sbp_dbp_metrics(predictions, truths):
    """
    计算预测和真实BP波形的SBP/DBP相关性和误差统计。
    返回 (SBP_r, DBP_r, SBP_ME, SBP_SD, DBP_ME, DBP_SD)。
    """
    # SBP: 每段BP波形的最大值; DBP: 每段的最小值
    sbp_pred = predictions.max(axis=1)
    sbp_true = truths.max(axis=1)
    dbp_pred = predictions.min(axis=1)
    dbp_true = truths.min(axis=1)
    # 皮尔逊相关系数
    sbp_r = pearsonr(sbp_pred, sbp_true)[0]
    dbp_r = pearsonr(dbp_pred, dbp_true)[0]
    # 误差均值和标准差
    sbp_diff = sbp_pred - sbp_true
    dbp_diff = dbp_pred - dbp_true
    sbp_me = np.mean(sbp_diff); sbp_sd = np.std(sbp_diff, ddof=0)
    dbp_me = np.mean(dbp_diff); dbp_sd = np.std(dbp_diff, ddof=0)
    return sbp_r, dbp_r, sbp_me, sbp_sd, dbp_me, dbp_sd
