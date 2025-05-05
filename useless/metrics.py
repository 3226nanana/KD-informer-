# utils/metrics.py
import numpy as np

def mlcss_distance(y_pred, y_true, epsilon=None):
    """
    Compute the Modified Longest Common Subsequence (MLCSS) distance between two sequences.
    y_pred, y_true: 1D numpy arrays of equal length representing the predicted and true waveform.
    epsilon: tolerance for matching points (if None, use 5% of true range as default).
    Returns: distance value (0 = perfect match, closer to 1 = poor match).
    """
    n = len(y_true)
    # Linear calibration of y_pred to y_true
    if np.std(y_pred) < 1e-6:
        # avoid division by zero if pred is nearly constant
        scaled = y_pred - np.mean(y_pred) + np.mean(y_true)
    else:
        alpha = np.std(y_true) / (np.std(y_pred) + 1e-8)
        beta = np.mean(y_true) - alpha * np.mean(y_pred)
        scaled = alpha * y_pred + beta
    # Set epsilon if not provided
    if epsilon is None:
        epsilon = 0.05 * (np.max(y_true) - np.min(y_true))  # 5% of true waveform range
    # Compute LCSS length via dynamic programming
    # dp[i][j] = length of LCSS of scaled[:i] and y_true[:j]
    m = n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if abs(scaled[i-1] - y_true[j-1]) <= epsilon:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = dp[i-1][j] if dp[i-1][j] >= dp[i][j-1] else dp[i][j-1]
    lcss_length = dp[n][m]
    # Similarity = fraction of points that can be matched
    similarity = lcss_length / n
    distance = 1.0 - similarity
    return distance

def compute_metrics(predictions, truths):
    """
    Compute MAE, ME, SD, and MLCSS distance given arrays of predicted and true waveforms.
    predictions, truths: numpy arrays of shape (N_samples, seq_len)
    Returns: dict of metrics.
    """
    # Flatten all sequences to compute overall error metrics across dataset
    pred_flat = predictions.flatten()
    true_flat = truths.flatten()
    error = pred_flat - true_flat
    mae = np.mean(np.abs(error))
    me = np.mean(error)
    sd = np.std(error, ddof=0)  # population standard deviation
    # Compute MLCSS on each sample and take average (mean distance across samples)
    mlcss_list = []
    for i in range(predictions.shape[0]):
        mlcss_list.append(mlcss_distance(predictions[i], truths[i]))
    mlcss_avg = float(np.mean(mlcss_list))
    return {"MAE": mae, "ME": me, "SD": sd, "MLCSS": mlcss_avg}