import numpy as np
from feat import load_ppg_abp_from_folder, segment_signal, extract_ppg_features_75
from models.kd_models import StudentModel, TeacherModel

# 1. 数据加载：从指定文件夹读取 PPG 和 ABP 原始信号记录
folder_path = 'data'  # 假设数据存放路径
records = load_ppg_abp_from_folder(folder_path)
print(f"Loaded {len(records)} records from {folder_path}.")

# 对 PPG 和 ABP 信号进行全局归一化（例如将[min, max]线性缩放到[0,1]范围），提高数值稳定性
all_ppg_vals = np.concatenate([rec['ppg'] for rec in records]) if len(records) > 0 else np.array([])
all_abp_vals = np.concatenate([rec['abp'] for rec in records]) if len(records) > 0 else np.array([])
if all_ppg_vals.size > 0 and all_abp_vals.size > 0:
    ppg_min, ppg_max = np.min(all_ppg_vals), np.max(all_ppg_vals)
    abp_min, abp_max = np.min(all_abp_vals), np.max(all_abp_vals)
    # 避免除以零，当数据恒定时跳过归一化
    if ppg_max - ppg_min > 1e-8:
        for rec in records:
            rec['ppg'] = (rec['ppg'] - ppg_min) / (ppg_max - ppg_min + 1e-8)
    if abp_max - abp_min > 1e-8:
        for rec in records:
            rec['abp'] = (rec['abp'] - abp_min) / (abp_max - abp_min + 1e-8)
    print("Normalized PPG and ABP signals to [0,1] range.")

# 2. 信号分段：将每条记录的 PPG/ABP 按固定窗口和步长切分为多个片段
window_size = 125 * 8   # 窗口长度，例如 125Hz 采样率下 8 秒窗（1000 个点）
step_size   = 125 * 3   # 滑动步长，例如 125Hz 下 3 秒步长（375 个点）
segments = []
for rec in records:
    segs = segment_signal(rec['ppg'], rec['abp'], window_size, step_size)
    segments.extend(segs)
print(f"Total segments before filtering: {len(segments)}")

# 3. 提取形态学特征：对每个 PPG 片段提取75维特征，并过滤异常片段
all_features = []
ppg_segments = []
bp_segments = []
skipped_segments = 0
for seg in segments:
    ppg_seg = seg['ppg_segment']
    bp_seg = seg['abp_segment']
    # 跳过包含 NaN/Inf 的异常片段
    if np.isnan(ppg_seg).any() or np.isnan(bp_seg).any() or np.isinf(ppg_seg).any() or np.isinf(bp_seg).any():
        skipped_segments += 1
        continue
    # 提取当前 PPG 片段的75维形态学特征
    feat_dict = extract_ppg_features_75(ppg_seg)
    feat_values = list(feat_dict.values())
    # 4. 特征选择：根据 BEFS 算法的结果，选取重要的特征子集 (30维)
    selected_indices = [0, 4, 6, 8, 9, 10, 14, 16, 17, 18,
                        19, 21, 23, 29, 36, 45, 46, 47, 48, 50,
                        56, 57, 61, 63, 64, 65, 66, 69, 70, 71]
    selected_feat = [feat_values[i] for i in selected_indices]
    # 如果特征向量中存在 NaN 或 Inf，跳过该片段
    if np.isnan(selected_feat).any() or np.isinf(selected_feat).any():
        skipped_segments += 1
        continue
    all_features.append(selected_feat)
    ppg_segments.append(ppg_seg)
    bp_segments.append(bp_seg)
print(f"Skipped {skipped_segments} segments due to NaN/Inf in inputs or features.")
print(f"Total valid segments: {len(ppg_segments)}")

# 转换列表为 NumPy 数组，供后续 Dataset 使用
PPG_array  = np.array(ppg_segments)    # 形状: (N_segments, seq_len)
BP_array   = np.array(bp_segments)     # 形状: (N_segments, seq_len)
feat_array = np.array(all_features)    # 形状: (N_segments, 30)
print("PPG_array shape:", PPG_array.shape, "feat_array shape:", feat_array.shape)

import torch
from torch.utils.data import DataLoader
from data.dataset import BPWaveformDataset  # 数据集定义

# 5. 划分训练/测试集 (例如按 8:2 拆分)
num_samples = PPG_array.shape[0]
train_size = int(0.8 * num_samples)
indices = np.random.permutation(num_samples)
train_idx, test_idx = indices[:train_size], indices[train_size:]
PPG_train, BP_train, feat_train = PPG_array[train_idx], BP_array[train_idx], feat_array[train_idx]
PPG_test,  BP_test,  feat_test  = PPG_array[test_idx],  BP_array[test_idx],  feat_array[test_idx]

# 构建 Dataset 和 DataLoader
train_dataset = BPWaveformDataset({'PPG': PPG_train, 'BP': BP_train, 'features': feat_train}, use_morph=True)
test_dataset  = BPWaveformDataset({'PPG': PPG_test,  'BP': BP_test,  'features': feat_test}, use_morph=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 6. 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = TeacherModel(morph_feat_dim=feat_train.shape[1]).to(device)
student_model = StudentModel().to(device)
print(f"Initialized TeacherModel (morph_feat_dim={feat_train.shape[1]}) and StudentModel on {device}.")

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

# 7. 训练教师模型（有先验特征监督）
teacher_model.train()
num_epochs = 10
for epoch in range(1, num_epochs+1):
    epoch_loss = 0.0
    for ppg, feat, bp in train_loader:
        ppg = ppg.to(device)       # (batch, seq_len, 1)
        feat = feat.to(device)     # (batch, 30)
        bp = bp.to(device)         # (batch, seq_len)
        teacher_optimizer.zero_grad()
        # 前向传播得到 BP 波形预测
        bp_pred = teacher_model(ppg, feat)       # 输出形状: (batch, seq_len)
        # 检查前向输出是否出现异常值
        if torch.isnan(bp_pred).any() or torch.isinf(bp_pred).any():
            print(f"[Debug] NaN/Inf detected in TeacherModel output at epoch {epoch}")
        # 计算与真实 BP 波形的均方误差损失
        loss = criterion(bp_pred, bp)
        if not torch.isfinite(loss):
            print(f"[Debug] TeacherModel loss is abnormal (NaN/Inf) at epoch {epoch}: {loss.item()}")
            continue  # 跳过该批次的更新，以避免传播 NaN
        # 反向传播和优化
        loss.backward()
        # 梯度裁剪以避免梯度爆炸
        torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)
        teacher_optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
    print(f"Epoch {epoch}/{num_epochs} - TeacherModel MSE: {avg_loss:.4f}")
# 教师模型训练完毕

# 8. 知识蒸馏训练学生模型
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

student_model.train()
distill_lambda = 0.5  # 知识蒸馏损失权重超参数
num_distill_epochs = 10
for epoch in range(1, num_distill_epochs+1):
    epoch_loss = 0.0
    for ppg, feat, bp in train_loader:
        ppg = ppg.to(device)
        feat = feat.to(device)
        bp = bp.to(device)
        student_optimizer.zero_grad()
        # 教师模型预测（不计算梯度）
        with torch.no_grad():
            teacher_pred = teacher_model(ppg, feat)   # 教师输出 BP 波形
        if torch.isnan(teacher_pred).any() or torch.isinf(teacher_pred).any():
            print(f"[Debug] NaN/Inf detected in teacher_pred at epoch {epoch}")
        # 学生模型预测
        student_pred = student_model(ppg)             # 学生输出 BP 波形
        if torch.isnan(student_pred).any() or torch.isinf(student_pred).any():
            print(f"[Debug] NaN/Inf detected in StudentModel output at epoch {epoch}")
        # 学生损失 = 与真实 BP 的误差 + 蒸馏误差（与教师输出的误差）
        loss_true = criterion(student_pred, bp)
        loss_kd   = criterion(student_pred, teacher_pred)
        loss = loss_true + distill_lambda * loss_kd
        if not torch.isfinite(loss):
            print(f"[Debug] StudentModel loss is abnormal (NaN/Inf) at epoch {epoch}: "
                  f"loss_true={loss_true.item()}, loss_kd={loss_kd.item()}")
            continue  # 跳过异常损失的更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        student_optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
    print(f"Epoch {epoch}/{num_distill_epochs} - StudentModel Distill MSE: {avg_loss:.4f}")
# 学生模型训练完毕

# 9. 模型评估：在测试集上比较教师和学生模型性能
teacher_model.eval()
student_model.eval()
teacher_preds = []
student_preds = []
true_BP = []
with torch.no_grad():
    for ppg, feat, bp in test_loader:
        ppg = ppg.to(device)
        feat = feat.to(device)
        bp = bp.to(device)
        # 获取教师和学生的预测输出
        teacher_out = teacher_model(ppg, feat)   # (batch, seq_len)
        student_out = student_model(ppg)         # (batch, seq_len)
        teacher_preds.append(teacher_out.cpu().numpy())
        student_preds.append(student_out.cpu().numpy())
        true_BP.append(bp.cpu().numpy())
# 如果有测试数据，计算评估指标
if len(teacher_preds) > 0:
    teacher_preds = np.concatenate(teacher_preds, axis=0)
    student_preds = np.concatenate(student_preds, axis=0)
    true_BP = np.concatenate(true_BP, axis=0)
    from utils.metrics import compute_metrics
    teacher_metrics = compute_metrics(teacher_preds, true_BP)
    student_metrics = compute_metrics(student_preds, true_BP)
    print("Teacher Model Metrics:", teacher_metrics)
    print("Student Model Metrics:", student_metrics)
else:
    print("No test data available to compute metrics.")
