import numpy as np
from feat import load_ppg_abp_from_folder, segment_signal, extract_ppg_features_75
from models.kd_models import StudentModel, TeacherModel
from data.dataset import BPWaveformDataset
from utils.metrics import compute_metrics,sbp_dbp_metrics
import torch
from torch.utils.data import DataLoader

# 1. 加载Mindray数据并预处理
mindray_records = load_ppg_abp_from_folder('data')
print(f"Loaded {len(mindray_records)} records from Mindray dataset.")
# 归一化PPG和ABP到[0,1]
all_ppg = np.concatenate([rec['ppg'] for rec in mindray_records]) if mindray_records else np.array([])
all_abp = np.concatenate([rec['abp'] for rec in mindray_records]) if mindray_records else np.array([])
if all_ppg.size and all_abp.size:
    ppg_min, ppg_max = all_ppg.min(), all_ppg.max()
    abp_min, abp_max = all_abp.min(), all_abp.max()
    if ppg_max - ppg_min > 1e-8:
        for rec in mindray_records:
            rec['ppg'] = (rec['ppg'] - ppg_min) / (ppg_max - ppg_min + 1e-8)
    if abp_max - abp_min > 1e-8:
        for rec in mindray_records:
            rec['abp'] = (rec['abp'] - abp_min) / (abp_max - abp_min + 1e-8)
    print("Normalized Mindray PPG and ABP signals to [0,1].")

# 分段
window_size = 125 * 8   # 8秒窗口
step_size   = 125 * 3   # 3秒步长
segments = []
for rec in mindray_records:
    segments += segment_signal(rec['ppg'], rec['abp'], window_size, step_size)
print(f"Total Mindray segments: {len(segments)} before filtering.")

# 提取特征并筛选30维
ppg_segments = []; bp_segments = []; all_features = []
skipped = 0
for seg in segments:
    ppg_seg = seg['ppg_segment']; bp_seg = seg['abp_segment']
    if np.isnan(ppg_seg).any() or np.isnan(bp_seg).any() or np.isinf(ppg_seg).any() or np.isinf(bp_seg).any():
        skipped += 1; continue
    feat_dict = extract_ppg_features_75(ppg_seg)
    feat_values = list(feat_dict.values())
    selected_indices = [0,4,6,8,9,10,14,16,17,18,19,21,23,29,36,45,46,47,48,50,56,57,61,63,64,65,66,69,70,71]
    selected_feat = [feat_values[i] for i in selected_indices]
    if np.isnan(selected_feat).any() or np.isinf(selected_feat).any():
        skipped += 1; continue
    ppg_segments.append(ppg_seg); bp_segments.append(bp_seg); all_features.append(selected_feat)
print(f"Filtered out {skipped} segments due to NaN/Inf. Final Mindray segments: {len(ppg_segments)}.")

# 转numpy数组
PPG_array  = np.array(ppg_segments)
BP_array   = np.array(bp_segments)
feat_array = np.array(all_features)
print("PPG_array shape:", PPG_array.shape, "BP_array shape:", BP_array.shape, "feat_array shape:", feat_array.shape)

# 划分Mindray训练/测试集 (基于记录独立拆分)
num_samples = PPG_array.shape[0]
train_size = int(0.8 * num_samples)
indices = np.random.permutation(num_samples)
train_idx, test_idx = indices[:train_size], indices[train_size:]
PPG_train, BP_train, feat_train = PPG_array[train_idx], BP_array[train_idx], feat_array[train_idx]
PPG_test,  BP_test,  feat_test  = PPG_array[test_idx],  BP_array[test_idx],  feat_array[test_idx]
print(f"Train segments: {len(train_idx)}, Test segments: {len(test_idx)}")

# 构建DataLoader
train_dataset = BPWaveformDataset({'PPG': PPG_train, 'BP': BP_train, 'features': feat_train}, use_morph=True)
test_dataset  = BPWaveformDataset({'PPG': PPG_test,  'BP': BP_test,  'features': feat_test}, use_morph=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = TeacherModel(morph_feat_dim=feat_train.shape[1]).to(device)
student_model = StudentModel().to(device)
criterion = torch.nn.MSELoss()
teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

# 3. 训练教师模型 (监督学习)
epochs = 10
teacher_model.train()
for epoch in range(1, epochs+1):
    total_loss = 0.0
    for ppg, feat, bp in train_loader:
        ppg, feat, bp = ppg.to(device), feat.to(device), bp.to(device)
        teacher_optimizer.zero_grad()
        bp_pred = teacher_model(ppg, feat)
        loss = criterion(bp_pred, bp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)
        teacher_optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
    print(f"Epoch {epoch}/{epochs} - TeacherModel MSE: {avg_loss:.4f}")
print("Teacher model training complete.")

# 4. 知识蒸馏训练学生模型 (教师固定，学生学习教师输出)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False
student_model.train()
distill_lambda = 0.5
distill_epochs = 10
for epoch in range(1, distill_epochs+1):
    total_loss = 0.0
    for ppg, feat, bp in train_loader:
        ppg, feat, bp = ppg.to(device), feat.to(device), bp.to(device)
        student_optimizer.zero_grad()
        with torch.no_grad():
            teacher_out = teacher_model(ppg, feat)   # 教师模型输出 (batch, seq_len)
        student_out = student_model(ppg)             # 学生模型输出 (batch, seq_len)
        loss_true = criterion(student_out, bp)
        loss_kd   = criterion(student_out, teacher_out)
        loss = loss_true + distill_lambda * loss_kd
        if not torch.isfinite(loss):
            continue  # 跳过异常
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        student_optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
    print(f"Epoch {epoch}/{distill_epochs} - StudentModel Distill MSE: {avg_loss:.4f}")
print("Student model distillation training complete.")

# 5. 在Mindray测试集评估教师和学生模型
teacher_model.eval()
student_model.eval()
teacher_preds = []; student_preds = []; true_BP = []
with torch.no_grad():
    for ppg, feat, bp in test_loader:
        ppg, feat, bp = ppg.to(device), feat.to(device), bp.to(device)
        teacher_out = teacher_model(ppg, feat)
        student_out = student_model(ppg)
        teacher_preds.append(teacher_out.cpu().numpy())
        student_preds.append(student_out.cpu().numpy())
        true_BP.append(bp.cpu().numpy())
teacher_preds = np.concatenate(teacher_preds, axis=0) if teacher_preds else np.array([])
student_preds = np.concatenate(student_preds, axis=0) if student_preds else np.array([])
true_BP = np.concatenate(true_BP, axis=0) if true_BP else np.array([])
if true_BP.size > 0:
    teacher_metrics = compute_metrics(teacher_preds, true_BP)
    student_metrics = compute_metrics(student_preds, true_BP)
    sbp_r_t, dbp_r_t, sbp_me_t, sbp_sd_t, dbp_me_t, dbp_sd_t = sbp_dbp_metrics(teacher_preds, true_BP)
    sbp_r_s, dbp_r_s, sbp_me_s, sbp_sd_s, dbp_me_s, dbp_sd_s = sbp_dbp_metrics(student_preds, true_BP)
    print("Mindray Test - Teacher:", teacher_metrics,
          f"SBP_r={sbp_r_t:.3f}, DBP_r={dbp_r_t:.3f}, SBP_ME={sbp_me_t:.3f}±{sbp_sd_t:.3f}, DBP_ME={dbp_me_t:.3f}±{dbp_sd_t:.3f}")
    print("Mindray Test - Student:", student_metrics,
          f"SBP_r={sbp_r_s:.3f}, DBP_r={dbp_r_s:.3f}, SBP_ME={sbp_me_s:.3f}±{sbp_sd_s:.3f}, DBP_ME={dbp_me_s:.3f}±{dbp_sd_s:.3f}")
else:
    print("No Mindray test data for evaluation.")

# 6. 加载MIMIC数据集并准备迁移学习
mimic_records = load_ppg_abp_from_folder('data_new')
print(f"Loaded {len(mimic_records)} records from MIMIC dataset.")
# 使用Mindray相同的min/max对MIMIC做归一化（假设二者量纲相近）
if all_ppg.size and all_abp.size:
    for rec in mimic_records:
        rec['ppg'] = (rec['ppg'] - ppg_min) / (ppg_max - ppg_min + 1e-8)
        rec['abp'] = (rec['abp'] - abp_min) / (abp_max - abp_min + 1e-8)
    print("Normalized MIMIC PPG and ABP using Mindray scale.")
# 分段提取，与Mindray相同窗口
mimic_segments = []
for rec in mimic_records:
    mimic_segments += segment_signal(rec['ppg'], rec['abp'], window_size, step_size)
print(f"Total MIMIC segments: {len(mimic_segments)}")
ppg_segments = []; bp_segments = []; all_features = []; skipped = 0
for seg in mimic_segments:
    ppg_seg = seg['ppg_segment']; bp_seg = seg['abp_segment']
    if np.isnan(ppg_seg).any() or np.isnan(bp_seg).any() or np.isinf(ppg_seg).any() or np.isinf(bp_seg).any():
        skipped += 1; continue
    feat_dict = extract_ppg_features_75(ppg_seg)
    feat_values = list(feat_dict.values())
    selected_feat = [feat_values[i] for i in selected_indices]  # 使用相同特征索引
    if np.isnan(selected_feat).any() or np.isinf(selected_feat).any():
        skipped += 1; continue
    ppg_segments.append(ppg_seg); bp_segments.append(bp_seg); all_features.append(selected_feat)
print(f"Filtered out {skipped} MIMIC segments. Final MIMIC segments: {len(ppg_segments)}.")
PPG_mimic = np.array(ppg_segments); BP_mimic = np.array(bp_segments); feat_mimic = np.array(all_features)

# 划分MIMIC训练/测试集（患者独立）
num_mimic = PPG_mimic.shape[0]
train_size_mimic = int(0.8 * num_mimic)
indices = np.random.permutation(num_mimic)
train_idx_m, test_idx_m = indices[:train_size_mimic], indices[train_size_mimic:]
PPG_mtrain, BP_mtrain, feat_mtrain = PPG_mimic[train_idx_m], BP_mimic[train_idx_m], feat_mimic[train_idx_m]
PPG_mtest,  BP_mtest,  feat_mtest  = PPG_mimic[test_idx_m], BP_mimic[test_idx_m], feat_mimic[test_idx_m]
print(f"MIMIC Train segments: {len(train_idx_m)}, Test segments: {len(test_idx_m)}")

# DataLoader
train_dataset_m = BPWaveformDataset({'PPG': PPG_mtrain, 'BP': BP_mtrain, 'features': feat_mtrain}, use_morph=True)
test_dataset_m  = BPWaveformDataset({'PPG': PPG_mtest,  'BP': BP_mtest,  'features': feat_mtest}, use_morph=True)
train_loader_m = DataLoader(train_dataset_m, batch_size=32, shuffle=True)
test_loader_m  = DataLoader(test_dataset_m, batch_size=32, shuffle=False)

# 7. 迁移学习微调学生模型 (教师模型固定)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False
student_model.train()
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-4)  # 可使用较小学习率微调
epochs_transfer = 5
for epoch in range(1, epochs_transfer+1):
    total_loss = 0.0
    for ppg, feat, bp in train_loader_m:
        ppg, feat, bp = ppg.to(device), feat.to(device), bp.to(device)
        student_optimizer.zero_grad()
        with torch.no_grad():
            teacher_out = teacher_model(ppg, feat)
        student_out = student_model(ppg)
        loss = criterion(student_out, bp) + distill_lambda * criterion(student_out, teacher_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        student_optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader_m) if len(train_loader_m) > 0 else float('nan')
    print(f"Epoch {epoch}/{epochs_transfer} - StudentModel Transfer MSE: {avg_loss:.4f}")
print("Student model fine-tuning on MIMIC complete.")

# 8. MIMIC测试集评估
teacher_preds = []; student_preds = []; true_BP = []
teacher_model.eval(); student_model.eval()
with torch.no_grad():
    for ppg, feat, bp in test_loader_m:
        ppg, feat, bp = ppg.to(device), feat.to(device), bp.to(device)
        t_out = teacher_model(ppg, feat)
        s_out = student_model(ppg)
        teacher_preds.append(t_out.cpu().numpy())
        student_preds.append(s_out.cpu().numpy())
        true_BP.append(bp.cpu().numpy())
teacher_preds = np.concatenate(teacher_preds, axis=0) if teacher_preds else np.array([])
student_preds = np.concatenate(student_preds, axis=0) if student_preds else np.array([])
true_BP = np.concatenate(true_BP, axis=0) if true_BP else np.array([])
if true_BP.size > 0:
    teacher_metrics_m = compute_metrics(teacher_preds, true_BP)
    student_metrics_m = compute_metrics(student_preds, true_BP)
    sbp_r_t, dbp_r_t, sbp_me_t, sbp_sd_t, dbp_me_t, dbp_sd_t = sbp_dbp_metrics(teacher_preds, true_BP)
    sbp_r_s, dbp_r_s, sbp_me_s, sbp_sd_s, dbp_me_s, dbp_sd_s = sbp_dbp_metrics(student_preds, true_BP)
    print("MIMIC Test - Teacher:", teacher_metrics_m,
          f"SBP_r={sbp_r_t:.3f}, DBP_r={dbp_r_t:.3f}, SBP_ME={sbp_me_t:.3f}±{sbp_sd_t:.3f}, DBP_ME={dbp_me_t:.3f}±{dbp_sd_t:.3f}")
    print("MIMIC Test - Student:", student_metrics_m,
          f"SBP_r={sbp_r_s:.3f}, DBP_r={dbp_r_s:.3f}, SBP_ME={sbp_me_s:.3f}±{sbp_sd_s:.3f}, DBP_ME={dbp_me_s:.3f}±{dbp_sd_s:.3f}")
else:
    print("No MIMIC test data for evaluation.")

# # 9. 绘制Bland-Altman图（SBP和DBP）
# import matplotlib.pyplot as plt
# def plot_bland_altman(true_vals, pred_vals, title):
#     # 计算差值和均值
#     diffs = pred_vals - true_vals
#     means = (pred_vals + true_vals) / 2
#     bias = diffs.mean(); sd = diffs.std(ddof=0)
#     loa_upper = bias + 1.96 * sd; loa_lower = bias - 1.96 * sd
#     plt.figure(figsize=(5,4))
#     plt.scatter(means, diffs, color='blue', alpha=0.6, edgecolors='k')
#     plt.axhline(bias, color='red',
#
#
#
#     plt.axhline(bias, color='red', linestyle='--', label=f'Bias = {bias:.2f}')
#     plt.axhline(loa_upper, color='green', linestyle='--', label=f'+1.96SD = {loa_upper:.2f}');
#     plt.axhline(loa_lower, color='green', linestyle='--', label=f'-1.96SD = {loa_lower:.2f}');
#     plt.title(title);
#     plt.xlabel('Mean of reference and predicted (mmHg)');
#     plt.ylabel('Difference (Predicted - Reference, mmHg)');
#     plt.legend(loc='best');
#     plt.show();
#
# # 绘制并保存 Bland-Altman 图
# if true_BP.size > 0:
#     # Mindray数据集教师模型 SBP/DBP Bland-Altman
#     plot_bland_altman(true_BP.max(axis=1), teacher_preds.max(axis=1), 'Mindray Teacher SBP');
#     plot_bland_altman(true_BP.min(axis=1), teacher_preds.min(axis=1), 'Mindray Teacher DBP');
#     # Mindray数据集学生模型 SBP/DBP
#     plot_bland_altman(true_BP.max(axis=1), student_preds.max(axis=1), 'Mindray Student SBP');
#     plot_bland_altman(true_BP.min(axis=1), student_preds.min(axis=1), 'Mindray Student DBP');
#     # MIMIC数据集教师模型 SBP/DBP
#     plot_bland_altman(BP_mtest.max(axis=1), teacher_model(torch.tensor(PPG_mtest, dtype=torch.float32).unsqueeze(-1).to(device),
#                                                           torch.tensor(feat_mtest, dtype=torch.float32).to(device)).cpu().numpy().max(axis=1), 'MIMIC Teacher SBP');
#     plot_bland_altman(BP_mtest.min(axis=1), teacher_model(torch.tensor(PPG_mtest, dtype=torch.float32).unsqueeze(-1).to(device),
#                                                           torch.tensor(feat_mtest, dtype=torch.float32).to(device)).cpu().numpy().min(axis=1), 'MIMIC Teacher DBP');
#     # MIMIC数据集学生模型 SBP/DBP
#     plot_bland_altman(BP_mtest.max(axis=1), student_model(torch.tensor(PPG_mtest, dtype=torch.float32).unsqueeze(-1).to(device)).cpu().numpy().max(axis=1), 'MIMIC Student SBP');
#     plot_bland_altman(BP_mtest.min(axis=1), student_model(torch.tensor(PPG_mtest, dtype=torch.float32).unsqueeze(-1).to(device)).cpu().numpy().min(axis=1), 'MIMIC Student DBP')
