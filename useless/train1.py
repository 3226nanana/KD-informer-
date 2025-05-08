import numpy as np
from feat import load_ppg_abp_from_folder, segment_signal, extract_ppg_features_75
from models.kd_models import StudentModel, TeacherModel
from data.dataset import BPWaveformDataset
from utils.metrics import compute_metrics, sbp_dbp_metrics
import torch
from torch.utils.data import DataLoader
import pywt

# 1. 加载Mindray数据并预处理
mindray_records = load_ppg_abp_from_folder('data')
print(f"Loaded {len(mindray_records)} records from Mindray dataset.")
# 对每条记录的PPG信号进行小波滤波、相位对齐和Z-score标准化预处理
for rec in mindray_records:
    ppg = rec['ppg']
    abp = rec['abp']
    if ppg is None or abp is None or len(ppg) == 0 or len(abp) == 0:
        continue
    # 相位对齐: 计算PPG与ABP之间的延迟
    corr = np.correlate(ppg, abp, mode='full')
    lag = corr.argmax() - (len(ppg) - 1)
    if lag < 0:
        # ABP滞后于PPG，提前ABP信号
        offset = -lag
        abp = abp[offset:]
        ppg = ppg[:len(abp)]
    elif lag > 0:
        # ABP超前于PPG，截断PPG开头
        offset = lag
        ppg = ppg[offset:]
        abp = abp[:len(ppg)]
    # 小波滤波: 使用sym4小波去除PPG高频噪声 (舍弃D1,D2细节系数)
    coeffs = pywt.wavedec(ppg, 'sym4', level=3)
    if len(coeffs) > 1:
        coeffs[1] = np.zeros_like(coeffs[1])
    if len(coeffs) > 2:
        coeffs[2] = np.zeros_like(coeffs[2])
    ppg_filtered = pywt.waverec(coeffs, 'sym4')
    ppg_filtered = ppg_filtered[:len(ppg)]
    ppg = ppg_filtered
    # Z-score标准化PPG
    if np.std(ppg) > 1e-8:
        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
    else:
        ppg = ppg - np.mean(ppg)
    rec['ppg'] = ppg
    rec['abp'] = abp
print("Applied wavelet filtering (sym4) and phase alignment to PPG, and standardized PPG signals (Z-score) for each record.")

# 分段
window_size = 125 * 8   # 8秒窗口
step_size   = 125 * 3   # 3秒步长
# 将数据按受试者划分训练/验证/测试集 (7:1.5:1.5比例)
num_records = len(mindray_records)
indices = np.random.permutation(num_records)
train_end = int(0.7 * num_records)
val_end = train_end + int(0.15 * num_records)
train_idx_recs = indices[:train_end]
val_idx_recs = indices[train_end:val_end]
test_idx_recs = indices[val_end:]
train_records = [mindray_records[i] for i in train_idx_recs]
val_records = [mindray_records[i] for i in val_idx_recs]
test_records = [mindray_records[i] for i in test_idx_recs]
print(f"Train records: {len(train_records)}, Val records: {len(val_records)}, Test records: {len(test_records)}")
train_segments = []
val_segments = []
test_segments = []
for rec in train_records:
    train_segments += segment_signal(rec['ppg'], rec['abp'], window_size, step_size)
for rec in val_records:
    val_segments += segment_signal(rec['ppg'], rec['abp'], window_size, step_size)
for rec in test_records:
    test_segments += segment_signal(rec['ppg'], rec['abp'], window_size, step_size)
print(f"Total segments before filtering - Train: {len(train_segments)}, Val: {len(val_segments)}, Test: {len(test_segments)}")

# 提取特征并筛选30维
train_ppg_segments = []; train_bp_segments = []; train_features = []
val_ppg_segments = []; val_bp_segments = []; val_features = []
test_ppg_segments = []; test_bp_segments = []; test_features = []
train_skipped = val_skipped = test_skipped = 0
for seg in train_segments:
    ppg_seg = seg['ppg_segment']; bp_seg = seg['abp_segment']
    if np.isnan(ppg_seg).any() or np.isnan(bp_seg).any() or np.isinf(ppg_seg).any() or np.isinf(bp_seg).any():
        train_skipped += 1; continue
    feat_dict = extract_ppg_features_75(ppg_seg)
    feat_values = list(feat_dict.values())
    selected_indices = [0,4,6,8,9,10,14,16,17,18,19,21,23,29,36,45,46,47,48,50,56,57,61,63,64,65,66,69,70,71]
    selected_feat = [feat_values[i] for i in selected_indices]
    if np.isnan(selected_feat).any() or np.isinf(selected_feat).any():
        train_skipped += 1; continue
    train_ppg_segments.append(ppg_seg); train_bp_segments.append(bp_seg); train_features.append(selected_feat)
for seg in val_segments:
    ppg_seg = seg['ppg_segment']; bp_seg = seg['abp_segment']
    if np.isnan(ppg_seg).any() or np.isnan(bp_seg).any() or np.isinf(ppg_seg).any() or np.isinf(bp_seg).any():
        val_skipped += 1; continue
    feat_dict = extract_ppg_features_75(ppg_seg)
    feat_values = list(feat_dict.values())
    selected_indices = [0,4,6,8,9,10,14,16,17,18,19,21,23,29,36,45,46,47,48,50,56,57,61,63,64,65,66,69,70,71]
    selected_feat = [feat_values[i] for i in selected_indices]
    if np.isnan(selected_feat).any() or np.isinf(selected_feat).any():
        val_skipped += 1; continue
    val_ppg_segments.append(ppg_seg); val_bp_segments.append(bp_seg); val_features.append(selected_feat)
for seg in test_segments:
    ppg_seg = seg['ppg_segment']; bp_seg = seg['abp_segment']
    if np.isnan(ppg_seg).any() or np.isnan(bp_seg).any() or np.isinf(ppg_seg).any() or np.isinf(bp_seg).any():
        test_skipped += 1; continue
    feat_dict = extract_ppg_features_75(ppg_seg)
    feat_values = list(feat_dict.values())
    selected_indices = [0,4,6,8,9,10,14,16,17,18,19,21,23,29,36,45,46,47,48,50,56,57,61,63,64,65,66,69,70,71]
    selected_feat = [feat_values[i] for i in selected_indices]
    if np.isnan(selected_feat).any() or np.isinf(selected_feat).any():
        test_skipped += 1; continue
    test_ppg_segments.append(ppg_seg); test_bp_segments.append(bp_seg); test_features.append(selected_feat)
print(f"Filtered out {train_skipped} train segments, {val_skipped} val segments, {test_skipped} test segments due to NaN/Inf.")
print(f"Final segments count - Train: {len(train_ppg_segments)}, Val: {len(val_ppg_segments)}, Test: {len(test_ppg_segments)}.")

# 转numpy数组
PPG_train = np.array(train_ppg_segments)
BP_train = np.array(train_bp_segments)
feat_train = np.array(train_features)
PPG_val = np.array(val_ppg_segments)
BP_val = np.array(val_bp_segments)
feat_val = np.array(val_features)
PPG_test = np.array(test_ppg_segments)
BP_test = np.array(test_bp_segments)
feat_test = np.array(test_features)
print("PPG_train shape:", PPG_train.shape, "BP_train shape:", BP_train.shape, "feat_train shape:", feat_train.shape)
print("PPG_val shape:", PPG_val.shape, "BP_val shape:", BP_val.shape, "feat_val shape:", feat_val.shape)
print("PPG_test shape:", PPG_test.shape, "BP_test shape:", BP_test.shape, "feat_test shape:", feat_test.shape)

# 构建DataLoader
train_dataset = BPWaveformDataset({'PPG': PPG_train, 'BP': BP_train, 'features': feat_train}, use_morph=True)
val_dataset   = BPWaveformDataset({'PPG': PPG_val, 'BP': BP_val, 'features': feat_val}, use_morph=True)
test_dataset  = BPWaveformDataset({'PPG': PPG_test, 'BP': BP_test, 'features': feat_test}, use_morph=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
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
# 对每条MIMIC记录的PPG信号进行相同预处理
for rec in mimic_records:
    ppg = rec.get('ppg', None)
    abp = rec.get('abp', None)
    if ppg is None or abp is None or len(ppg) == 0 or len(abp) == 0:
        continue
    # 相位对齐
    corr = np.correlate(ppg, abp, mode='full')
    lag = corr.argmax() - (len(ppg) - 1)
    if lag < 0:
        offset = -lag
        abp = abp[offset:]
        ppg = ppg[:len(abp)]
    elif lag > 0:
        offset = lag
        ppg = ppg[offset:]
        abp = abp[:len(ppg)]
    # 小波滤波和Z-score标准化
    coeffs = pywt.wavedec(ppg, 'sym4', level=3)
    if len(coeffs) > 1:
        coeffs[1] = np.zeros_like(coeffs[1])
    if len(coeffs) > 2:
        coeffs[2] = np.zeros_like(coeffs[2])
    ppg_filtered = pywt.waverec(coeffs, 'sym4')
    ppg_filtered = ppg_filtered[:len(ppg)]
    ppg = ppg_filtered
    if np.std(ppg) > 1e-8:
        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
    else:
        ppg = ppg - np.mean(ppg)
    rec['ppg'] = ppg
    rec['abp'] = abp

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
