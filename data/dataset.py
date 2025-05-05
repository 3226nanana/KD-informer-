import torch
from torch.utils.data import Dataset
import numpy as np

class BPWaveformDataset(Dataset):
    """BP波形估计数据集，将PPG片段及对应BP波形（和可选特征）提供给模型。"""
    def __init__(self, data, use_morph=True):
        """
        data: 字典{'PPG': array(N, seq_len), 'BP': array(N, seq_len), 'features': array(N, feat_dim)} 或同结构的NPZ文件路径
        use_morph: 是否提供形态学特征（教师模型训练阶段为True，学生模型单独使用可为False）
        """
        # 加载数据
        if isinstance(data, str):
            data = np.load(data)
        self.ppg_segments = data['PPG']
        self.bp_segments = data['BP']
        if use_morph:
            if 'features' in data:
                self.morph_features = data['features']
            else:
                raise ValueError("use_morph=True 但未提供形态特征features")
        else:
            self.morph_features = None
        self.use_morph = use_morph

    def __len__(self):
        return len(self.ppg_segments)

    def __getitem__(self, idx):
        # 提取第idx个片段，并转为torch.Tensor
        ppg = torch.tensor(self.ppg_segments[idx], dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        bp  = torch.tensor(self.bp_segments[idx], dtype=torch.float32)                # (seq_len,)
        if self.use_morph:
            feat = torch.tensor(self.morph_features[idx], dtype=torch.float32)        # (feat_dim,)
            return ppg, feat, bp
        else:
            return ppg, bp
