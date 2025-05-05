from models.informer import InformerEncoder
from models.morphological_net import SEResNetFeatureExtractor
import torch
import torch.nn as nn


class TeacherModel(nn.Module):
    """教师模型：包含Transformer主干和形态特征分支，将两者融合用于BP波形预测。"""
    def __init__(self, ppg_input_dim: int = 1, d_model: int = 64, n_heads: int = 8,
                 d_ff: int = 256, num_layers: int = 4, morph_feat_dim: int = 30):
        super(TeacherModel, self).__init__()
        # 1. Transformer编码器，从原始PPG序列提取时序特征
        self.informer = InformerEncoder(input_dim=ppg_input_dim, d_model=d_model,
                                        n_heads=n_heads, d_ff=d_ff, num_layers=num_layers)
        # 2. 形态学特征提取网络，将先验特征向量嵌入到d_model维空间
        self.morph_net = SEResNetFeatureExtractor(in_features=morph_feat_dim, embed_dim=d_model)
        # 3. 特征融合输出头：将Informer特征与形态embedding拼接，经两层MLP输出BP波形
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)  # 对每个时间步输出一个BP值
        )

    def forward(self, ppg_segment: torch.Tensor, morph_features: torch.Tensor) -> torch.Tensor:
        # ppg_segment: (batch, seq_len, 1); morph_features: (batch, morph_feat_dim)
        feats = self.informer(ppg_segment)                   # 提取PPG时序特征 (batch, seq_len, d_model)
        morph_emb = self.morph_net(morph_features)           # 提取形态学embedding (batch, d_model)
        # 将形态embedding扩展并与每个时间步的Transformer特征拼接
        morph_rep = morph_emb.unsqueeze(1).expand(-1, feats.size(1), -1)  # (batch, seq_len, d_model)
        fused = torch.cat([feats, morph_rep], dim=-1)                    # (batch, seq_len, 2*d_model)
        bp_pred = self.fusion_head(fused).squeeze(-1)                    # (batch, seq_len) 输出BP波形
        return bp_pred

class StudentModel(nn.Module):
    """学生模型：仅包含Transformer编码器，从PPG直接预测BP波形。"""
    def __init__(self, ppg_input_dim: int = 1, d_model: int = 64, n_heads: int = 8,
                 d_ff: int = 256, num_layers: int = 4):
        super(StudentModel, self).__init__()
        # 学生模型的PPG编码器
        self.informer = InformerEncoder(input_dim=ppg_input_dim, d_model=d_model,
                                        n_heads=n_heads, d_ff=d_ff, num_layers=num_layers)
        # 输出头：将Transformer提取的特征映射为BP波形
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, ppg_segment: torch.Tensor) -> torch.Tensor:
        # ppg_segment: (batch, seq_len, 1)
        feats = self.informer(ppg_segment)            # 提取PPG时序特征 (batch, seq_len, d_model)
        bp_pred = self.output_head(feats).squeeze(-1) # 映射到BP输出 (batch, seq_len)
        return bp_pred
