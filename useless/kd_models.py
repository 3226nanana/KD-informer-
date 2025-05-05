import torch
import torch.nn as nn
from models.informer import InformerEncoder
from models.morphological_net import SEResNetFeatureExtractor

class TeacherModel(nn.Module):
    """
    教师模型：包含Informer主干和形态学特征分支，通过融合两者来预测血压波形。
    教师模型利用先验特征提高预测准确度，作为知识蒸馏中的教师网络。
    """
    def __init__(self, ppg_input_dim: int = 1, d_model: int = 64, n_heads: int = 8,
                 d_ff: int = 256, num_layers: int = 4, morph_feat_dim: int = 30):
        super(TeacherModel, self).__init__()
        # 1. Informer主干，用于从原始PPG序列提取时序特征
        self.informer = InformerEncoder(input_dim=ppg_input_dim, d_model=d_model,
                                        n_heads=n_heads, d_ff=d_ff, num_layers=num_layers)
        # 2. 形态学特征提取网络，将先验形态学特征嵌入到d_model维空间
        self.morph_net = SEResNetFeatureExtractor(in_features=morph_feat_dim, embed_dim=d_model)
        # 3. 特征融合头：将Informer时序特征与形态学特征嵌入拼接后，通过MLP输出BP值
        #   使用两层MLP实现融合 (公式(9) 的实现)
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    def forward(self, ppg_segment: torch.Tensor, morph_features: torch.Tensor) -> torch.Tensor:
        """
        输入:
            ppg_segment: 张量 (batch, seq_len, 1)，原始PPG信号片段。
            morph_features: 张量 (batch, morph_feat_dim)，对应PPG片段的形态学特征向量。
        输出:
            张量 (batch, seq_len)，预测的对应BP波形序列。
        """
        # (a) 提取PPG序列的时序特征表示 (batch, seq_len, d_model)
        feats = self.informer(ppg_segment)                   # Informer编码输出
        # (b) 提取形态学特征嵌入表示 (batch, d_model)
        morph_emb = self.morph_net(morph_features)           # 形态学特征提取网络输出
        # (c) 将形态学嵌入扩展并与每个时间步的Informer特征拼接
        # morph_emb.unsqueeze(1) 得到 (batch, 1, d_model)，沿时间步重复以匹配序列长度
        morph_rep = morph_emb.unsqueeze(1).expand(-1, feats.size(1), -1)  # (batch, seq_len, d_model)
        fused = torch.cat([feats, morph_rep], dim=-1)        # (batch, seq_len, 2*d_model)
        # (d) 将融合特征输入MLP，输出每个时间步对应的BP值
        bp_pred = self.fusion_head(fused).squeeze(-1)        # (batch, seq_len)
        return bp_pred


class StudentModel(nn.Module):
    """
    学生模型：仅包含Informer主干，用于从PPG直接预测BP波形。
    学生模型结构更轻量，不使用形态学先验特征，在知识蒸馏过程中从教师模型学习。
    """
    def __init__(self, ppg_input_dim: int = 1, d_model: int = 64, n_heads: int = 8,
                 d_ff: int = 256, num_layers: int = 4):
        super(StudentModel, self).__init__()
        # 学生仅有PPG的Informer编码器
        self.informer = InformerEncoder(input_dim=ppg_input_dim, d_model=d_model,
                                        n_heads=n_heads, d_ff=d_ff, num_layers=num_layers)
        # 输出头：将Informer提取的特征映射为BP值
        self.output_head = nn.Linear(d_model, 1)
        # （可选）可使用两层MLP增强学生模型容量:
        # self.output_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
    def forward(self, ppg_segment: torch.Tensor) -> torch.Tensor:
        """
        输入:
            ppg_segment: 张量 (batch, seq_len, 1)，PPG信号片段。
        输出:
            张量 (batch, seq_len)，预测的BP波形序列。
        """
        # 提取PPG时序特征
        feats = self.informer(ppg_segment)            # (batch, seq_len, d_model)
        # 映射到BP输出
        bp_pred = self.output_head(feats).squeeze(-1) # (batch, seq_len)
        return bp_pred