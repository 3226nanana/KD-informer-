import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """正弦位置编码模块，为Transformer提供位置信息。"""
    def __init__(self, d_model: int, max_len: int = 10000):
        super(PositionalEncoding, self).__init__()
        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 按Transformer论文中的公式计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x形状: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # 将对应长度的位置编码加到输入上
        x = x + self.pe[:, :seq_len, :]
        return x

class InformerEncoder(nn.Module):
    """基于Transformer Encoder的PPG特征提取主干（Informer简化实现）。"""
    def __init__(self, input_dim: int = 1, d_model: int = 64, n_heads: int = 8,
                 d_ff: int = 256, num_layers: int = 4, dropout: float = 0.1):
        super(InformerEncoder, self).__init__()
        self.d_model = d_model
        # 将原始PPG序列的每个时间点的值映射到d_model维空间（线性嵌入层）
        self.input_proj = nn.Linear(input_dim, d_model)
        # 加入位置编码以注入时间位置信息
        self.pos_enc = PositionalEncoding(d_model, max_len=10000)
        # 堆叠Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                  dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: 张量，形状 (batch, seq_len, input_dim)，原始PPG序列片段。
        输出:
            Tensor (batch, seq_len, d_model)，每个时间步提取的高维特征表示。
        """
        # 1. 线性投影到模型维度
        x_proj = self.input_proj(x)               # (batch, seq_len, d_model)
        # 2. 加上位置编码
        x_enc = self.pos_enc(x_proj)             # (batch, seq_len, d_model)
        # 3. 经过多层Transformer Encoder提取时序依赖特征
        x_feats = self.encoder(x_enc)            # (batch, seq_len, d_model)
        return x_feats