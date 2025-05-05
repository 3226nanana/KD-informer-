import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """正弦位置编码，为Transformer提供时序位置信息。"""
    def __init__(self, d_model: int, max_len: int = 10000):
        super(PositionalEncoding, self).__init__()
        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class InformerEncoder(nn.Module):
    """基于Transformer Encoder的PPG特征提取模块（Informer简化实现）。"""
    def __init__(self, input_dim: int = 1, d_model: int = 64, n_heads: int = 8,
                 d_ff: int = 256, num_layers: int = 4, dropout: float = 0.1):
        super(InformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)      # 输入投影层，将PPG值映射到d_model维
        self.pos_enc = PositionalEncoding(d_model)           # 正弦位置编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                  dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)，input_dim=1时即PPG原始序列
        x_proj = self.input_proj(x)           # (batch, seq_len, d_model)
        x_enc = self.pos_enc(x_proj)         # 加入位置编码
        x_feats = self.encoder(x_enc)        # Transformer Encoder提取特征 (batch, seq_len, d_model)
        return x_feats
