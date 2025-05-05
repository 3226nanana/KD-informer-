import torch
import torch.nn as nn

class SEResNetFeatureExtractor(nn.Module):
    """
    形态学特征融合模块（SE-ResNet），将PPG形态学特征向量嵌入到模型空间。
    输入: 形状(batch, in_features)的特征向量，输出: (batch, embed_dim)的嵌入表示。
    """
    def __init__(self, in_features: int, embed_dim: int, hidden_dim: int = 64, reduction: int = 16):
        super(SEResNetFeatureExtractor, self).__init__()
        self.fc_in = nn.Linear(in_features, hidden_dim)
        self.fc_res1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_res2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        # Squeeze-and-Excitation (SE) 注意力层
        self.se_fc1 = nn.Linear(hidden_dim, max(hidden_dim // reduction, 1))
        self.se_fc2 = nn.Linear(max(hidden_dim // reduction, 1), hidden_dim)
        # 输出层：映射到embed_dim，与Transformer输出维度对齐
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        out = self.relu(self.fc_in(x))           # 初始全连接映射 (batch, hidden_dim)
        # 残差块：两层FC并加入跳跃连接
        residual = out
        out_block = self.relu(self.fc_res1(out))
        out_block = self.fc_res2(out_block)
        out = self.relu(residual + out_block)   # 残差相加后激活
        # SE模块：全局信息获取并缩放特征
        w = self.relu(self.se_fc1(out))
        w = torch.sigmoid(self.se_fc2(w))
        out = out * w                           # 通道加权
        out_embed = self.fc_out(out)            # 映射到embed_dim
        return out_embed
