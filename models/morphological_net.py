import torch
import torch.nn as nn

class SEResNetFeatureExtractor(nn.Module):
    """
    形态学特征融合模块（SE-ResNet）：将提取的PPG形态学特征向量融合到模型中。
    输入一个形态学特征向量，经过残差网络和SE注意力机制，输出一个embedding向量。
    """
    def __init__(self, in_features: int, embed_dim: int, hidden_dim: int = 64, reduction: int = 16):
        super(SEResNetFeatureExtractor, self).__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        # 输入层：将原始特征维度映射到隐藏维度
        self.fc_in = nn.Linear(in_features, hidden_dim)
        # 残差块的两层全连接
        self.fc_res1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_res2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        # Squeeze-and-Excitation (SE) 注意力层
        self.se_fc1 = nn.Linear(hidden_dim, max(hidden_dim // reduction, 1))
        self.se_fc2 = nn.Linear(max(hidden_dim // reduction, 1), hidden_dim)
        # 输出投影层：将融合后的隐藏表示映射到embed_dim，与Informer输出维度对齐
        self.fc_out = nn.Linear(hidden_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: 张量，形状 (batch, in_features)，单个PPG片段提取的形态学特征向量。
        输出:
            张量，形状 (batch, embed_dim)，融合后的特征嵌入向量。
        """
        # 1. 初始全连接映射到隐藏维度并激活
        out = self.relu(self.fc_in(x))           # (batch, hidden_dim)
        # 2. 残差块：两层全连接，带跳跃连接
        residual = out                          # 保存残差
        out_block = self.relu(self.fc_res1(out))# 第一层ReLU激活
        out_block = self.fc_res2(out_block)     # 第二层线性
        out = self.relu(residual + out_block)   # 残差相加后激活 (ResNet单元)
        # 3. Squeeze-and-Excitation注意力：
        # 压缩(squeeze)：全局信息，通过全连接降维
        w = self.relu(self.se_fc1(out))         # (batch, hidden_dim//reduction)
        w = torch.sigmoid(self.se_fc2(w))       # (batch, hidden_dim)，得到每个特征通道的权重
        # 激发(excitation)：按权重缩放隐藏表示
        out = out * w                           # (batch, hidden_dim)
        # 4. 输出层：映射到所需的嵌入维度
        out_embed = self.fc_out(out)            # (batch, embed_dim)
        return out_embed