# mine_network.py
"""
定义 MINE 网络结构，支持 Float 和 Index 输入
"""
import torch
import torch.nn as nn

def _build_mlp(input_dim, hidden_dims, output_dim=1, activation="relu", batch_norm=True, dropout=0.1):
    """辅助函数构建 MLP 层"""
    layers = []
    prev_dim = input_dim
    for i, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim)) # 输入是 [N, C]

        # 激活函数
        if activation.lower() == 'relu':
            layers.append(nn.ReLU())
        elif activation.lower() == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation.lower() == 'elu':
            layers.append(nn.ELU())
        else:
            layers.append(nn.ReLU()) # 默认ReLU

        if dropout > 0 and i < len(hidden_dims) - 1: # 通常不在最后一层隐藏层后加dropout
            layers.append(nn.Dropout(dropout))

        prev_dim = dim

    # 输出层
    network = nn.Sequential(*layers)
    output_layer = nn.Linear(prev_dim, output_dim)

    # 初始化输出层权重以稳定训练早期
    nn.init.normal_(output_layer.weight, std=0.01)
    nn.init.constant_(output_layer.bias, 0)

    return network, output_layer


class MINENetworkFloat(nn.Module):
    """MINE网络 - 处理 float 类型的 x 和 y 输入"""

    def __init__(self, x_dim, y_dim, hidden_dims=[512, 512, 256],
                 activation="relu", batch_norm=True, dropout=0.1):
        """
        初始化 Float MINE 网络

        参数:
            x_dim: x 特征维度
            y_dim: y 特征维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            batch_norm: 是否使用批归一化
            dropout: Dropout 率
        """
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        mlp_input_dim = x_dim + y_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )

    def forward(self, x, y):
        """
        前向传播

        参数:
            x: float 类型输入, 形状 [batch_size, seq_len, x_dim] 或 [batch_size, x_dim]
            y: float 类型输入, 形状 [batch_size, seq_len, y_dim] 或 [batch_size, y_dim]

        返回:
            网络输出 T(x, y)，形状 [batch_size * seq_len, 1] 或 [batch_size, 1]
        """
        # 统一处理输入形状，reshape 成 [N, D]
        if x.ndim == 3: # [B, S, Dx]
            B, S, Dx = x.shape
            N = B * S
            x_flat = x.reshape(N, Dx)
            if y.ndim == 3:
                assert y.shape[0] == B and y.shape[1] == S, "y shape mismatch"
                y_flat = y.reshape(N, self.y_dim)
            else: # y is [B, Dy]? This case seems less likely if x is [B,S,Dx]
                raise ValueError("Shape mismatch: x is 3D but y is not")
        elif x.ndim == 2: # [B, Dx]
            N = x.shape[0]
            x_flat = x
            if y.ndim == 2:
                 assert y.shape[0] == N, "y shape mismatch"
                 y_flat = y
            else:
                raise ValueError("Shape mismatch: x is 2D but y is not")
        else:
             raise ValueError(f"Unsupported x dimension: {x.ndim}")


        # 拼接特征
        combined = torch.cat([x_flat, y_flat], dim=1) # [N, x_dim + y_dim]

        # 通过 MLP 网络
        features = self.network(combined) # [N, hidden_dims[-1]]

        # 通过输出层
        output = self.output_layer(features) # [N, 1]
        return output


class MINENetworkIndex(nn.Module):
    """MINE网络 - 处理 long (index) 类型的 x 和 float 类型的 y 输入"""

    def __init__(self, vocab_size, embedding_dim, y_dim, hidden_dims=[512, 512, 256],
                 activation="relu", batch_norm=True, dropout=0.1):
        """
        初始化 Index MINE 网络

        参数:
            vocab_size: x 的词汇表大小 (用于 Embedding)
            embedding_dim: x 的 Embedding 维度
            y_dim: y 特征维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            batch_norm: 是否使用批归一化
            dropout: Dropout 率
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.y_dim = y_dim

        # Embedding 层处理 x
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # MLP 输入维度是 embedding 输出维度 + y 的维度
        mlp_input_dim = embedding_dim + y_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )

    def forward(self, x, y):
        """
        前向传播

        参数:
            x: long 类型输入 (索引), 形状 [batch_size, seq_len] 或 [batch_size]
            y: float 类型输入, 形状 [batch_size, seq_len, y_dim] 或 [batch_size, y_dim]

        返回:
            网络输出 T(x, y)，形状 [batch_size * seq_len, 1] 或 [batch_size, 1]
        """
        # 统一处理输入形状，reshape 成 [N] for x, [N, Dy] for y
        if x.ndim == 2: # [B, S]
            B, S = x.shape
            N = B * S
            x_flat = x.reshape(N)
            if y.ndim == 3:
                assert y.shape[0] == B and y.shape[1] == S, "y shape mismatch"
                y_flat = y.reshape(N, self.y_dim)
            else:
                raise ValueError("Shape mismatch: x is 2D but y is not")
        elif x.ndim == 1: # [B]
            N = x.shape[0]
            x_flat = x
            if y.ndim == 2:
                assert y.shape[0] == N, "y shape mismatch"
                y_flat = y
            else:
                 raise ValueError("Shape mismatch: x is 1D but y is not")
        else:
             raise ValueError(f"Unsupported x dimension: {x.ndim}")

        # 将 x 索引通过 Embedding 层
        x_embedded = self.embedding(x_flat) # [N, embedding_dim]

        # 拼接嵌入后的 x 和 y
        combined = torch.cat([x_embedded, y_flat], dim=1) # [N, embedding_dim + y_dim]

        # 通过 MLP 网络
        features = self.network(combined) # [N, hidden_dims[-1]]

        # 通过输出层
        output = self.output_layer(features) # [N, 1]
        return output