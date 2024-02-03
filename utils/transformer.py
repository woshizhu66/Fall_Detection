import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, n_inputs, units, dropout_rate):
        super(PositionalEmbedding, self).__init__()

        self.projection = nn.Linear(n_inputs, units)
        self.dropout = nn.Dropout(dropout_rate)

        self.position = nn.Parameter(torch.randn(1, 1, units))

        truncated_normal_(self.position, std=0.02)
        truncated_normal_(self.projection.weight, std=0.02)

        nn.init.zeros_(self.projection.bias)

    def forward(self, inputs):
        x = inputs.transpose(1, 2)
        x = self.projection(x)
        x = x + self.position
        return self.dropout(x)


def truncated_normal_(tensor, mean=0, std=0.02):
    with torch.no_grad():
        size = tensor.size()
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


class Encoder(nn.Module):
    def __init__(self, embed_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate):
        super(Encoder, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attention_dropout_rate)
        self.dense_0 = nn.Linear(embed_dim, mlp_dim)
        self.dense_1 = nn.Linear(mlp_dim, embed_dim)
        self.dropout_0 = nn.Dropout(dropout_rate)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.norm_0 = nn.LayerNorm(embed_dim)
        self.norm_1 = nn.LayerNorm(embed_dim)

    def forward(self, inputs, count):
        if count == 0:
            x = inputs.transpose(0, 1)
        else:
            x = inputs
        # 多头注意力块
        x = self.norm_0(x)
        attn_output, _ = self.mha(x, x, x)
        x = x + self.dropout_0(attn_output)

        # 前馈网络块
        y = self.norm_1(x)
        y = self.dense_0(y)
        y = F.gelu(y)
        y = self.dense_1(y)
        y = self.dropout_1(y)
        y = x + y
        return y


class Transformer(nn.Module):
    def __init__(self, n_inputs, num_layers, embed_dim, mlp_dim, num_heads, num_classes, dropout_rate,
                 attention_dropout_rate, flatten_method):
        super(Transformer, self).__init__()

        # 输入标准化（根据具体情况可能需要调整）
        self.input_norm = nn.BatchNorm1d(n_inputs)  # 假设输入已经正确地转置

        # 位置嵌入
        self.pos_embs = PositionalEmbedding(n_inputs, embed_dim, dropout_rate)

        self.flatten_method = flatten_method

        # 编码器层
        self.e_layers = nn.ModuleList([
            Encoder(embed_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate)
            for _ in range(num_layers)
        ])

        # 输出层标准化
        self.norm = nn.LayerNorm(embed_dim)

        # 最终的全连接层
        self.fc = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = self.pos_embs(x)
        count = 0
        for layer in self.e_layers:
            x = layer(x, count)
            count += 1
        x = self.norm(x)
        # print(x.shape)
        x = x.transpose(0, 1)
        # Flatten the output based on the specified method
        if self.flatten_method == "last":
            flattened = x[:, -1, :]
        elif self.flatten_method == "mean":
            flattened = x.mean(dim=1)
        elif self.flatten_method == "max":
            flattened = x.max(dim=1)[0]
        return self.fc(flattened)
