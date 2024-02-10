import torch
from torch import nn
import torch.nn.functional as F


class SingleCnnGru(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            n_channels,
            dropout,
            kernel_size,
            flatten_method):
        super(SingleCnnGru, self).__init__()
        if n_channels is None:
            n_channels = [32, 64]
        self.flatten_method = flatten_method
        self.batch_norm = nn.BatchNorm1d(16)

        # padding = (kernel_size - 1) / 2
        self.cnn_channel = CNNBlock(n_inputs, n_channels, kernel_size=kernel_size, dropout=dropout)

        self.gru1 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=16, batch_first=True)

        # Dense layer with softmax for classification
        self.fc = nn.Linear(16, n_outputs)  # num_classes needs to be defined based on your dataset

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn_channel(x)

        x = x.transpose(1, 2)

        # GRU layers
        output, _ = self.gru1(x)
        output, _ = self.gru2(output)

        output = output.transpose(1, 2)
        # Flatten the output based on the specified method
        if self.flatten_method == "last":
            flattened = output[:, :, -1]
        elif self.flatten_method == "mean":
            flattened = output.mean(dim=-1)
        elif self.flatten_method == "max":
            flattened = output.max(dim=-1)[0]
        else:
            raise ValueError("Invalid flatten method")

        # Batch Normalization
        normalized = self.batch_norm(flattened)

        # Dense layer with softmax
        out = self.fc(normalized)

        return out


class CNNBlock(nn.Module):
    def __init__(self, n_inputs, n_channels, kernel_size, dropout):
        super(CNNBlock, self).__init__()

        # Convolutional block for each channel
        self.conv_block_channel = nn.Sequential(
            Conv1dSamePadding(n_inputs, n_channels[0], kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm1d(n_channels[0]),
            Conv1dSamePadding(n_channels[0], n_channels[1], kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        # Apply convolutional blocks to the input tensor
        return self.conv_block_channel(x)


class Conv1dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv1dSamePadding, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # 计算所需的总填充量
        if x.size(2) % self.stride == 0:
            total_padding = max(self.kernel_size - self.stride, 0)
        else:
            total_padding = max(self.kernel_size - (x.size(2) % self.stride), 0)

        # 将填充均匀地分配到两侧
        padding_left = total_padding // 2
        padding_right = total_padding - padding_left

        # 应用填充
        x_padded = F.pad(x, (padding_left, padding_right), "constant", 0)

        # 应用卷积
        return self.conv(x_padded)
