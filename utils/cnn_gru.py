import torch
from torch import nn
import torch.nn.functional as F

class CnnGru(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            n_channels,
            dropout,
            flatten_method):
        super(CnnGru, self).__init__()
        if n_channels is None:
            n_channels = [64, 128]
        self.flatten_method = flatten_method
        self.batch_norm = nn.BatchNorm1d(64)

        # padding = (kernel_size - 1) / 2
        self.cnn_channel1 = CNNBlock(n_inputs, n_channels, kernel_size=3, dropout=dropout)
        self.cnn_channel2 = CNNBlock(n_inputs, n_channels, kernel_size=5, dropout=dropout)
        self.cnn_channel3 = CNNBlock(n_inputs, n_channels, kernel_size=7, dropout=dropout)

        self.gru1 = nn.GRU(input_size=384, hidden_size=128, batch_first=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, batch_first=True)

        # Dense layer with softmax for classification
        self.fc = nn.Linear(64, n_outputs)  # num_classes needs to be defined based on your dataset

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.cnn_channel1(x)
        x2 = self.cnn_channel2(x)
        x3 = self.cnn_channel3(x)

        combined_out = torch.cat([x1, x2, x3], dim=1)

        combined_out = combined_out.transpose(1, 2)

        # GRU layers
        output, _ = self.gru1(combined_out)
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