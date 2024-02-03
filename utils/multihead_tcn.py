import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadTCN(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            num_channels,
            kernel_size1,
            kernel_size2,
            dropout,
            flatten_method="last"):
        """
        Initialize the TCN model.

        Args:
            input_size (int)
            output_size (int)
            num_channels (list)
            kernel_size (int)
            dropout (float)
            flatten_method (str, optional): Method to flatten the TCN outputs. "last", "mean", or "max".
                Defaults to "last".
        """
        super(MultiHeadTCN, self).__init__()

        # Initialize the TemporalConvNet
        self.m_tcn = MultiHeadTemporalConvNet(input_size, num_channels, kernel_size1, kernel_size2, dropout=dropout)

        # Fully connected layer for final output
        self.fc = nn.Linear(num_channels[-1], output_size)

        # Check if the provided flatten_method is valid
        assert flatten_method in ["last", "mean", "max"], "Invalid flatten method"  # 检查参数的值
        self.flatten_method = flatten_method

    def forward(self, x):
        """
        Forward pass through the TCN model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, num_channels).

        Returns:
            output (Tensor): Output tensor after passing through the TCN layers and the final fully connected layer.
        """
        # Transpose the input tensor to match ths shape (batch_size, num_channels, sequence_length)
        x = x.transpose(1, 2)
        features = self.m_tcn(x)

        # Apply the chosen flatten method
        if self.flatten_method == "last":
            features = features[:, :, -1]
        elif self.flatten_method == "mean":
            features = features.mean(dim=-1)
        elif self.flatten_method == "max":
            features = features.max(dim=-1)[0]

        # output.shape: [batch_size, num_classes]
        output = self.fc(features)
        return output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class MultiHeadTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size1, kernel_size2, stride, dilation, dropout=0.2):
        super(MultiHeadTemporalBlock, self).__init__()
        padding1 = (kernel_size1 - 1) * dilation
        padding2 = (kernel_size2 - 1) * dilation
        half_n_outputs = n_outputs // 2
        # First convolutional block
        self.block1 = nn.Sequential(
            nn.Conv1d(n_inputs, half_n_outputs, kernel_size1, stride=stride, padding=padding1, dilation=dilation),
            Chomp1d(padding1),
            nn.BatchNorm1d(half_n_outputs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(half_n_outputs, half_n_outputs, kernel_size1, stride=stride, padding=padding1, dilation=dilation),
            Chomp1d(padding1),
            nn.BatchNorm1d(half_n_outputs),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Second convolutional block
        self.block2 = nn.Sequential(
            nn.Conv1d(n_inputs, half_n_outputs, kernel_size2, stride=stride, padding=padding2, dilation=dilation),
            Chomp1d(padding2),
            nn.BatchNorm1d(half_n_outputs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(half_n_outputs, half_n_outputs, kernel_size2, stride=stride, padding=padding2, dilation=dilation),
            Chomp1d(padding2),
            nn.BatchNorm1d(half_n_outputs),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # Process the input through both blocks in parallel
        out1 = self.block1(x)
        out2 = self.block2(x)
        combined_out = torch.cat([out1, out2], dim=1)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(combined_out + res)


class MultiHeadTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size1, kernel_size2, dropout):
        """
        Initialize the TemporalConvNet.

        Args:
            num_inputs (int)
            num_channels (list)
            kernel_size (int, optional)
            dropout (float, optional)
        """
        super(MultiHeadTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [MultiHeadTemporalBlock(in_channels, out_channels, kernel_size1, kernel_size2, stride=1,
                                              dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, channels, sequence_length)
        """
        return self.network(x)
