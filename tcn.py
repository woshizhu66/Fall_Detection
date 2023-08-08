import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TCNSeg(nn.Module):
    def __init__(
            self,
            input_size,
            num_subwindows,
            num_channels,
            kernel_size,
            dropout,
    ):  # Adding num_subwindows as argument
        super(TCNSeg, self).__init__()
        self.tcn = TCN(input_size, num_channels[-1], num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], num_subwindows)

    def forward(self, x):
        # No need to transpose x
        features = self.tcn(x)
        # We don't take only the last feature vector, because we need to make a prediction for each subwindow.
        output = self.fc(features.contiguous().view(x.size(0), -1))
        return output


class TCN(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            num_channels,
            kernel_size,
            dropout,
            flatten_method="last"):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)
        assert flatten_method in ["last", "mean", "max"], "Invalid flatten method"  # 检查参数的值
        self.flatten_method = flatten_method  # 保存参数的值

    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.tcn(x)

        # 根据选择的flatten方法进行不同的操作
        if self.flatten_method == "last":
            features = features[:, :, -1]
        elif self.flatten_method == "mean":
            features = features.mean(dim=-1)
        elif self.flatten_method == "max":
            features = features.max(dim=-1)[0]  # 使用最大值

        output = self.fc(features)
        return output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
