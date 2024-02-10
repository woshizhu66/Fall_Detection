"""
This ResNet implementation is modified from https://github.com/hsd1503/resnet1d Shenda Hong, Oct 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=True):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=bias)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do,
                 is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=not use_bn)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1D2(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, n_block, groups=1, downsample_gap=2,
                 increasefilter_gap=4, use_bn=True, use_do=True):
        super(ResNet1D2, self).__init__()
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increase_filter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters,
                                                kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            is_first_block = i_block == 0
            # downsample at every self.downsample_gap blocks
            downsample = i_block % self.downsample_gap == 1
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increase_filter_gap))
                if (i_block % self.increase_filter_gap == 0) and (not is_first_block):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        # first conv
        x = self.first_block_conv(x)
        if self.use_bn:
            x = self.first_block_bn(x)
        x = self.first_block_relu(x)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            x = self.basicblock_list[i_block](x)

        # final prediction
        if self.use_bn:
            x = self.final_bn(x)
        x = self.final_relu(x)

        return x


class MSResNet1D(nn.Module):
    def __init__(self, resnets: nn.ModuleDict,flatten_method,
                 in_channels, in_resnet_channels, n_outputs, first_kernel, drop_rate=0.5):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv1d(in_channels, in_resnet_channels, kernel_size=first_kernel, padding='same'),
            nn.BatchNorm1d(in_resnet_channels),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        self.flatten_method = flatten_method
        self.resnets = resnets
        self.fc = nn.Linear(384, n_outputs)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.first_block(x)
        x_scales = [
            self.resnets[key](x)
            for key in self.resnets.keys()
        ]
        x_scales = torch.cat(x_scales, dim=1)
        # Flatten the output based on the specified method
        if self.flatten_method == "mean":
            flattened = x_scales.mean(dim=-1)
        elif self.flatten_method == "max":
            flattened = x_scales.max(dim=-1)[0]

        # Dense layer with softmax
        out = self.fc(flattened)

        return out


# if __name__ == '__main__':
#     model = MSResNet1D(
#         resnets=nn.ModuleDict({
#             'resnet3': ResNet1D(in_channels=64, base_filters=64, kernel_size=3, n_block=6, stride=2),
#             'resnet5': ResNet1D(in_channels=64, base_filters=64, kernel_size=5, n_block=6, stride=2),
#             'resnet7': ResNet1D(in_channels=64, base_filters=64, kernel_size=7, n_block=6, stride=2),
#         }),
#         in_channels=3,
#         in_resnet_channels=64,
#         first_kernel=7
#     )
#     '''
#     in_channels: dim of input, the same as n_channel
#     base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
#     kernel_size: width of kernel
#     stride: stride of kernel moving
#     groups: set larget to 1 as ResNeXt
#     n_block: number of blocks
#     '''
#     data = torch.ones([8, 3, 128])
#     output = model(data)
#     _ = 1