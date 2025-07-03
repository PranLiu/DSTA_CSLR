import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from modules.attention import DynamicTemporalAttention
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Get_Correlation(nn.Module):
    def __init__(self, channels, save_weight_map=False):
        super().__init__()
        self.save_weight_map = save_weight_map
        reduction_channel = channels//16
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,1,1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,2,2), dilation=(1,2,2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,3,3), dilation=(1,3,3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):

        x2 = self.down_conv2(x)
        affinities = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2))  # repeat the last frame
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2))  # repeat the first frame
        features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2), F.sigmoid(affinities)-0.5 )* self.weights2[0] + \
            torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2), F.sigmoid(affinities2)-0.5 ) * self.weights2[1]

        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x)*self.weights[0] + self.spatial_aggregation2(x)*self.weights[1] \
                    + self.spatial_aggregation3(x)*self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)
        if self.save_weight_map:
            torch.save(F.sigmoid(aggregated_x)-0.5, "./weight_map.pth")
            return features * (F.sigmoid(aggregated_x)-0.5)
        else:
            return features * (F.sigmoid(aggregated_x)-0.5)


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, attention_params=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # 使用配置文件中的注意力参数，如果没有则使用默认值
        max_window_size = 9
        kernel_sizes = [3, 5, 7]
        reduction_ratio = 16

        if attention_params is not None:
            try:
                max_window_size = attention_params.get('max_window_size', 9)
                kernel_sizes = attention_params.get('kernel_sizes', [3, 5, 7])
                reduction_ratio = attention_params.get('reduction_ratio', 16)
                print(f"Using attention parameters: max_window_size={max_window_size}, kernel_sizes={kernel_sizes}, reduction_ratio={reduction_ratio}")
            except Exception as e:
                print(f"Error parsing attention parameters: {e}")
                print(f"Using default attention parameters")

        self.corr1 = DynamicTemporalAttention(
            channels=self.inplanes,
            max_window_size=max_window_size,
            kernel_sizes=kernel_sizes,
            reduction_ratio=reduction_ratio
        )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr2 = DynamicTemporalAttention(
            channels=self.inplanes,
            max_window_size=max_window_size,
            kernel_sizes=kernel_sizes,
            reduction_ratio=reduction_ratio
        )
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr3 = DynamicTemporalAttention(
            channels=self.inplanes,
            max_window_size=max_window_size,
            kernel_sizes=kernel_sizes,
            reduction_ratio=reduction_ratio
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = x + self.corr1(x) * self.alpha[0]
        x = self.layer3(x)
        x = x + self.corr2(x) * self.alpha[1]
        x = self.layer4(x)
        x = x + self.corr3(x) * self.alpha[2]
        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:]) #bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #bt,c
        x = self.fc(x) #bt,c

        return x

def resnet18(attention_params=None, **kwargs):
    """Constructs a ResNet-18 based model.

    Args:
        attention_params: 动态时间注意力机制的参数
        **kwargs: 其他参数
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], attention_params=attention_params, **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    # 只加载基础ResNet层的权重，不加载注意力模块的权重
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and not k.startswith('corr')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def test():
    net = resnet18()
    y = net(torch.randn(1,3,224,224))
    print(y.size())

#test()