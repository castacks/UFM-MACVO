from torchvision import models as tvm
import torch
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(
        self,
        pretrained=False,
        high_res=False,
        weights=None,
        dilation=None,
        freeze_bn=True,
        anti_aliased=False,
        early_exit=False,
        amp=False,
        amp_dtype=torch.float16,
    ) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False, False, False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights=weights, replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained, replace_stride_with_dilation=dilation)

        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):

        net = self.net
        feats = {1: x}
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats[2] = x
        x = net.maxpool(x)
        x = net.layer1(x)
        feats[4] = x
        x = net.layer2(x)
        feats[8] = x
        if self.early_exit:
            return feats
        x = net.layer3(x)
        feats[16] = x
        x = net.layer4(x)
        feats[32] = x
        return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass