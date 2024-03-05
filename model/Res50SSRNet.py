
from torchvision import models

from model.layers import *
from model.SMSEmodels import *

import torch.nn.functional as F
from util.misc import *

# model_path = '../PyTorch_Pretrained/resnet50-19c8e357.pth'

class ResSSR(nn.Module):
    def __init__(self,  pretrained=True):
        super(ResSSR, self).__init__()

        self.de_pred = nn.Sequential(Conv2d(128, 32, 1, same_padding=True, NL='relu'),
                                     Conv2d(32, 3, 1, same_padding=True, NL='relu'))

        initialize_weights(self.modules())

        res = models.resnet50(pretrained=pretrained)
        # pre_wts = torch.load(model_path)
        # res.load_state_dict(pre_wts)
        self.fronthead = nn.Sequential(
            res.conv1, res.bn1, res.relu
        )
        self.frontend = nn.Sequential(
            res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)        
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

        self.up1 = nn.Sequential(
            FGUpsampling2(inplanes=256, outplanes=256, scale=4, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )
        self.up5 = UNetcat(256, 64)
        self.conv_channel = nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False)
        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

    def forward(self,x):

        
        x1 = self.fronthead(x) #64, 112,112

        x = self.frontend(x1) #512,28,28

        x = self.own_reslayer_3(x) #1024,28,28

        x = self.conv_channel(x)
        x = self.up1(x)
        x = self.up5(x, x1)

        count = self.level_class(x)

        x = self.de_pred(x)

        x = F.upsample(x,scale_factor=2)
        return x, count

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)   


def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)  


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class levelclass(nn.Module):
    def __init__(self,input_channel=512,pred_shapes=[1, 128, 1, 1]):
        super(levelclass, self).__init__()

        self.pred_shape = pred_shapes

        self.pred_vectors = nn.Parameter(torch.rand(self.pred_shape),
                                          requires_grad=True)

        self.inputlayer=nn.Sequential(nn.AdaptiveAvgPool2d((32,32)), #fax feature size
                                nn.Conv2d(input_channel, 1, 1, stride=1, padding=0),#down channel
                                # nn.InstanceNorm2d(128),
                                nn.ReLU(),
                                )


        self.count_layer = nn.Conv2d(32, 1, 32, stride=1, padding=0)

    def forward(self, input):
        x = self.inputlayer(input)
        count = F.conv2d(input=x, weight=self.pred_vectors)

        return count
if __name__ == '__main__':
    torch.cuda.set_device(0)
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input = torch.randn(1, 3, 224, 224).cuda()
    model = ResSSR().cuda()
    output = model(input)
    print(input.size())
    print(output.size())
    # print(attention.size())