
import torchvision.models as models
from model.layers import *
from models import *


class VGGSSR(nn.Module):
    def __init__(self):
        super(VGGSSR, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.amp = BackEnd()
        # self.size1 = size1
        # self.dmp = BackEnd()
        # self.upsample = SSRupsampling(1, 4, num_class=1)

        # self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        # self.conv_out = nn.Sequential(BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True),
        #                               BaseConv(64, 32, 1, 1, activation=None, use_bn=False),
        #                                 BaseConv(32, 1, 1, 1, activation=None, use_bn=False))

    def forward(self, input):
        size1 = input.shape[2:]
        input = self.vgg(input)
        predict, count = self.amp(input)
        # dmp_out = self.dmp(input)
        # dmp_out = input[0]
        # amp_out = self.conv_att(amp_out)
        # dmp_out = amp_out * dmp_out
        # dmp_out = self.conv_out(dmp_out)
        # dmp_out = F.upsample(dmp_out, size=size1, mode='bilinear')
        # dmp_out = self.upsample(dmp_out)
        # amp_out = F.upsample(amp_out, size=size1, mode='bilinear')
        return predict, count

    def load_vgg(self):
        # state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        vgg = models.vgg16_bn()
        pre = torch.load('D:/cc/cc_master/models/vgg16_bn-6c64b313.pth')
        vgg.load_state_dict(pre)
        state_dict =  vgg.state_dict()
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_3, conv4_3, conv5_3


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

        # self.up1 = UNetDDUp2(512, 512, dropout=0.5)
        # self.up2 = UNetDDUp2(1024, 512, dropout=0.5)
        # self.up3 = UNetDDUp2(1024, 256)
        # self.up4 = UNetUp(512, 128)
        self.up5 = UNetcat(256, 128)
        self.up1 = nn.Sequential(
            FGUpsampling3(inplanes=512, outplanes=256, scale=8, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )
        self.conv_channel = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.final = nn.Sequential(
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 3, padding=1),
            # nn.Tanh(),
            # FGUpsampling2(inplanes=128, outplanes=out_channels, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.up1(conv5_3)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        input = self.up5(input, conv2_2)
        input = self.conv_channel(input)

        count = self.level_class(input)

        return self.final(input), count


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


####  self.dupsample = SSRupsampling(256, 16, num_class=21)
class SSRupsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(SSRupsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        x_permuted = x_permuted.permute(0, 2, 1, 3) # N, H*scale, W, C/scale

        x_permuted = x_permuted.contiguous().view(
            (N, H * self.scale, W * self.scale, int(C / (self.scale * self.scale))))# N, H*scale, W*scale, C/(scale**2)

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)

        return x

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

def build_den_net(args):
    den_net = Model(args.backbone, True)
    return den_net

if __name__ == '__main__':
    input = torch.randn(2, 3, 400, 400).cuda()
    model = VGGSSR().cuda()
    output, attention = model(input)
    print(input.size())
    print(output.size())
    print(attention.size())
