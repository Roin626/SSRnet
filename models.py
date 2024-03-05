import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetDUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetDUp, self).__init__()
        layers = [
            FBupsampling(in_size, out_size, 2, num_class=out_size),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x
class UNetnearUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetnearUp, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class UNetbilineUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetbilineUp, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class UNetcat(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetcat, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        # self.FBU = FBUpsampling(in_size, out_size, scale=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # xFBU = self.FBU(x)
        # x = xDconv + xFBU
        skip_input = self.conv1(skip_input)
        x = torch.cat((x, skip_input), 1)

        return x
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.count = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),
                                nn.Conv2d(512, 1, 8, stride=1, padding=0)
        )

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        count = self.count(d8)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7),count


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),#left, right, up, down
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )


    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        out = self.model(img_input)

        return out

class Discriminator2(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator2, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),#left, right, up, down
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )


    def forward(self, img_A):
        # Concatenate image and condition image by channels to produce input
        # img_input = torch.cat((img_A, img_B), 1)
        out = self.model(img_A)

        return out
####################
#SFAN
###################
class Model(nn.Module):
    def __init__(self,backbone):
        super(Model, self).__init__()
        self.vgg = backbone
        # self.load_vgg()
        # self.amp = BackEnd()
        self.amp = NomalUP()
        self.count = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),
                                nn.Conv2d(512, 1, 8, stride=1, padding=0)
        )
        # self.size1 = size1
        # self.dmp = BackEnd()
        # self.upsample = DUpsampling(1, 4, num_class=1)

        # self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = nn.Sequential(BaseConv(32, 3, 1, 1, activation=nn.Tanh(), use_bn=False)
                                      # BaseConv(64, 32, 1, 1, activation=None, use_bn=False),
                                      #   BaseConv(16, 3, 1, 1, activation=None, use_bn=False)
                                    )

    def forward(self, input):
        size1 = input.shape[2:]
        input = self.vgg(input)

        amp_out = self.amp(input)
        # dmp_out = self.dmp(input)
        # dmp_out = input[0]
        amp_out = self.conv_out(amp_out)
        # dmp_out = amp_out * dmp_out
        # dmp_out = self.conv_out(dmp_out)
        # dmp_out = F.upsample(dmp_out, size=size1, mode='bilinear')
        # dmp_out = self.upsample(dmp_out)
        amp_out = F.upsample(amp_out, size=size1, mode='nearest')
        count = self.count(input[3])
        return amp_out, count


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
        self.dupsample4_3 = FBupsampling(512, 512, 2, num_class=512)
        self.dupsample3_2 = FBupsampling(256, 256, 2, num_class=256)
        self.dupsample2_1 = FBupsampling(128, 128, 2, num_class=128)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.Tanh(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.Tanh(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.Tanh(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.Tanh(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.Tanh(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.Tanh(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 1, 1, activation=nn.Tanh(), use_bn=True)

    def forward(self, input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.dupsample4_3(conv5_3)


        input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.dupsample3_2(input)

        input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.dupsample2_1(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input

class NomalUP(nn.Module):
    def __init__(self):
        super(NomalUP, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.dupsample4_3 = DUpsampling(512, 512, 2, num_class=512)
        # self.dupsample3_2 = DUpsampling(256, 256, 2, num_class=256)
        # self.dupsample2_1 = DUpsampling(128, 128, 2, num_class=128)

        self.conv1 = BaseConv(512, 256, 1, 1, activation=nn.Tanh(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.Tanh(), use_bn=True)

        self.conv3 = BaseConv(256, 128, 1, 1, activation=nn.Tanh(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.Tanh(), use_bn=True)

        self.conv5 = BaseConv(128, 64, 1, 1, activation=nn.Tanh(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.Tanh(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 1, 1, activation=nn.Tanh(), use_bn=True)

    def forward(self, input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input =conv5_3# self.dupsample4_3(conv5_3)


        # input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = F.upsample(input, scale_factor=2, mode='nearest')#self.dupsample3_2(input)

        # input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = F.upsample(input, scale_factor=2, mode='nearest')#self.dupsample2_1(input)

        # input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input
class QuickUP(nn.Module):
    def __init__(self):
        super(QuickUP, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.dupsample4_3 = DUpsampling(512, 512, 2, num_class=512)
        # self.dupsample3_2 = DUpsampling(256, 256, 2, num_class=256)
        # self.dupsample2_1 = DUpsampling(128, 128, 2, num_class=128)

        self.conv1 = BaseConv(512, 256, 1, 1, activation=nn.Tanh(), use_bn=True)
        # self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.Tanh(), use_bn=True)

        self.conv3 = BaseConv(256, 128, 1, 1, activation=nn.Tanh(), use_bn=True)
        # self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.Tanh(), use_bn=True)

        self.conv5 = BaseConv(128, 64, 1, 1, activation=nn.Tanh(), use_bn=True)
        # self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.Tanh(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 1, 1, activation=nn.Tanh(), use_bn=True)

    def forward(self, input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input =conv5_3# self.dupsample4_3(conv5_3)


        # input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        # input = self.conv2(input)
        # input = F.upsample(input, scale_factor=2, mode='nearest')#self.dupsample3_2(input)

        # input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        # input = self.conv4(input)
        # input = F.upsample(input, scale_factor=2, mode='nearest')#self.dupsample2_1(input)

        # input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        # input = self.conv6(input)
        input = self.conv7(input)

        return input

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


####  self.dupsample = DUpsampling(256, 16, num_class=21)
class FBupsampling(nn.Module):
    def __init__(self, inplanes,outplanes, scale, num_class=21, pad=0):
        super(FBupsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(num_class * scale * scale, outplanes, kernel_size=1, padding=pad, bias=False)
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

class FBUpsampling(nn.Module):
    def __init__(self, inplanes, outplanes, scale, pad=0):
        super(FBUpsampling, self).__init__()
        ## new matrix
        self.conv_w = nn.Conv2d(inplanes, outplanes * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x) # N,C*scale*scale, H, W
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