import argparse
from thop import profile
from thop import clever_format
from models import *
import numpy as np
from model.layers import *

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512,normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512,normalize=False, dropout=0.5)
        # self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.count = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),
                                nn.Conv2d(512, 1, 8, stride=1, padding=0)
        )

        self.up1 = UNetDUp(512, 512, dropout=0.5)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetDUp(1024, 512, dropout=0.5)
        self.up5 = UNetDUp(1024, 256)
        self.up6 = UNetDUp(512, 128)
        self.up7 = UNetDUp(256, 64)

        self.final = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 32, 4, padding=1),
            nn.Conv2d(32,1,1),
            # FBUpsampling(inplanes=32, outplanes=out_channels, scale=2),
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
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        count = self.count(d6)
        u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d6)
        # u3 = self.up3(u1, d5)
        u2 = self.up4(u1, d4)
        u3 = self.up5(u2, d3)
        u4 = self.up6(u3, d2)
        u5 = self.up7(u4, d1)

        return self.final(u5),count
class FMMG_nearest(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(FMMG_nearest, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512,normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512,normalize=False, dropout=0.5)
        # self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.count = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),
                                nn.Conv2d(512, 1, 8, stride=1, padding=0)
        )

        self.up1 = UNetnearUp(512, 512, dropout=0.5)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetnearUp(1024, 512, dropout=0.5)
        self.up5 = UNetnearUp(1024, 256)
        self.up6 = UNetnearUp(512, 128)
        self.up7 = UNetnearUp(256, 64)


        self.final = nn.Sequential(
            nn.Conv2d(512,128,1),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Upsample(scale_factor=32),
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
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        count = self.count(d6)
        # u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d6)
        # u3 = self.up3(u1, d5)
        # u4 = self.up4(u1, d4)
        # u5 = self.up5(u4, d3)
        # u6 = self.up6(u5, d2)
        # u7 = self.up7(u6, d1)

        return self.final(d6),count
class FMMG_bilinear(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(FMMG_bilinear, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512,normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512,normalize=False, dropout=0.5)
        # self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.count = nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),
                                nn.Conv2d(512, 1, 8, stride=1, padding=0)
        )

        self.up1 = UNetbilineUp(512, 512, dropout=0.5)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetbilineUp(1024, 512, dropout=0.5)
        self.up5 = UNetbilineUp(1024, 256)
        self.up6 = UNetbilineUp(512, 128)
        self.up7 = UNetbilineUp(256, 64)

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
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        count = self.count(d6)
        u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d6)
        # u3 = self.up3(u1, d5)
        u4 = self.up4(u1, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7),count

class SSRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SSRNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

        # self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up1 = nn.Sequential(
            SSRupsampling2(inplanes=512, outplanes=256, scale=32, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 256)
        # self.up4 = UNetUp(512, 128)
        self.up5 = UNetcat(256, 64)

        self.final = nn.Sequential(

            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            # nn.Tanh(),
            # SSRupsampling2(inplanes=128, outplanes=out_channels, scale=2, k_size=1, pad=0),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u1, d2)
        u5 = self.up5(u1, d1)
        # count = self.count(u5)
        count = self.level_class(u5)
        return self.final(u5),count

class levelclass(nn.Module):
    def __init__(self,input_channel=512,pred_shapes=[1, 128, 1, 1]):
        super(levelclass, self).__init__()

        self.pred_shape = pred_shapes

        self.pred_vectors = nn.Parameter(torch.rand(self.pred_shape),
                                          requires_grad=True)

        self.inputlayer=nn.Sequential(nn.AdaptiveAvgPool2d((32,32)), #fax feature size
                                nn.Conv2d(input_channel, 1, 1, stride=1, padding=0),#down channel
                                nn.InstanceNorm2d(128),
                                nn.LeakyReLU(0.2),
                                )


        self.count_layer = nn.Conv2d(32, 1, 32, stride=1, padding=0)

    def forward(self, input):
        x = self.inputlayer(input)
        count = F.conv2d(input=x, weight=self.pred_vectors)

        return count
#SSRNet
class SMSEnet_N(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SMSEnet_N, self).__init__()#SSRNet

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

        self.up1 = UNetnearUp(512, 512, dropout=0.5)
        self.up2 = UNetnearUp(1024, 512, dropout=0.5)
        self.up3 = UNetnearUp(1024, 256)
        self.up4 = UNetnearUp(512, 128)
        self.up5 = UNetnearUp(256, 64)
        # self.up5 = UNetcat(256, 64)
        # self.up1 = nn.Sequential(
        #     SSRupsampling2(inplanes=512, outplanes=256, scale=32, k_size=1, pad=0),
        #     nn.ReLU(inplace=True),
        # )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            # nn.Tanh(),
            # SSRupsampling2(inplanes=128, outplanes=out_channels, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        # u1 = self.up1(u5)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u1, d2)
        # u5 = self.up5(u1, d1)
        # count = self.count(u5)
        count = self.level_class(u5)
        return self.final(u5),count

class SMSEnet_B(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SMSEnet_B, self).__init__()#SSRNet

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

        self.up1 = UNetbilineUp(512, 512, dropout=0.5)
        self.up2 = UNetbilineUp(1024, 512, dropout=0.5)
        self.up3 = UNetbilineUp(1024, 256)
        self.up4 = UNetbilineUp(512, 128)
        self.up5 = UNetbilineUp(256, 64)
        # self.up5 = UNetcat(256, 64)
        # self.up1 = nn.Sequential(
        #     SSRupsampling2(inplanes=512, outplanes=256, scale=32, k_size=1, pad=0),
        #     nn.ReLU(inplace=True),
        # )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            # nn.Tanh(),
            # SSRupsampling2(inplanes=128, outplanes=out_channels, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        # u1 = self.up1(u5)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u1, d2)
        # u5 = self.up5(u1, d1)
        # count = self.count(u5)
        count = self.level_class(u5)
        return self.final(u5),count

class SSRNetx16(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SSRNetx16, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

        # self.up1 = UNetDDUp2(512, 512, dropout=0.5)
        # self.up2 = UNetDDUp2(1024, 512, dropout=0.5)
        # self.up3 = UNetDDUp2(1024, 256)
        # self.up4 = UNetDDUp2(512, 128)
        self.up5 = UNetUp(256, 64)
        # self.up5 = UNetcat(256, 64)
        self.up1 = nn.Sequential(
            # nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Upsample(scale_factor=16, mode='bilinear'),
            # nn.ConvTranspose2d(512, 256, 4, 16,output_padding=12, bias=False),
            # SSRupsampling2(inplanes=512, outplanes=256, scale=16,k_size=1,  pad=0),
            SSRupsampling2(inplanes=512, outplanes=256, scale=16, k_size=1, pad=0),

            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            # nn.Tanh(),
            # SSRupsampling2(inplanes=128, outplanes=out_channels, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u3, d2)
        # u5 = self.up5(u4, d1)
        u1 = self.up1(d6)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u1, d2)
        u5 = self.up5(u1, d1)
        # count = self.count(u5)
        count = self.level_class(u5)
        return self.final(u5),count


class SSRNetx8(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SSRNetx8, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

        # self.up1 = UNetDDUp2(512, 512, dropout=0.5)
        # self.up2 = UNetDDUp2(1024, 512, dropout=0.5)
        # self.up3 = UNetDDUp2(1024, 256)
        # self.up4 = UNetDDUp2(512, 128)
        self.up5 = UNetUp(256, 64)
        # self.up5 = UNetcat(256, 64)
        self.up1 = nn.Sequential(
            SSRupsampling2(inplanes=512, outplanes=256, scale=8, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=4, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
            SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            # nn.Tanh(),
            # SSRupsampling2(inplanes=128, outplanes=out_channels, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u3, d2)
        # u5 = self.up5(u4, d1)
        u1 = self.up1(d6)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u1, d2)
        u5 = self.up5(u1, d1)
        # count = self.count(u5)
        count = self.level_class(u5)
        return self.final(u5), count

class SSRNetx4(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SSRNetx4, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

        # self.up1 = UNetDDUp2(512, 512, dropout=0.5)
        # self.up2 = UNetDDUp2(1024, 512, dropout=0.5)
        # self.up3 = UNetDDUp2(1024, 256)
        # self.up4 = UNetDDUp2(512, 128)
        self.up5 = UNetUp(256, 64)
        # self.up5 = UNetcat(256, 64)
        self.up1 = nn.Sequential(
            SSRupsampling2(inplanes=512, outplanes=256, scale=4, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            SSRupsampling2(inplanes=256, outplanes=256, scale=4, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            # nn.Tanh(),
            # SSRupsampling2(inplanes=128, outplanes=out_channels, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u3, d2)
        # u5 = self.up5(u4, d1)
        u1 = self.up1(d6)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u1, d2)
        u5 = self.up5(u1, d1)
        # count = self.count(u5)
        count = self.level_class(u5)
        return self.final(u5), count

class SSRNetx2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SSRNetx2, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])

        # self.up1 = UNetDDUp2(512, 512, dropout=0.5)
        # self.up2 = UNetDDUp2(1024, 512, dropout=0.5)
        # self.up3 = UNetDDUp2(1024, 256)
        # self.up4 = UNetDDUp2(512, 128)
        self.up5 = UNetUp(256, 64)
        # self.up5 = UNetcat(256, 64)
        self.up1 = nn.Sequential(
            SSRupsampling2(inplanes=512, outplanes=256, scale=4, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            SSRupsampling2(inplanes=256, outplanes=256, scale=4, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
            # SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            # nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            # nn.Tanh(),
            # SSRupsampling2(inplanes=128, outplanes=out_channels, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u3, d2)
        # u5 = self.up5(u4, d1)
        u1 = self.up1(d6)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u1, d2)
        u5 = self.up5(u1, d1)
        # count = self.count(u5)
        count = self.level_class(u5)
        return self.final(u5), count

FMMG = GeneratorUNet()
FMMG_N=FMMG_nearest()
FMMG_B=FMMG_bilinear()
SSRNet_N=SMSEnet_N()
SSRNet_B=SMSEnet_B()

# VGG16 based generator
parser = argparse.ArgumentParser()
# * Backbone
parser.add_argument('--backbone', default='vgg16_bn', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=196608, help="dimensionality of the latent space")
parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
opt = parser.parse_args()
backbone=build_backbone(opt)
class Model(nn.Module):
    def __init__(self,backbone):
        super(Model, self).__init__()
        self.vgg = backbone
        # self.load_vgg()
        # self.amp = BackEnd()
        self.amp = QuickUP()
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
VGG16 = Model(backbone)
# U-Net
img_shape = (3, 256, 256)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.bloch1 = nn.Sequential(*block(opt.latent_dim, 128, normalize=False))
        self.bloch2 = nn.Sequential(*block(128, 256))
        self.bloch3 = nn.Sequential(*block(256, 512))
        self.bloch4 =nn.Sequential(*block(512, 1024))
        self.bloch5 = nn.Linear(1024, int(np.prod(img_shape)))
        self.tanh = nn.Tanh()
        # self.model = nn.Sequential(
        #     *block(opt.latent_dim, 128, normalize=False),
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, int(np.prod(img_shape))),
        #     nn.Tanh()
        # )

    def forward(self, z):
        img = z.view(2,3*256*256)
        img = self.bloch1(img)
        img = self.bloch2(img)
        img = self.bloch3(img)
        img = self.bloch4(img)
        img = self.bloch5(img)
        img = self.tanh(img)
        img = img.view(z.size(0), *img_shape)
        return img
GAN = Generator()

class cgan(nn.Module):
    def __init__(self):
        super(cgan, self).__init__()

        # self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(2*opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, img):
        labels=img
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((img.view(img.size(0), -1), labels.view(labels.size(0), -1)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img
cGAN = cgan()
input = torch.randn(1, 3, 256, 256)

from unetmodel import U_Net
UNet = U_Net()#GeneratorUNet()
from model.Res50 import Res50
Res50=Res50()
from model.Res50SSRNet import ResSSR
ResSSR = ResSSR()
from model.MobileCount2 import MobileCountSSR
MobileCountSSR=MobileCountSSR()

import time
def running_time():
    im=torch.randn(2, 3, 256, 256)
    t1=time.time()
    for i in range(100):
        output=UNet(im)
    t2=time.time()
    time_left=t2-t1
    print("time_left:",time_left)
    print(input.size())
    print(output.size())

if __name__ == '__main__':
    torch.cuda.set_device(0)
    flops, params = profile(UNet, inputs=(input,))  # resnet18
    print(flops, params)  # 1819066368.0 11689512.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)  # 1.819G 11.690M
    from torchstat import stat

    stat(UNet, (3, 256, 256))
    running_time()
