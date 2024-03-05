
import torch
import torch.nn as nn
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

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class convDU(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(9,1)
        ):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).resize(n,c,1,w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)
            # pdb.set_trace()
            # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)


        for i in range(h):
            pos = h-i-1
            if pos == h-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]
        # pdb.set_trace()
        fea = torch.cat(fea_stack, 2)
        return fea

class convLR(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(1,9)
        ):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).resize(n,c,h,1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(w):
            pos = w-i-1
            if pos == w-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]


        fea = torch.cat(fea_stack, 3)
        return fea

class SSRupsampling(nn.Module):
    #先H向扩大，再连接图像H方向上的空洞，再在W向重复操作。
    def __init__(self, inplanes, outplanes, scale,k_size=1, pad=0):
        super(SSRupsampling, self).__init__()
        self.conv_channel=nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, bias=False)
        ## new matrix
        self.conv_w = nn.Conv2d(outplanes, outplanes * scale, kernel_size=k_size, padding=pad, bias=False)
        self.conv_H = nn.Conv2d(outplanes, outplanes * scale, kernel_size=k_size, padding=pad, bias=False)
        self.scale = scale
        self.convDU = convDU2(in_out_channels=outplanes, kernel_size=(3, 3), stride=3)
        self.convLR = convLR2(in_out_channels=outplanes, kernel_size=(3, 3), stride=3)

    def forward(self, x):
        x = self.conv_channel(x)
        x = self.conv_w(x) # N,C*scale, H, W
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))
        # N, C, H*scale, W
        x_permuted = x_permuted.permute(0, 3, 2, 1)

        x_permuted = self.convLR(x_permuted)


        x_h = self.conv_H(x_permuted) # N, C*scale, H*scale,W

        x_h = x_h.permute(0, 2, 3, 1) # N, H*scale, W, C/scale

        x_h = x_h.contiguous().view(
            (N, H * self.scale, W * self.scale, int(C / (self.scale))))# N, H*scale, W*scale, C

        # N, C, H*scale, W*scale
        x = x_h.permute(0, 3, 1, 2)

        x = self.convDU(x)

        return x

class SSRupsampling2(nn.Module):
    # 先进行DULR连连接图像的空洞，然后再用用快速扩大图像，=
    def __init__(self, inplanes, outplanes, scale,k_size=1, pad=0):
        super(SSRupsampling2, self).__init__()
        self.conv_channel=nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, bias=False)
        ## new matrix
        self.conv_w = nn.Conv2d(outplanes, outplanes * scale, kernel_size=k_size, padding=pad, bias=False)
        self.conv_H = nn.Conv2d(outplanes, outplanes * scale, kernel_size=k_size, padding=pad, bias=False)
        self.scale = scale
        self.convDU = convDU2(in_out_channels=outplanes, kernel_size=(3, 1), stride=1)
        self.convLR = convLR2(in_out_channels=outplanes, kernel_size=(1, 3), stride=1)

    def forward(self, x):
        x = self.conv_channel(x)
        x = self.convDU(x)
        x = self.convLR(x)

        x = self.conv_w(x) # N,C*scale, H, W
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))
        # N, C, H*scale, W
        x_permuted = x_permuted.permute(0, 3, 2, 1)

        x_h = self.conv_H(x_permuted) # N, C*scale, H*scale,W

        x_h = x_h.permute(0, 2, 3, 1) # N, H*scale, W, C/scale

        x_h = x_h.contiguous().view(
            (N, H * self.scale, W * self.scale, int(C / (self.scale))))# N, H*scale, W*scale, C

        # N, C, H*scale, W*scale
        x = x_h.permute(0, 3, 1, 2)

        return x


class SSRupsampling3(nn.Module):
    # 先进行DULR连连接图像的空洞，然后再用用快速扩大图像，=
    def __init__(self, inplanes, outplanes, scale,k_size=1, pad=0):
        super(SSRupsampling3, self).__init__()
        self.conv_channel = nn.Conv2d(outplanes*3, outplanes, kernel_size=1, padding=0, bias=False)
        ## new matrix
        self.conv_w = nn.Conv2d(outplanes, outplanes * scale, kernel_size=k_size, padding=pad, bias=False)
        self.conv_H = nn.Conv2d(outplanes, outplanes * scale, kernel_size=k_size, padding=pad, bias=False)
        self.scale = scale
        self.convDU = convDU3(in_channels=inplanes, out_channels=outplanes, kernel_size=(3, 1), stride=1)
        self.convLR = convLR3(in_channels=inplanes, out_channels=outplanes, kernel_size=(1, 3), stride=1)
        self.conv1x1 = nn.Conv2d(inplanes, outplanes, kernel_size=k_size, padding=pad, bias=False)

    def forward(self, x):
        # 重参数化，图像增强处理
        xdu = self.convDU(x)
        xlr = self.convLR(x)
        x1 = self.conv1x1(x)
        x = torch.cat((x1,xlr,xdu),1)

        x = self.conv_channel(x)

        x = self.conv_w(x)  # N,C*scale, H, W
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))
        # N, C, H*scale, W
        x_permuted = x_permuted.permute(0, 3, 2, 1)

        x_h = self.conv_H(x_permuted)  # N, C*scale, H*scale,W

        x_h = x_h.permute(0, 2, 3, 1)  # N, H*scale, W, C/scale

        x_h = x_h.contiguous().view(
            (N, H * self.scale, W * self.scale, int(C / (self.scale))))  # N, H*scale, W*scale, C

        # N, C, H*scale, W*scale
        x = x_h.permute(0, 3, 1, 2)

        return x

class SSRupsampling4(nn.Module):
        # 先进行DULR连连接图像的空洞，然后再用用快速扩大图像，重参数化，leaky relu 4 GAN
    def __init__(self, inplanes, outplanes, scale, k_size=1, pad=0):
        super(SSRupsampling4, self).__init__()
        self.conv_channel = nn.Conv2d(outplanes * 3, outplanes, kernel_size=1, padding=0, bias=False)
            ## new matrix
        self.conv_w = nn.Conv2d(outplanes, outplanes * scale, kernel_size=k_size, padding=pad, bias=False)
        self.conv_H = nn.Conv2d(outplanes, outplanes * scale, kernel_size=k_size, padding=pad, bias=False)
        self.scale = scale
        self.convDU = convDU4(in_channels=inplanes, out_channels=outplanes, kernel_size=(3, 1), stride=1)
        self.convLR = convLR4(in_channels=inplanes, out_channels=outplanes, kernel_size=(1, 3), stride=1)
        self.conv1x1 = nn.Conv2d(inplanes, outplanes, kernel_size=k_size, padding=pad, bias=False)

    def forward(self, x):
        # 重参数化，图像增强处理
        xdu = self.convDU(x)
        xlr = self.convLR(x)
        x1 = self.conv1x1(x)
        x = torch.cat((x1, xlr, xdu), 1)

        x = self.conv_channel(x)

        x = self.conv_w(x)  # N,C*scale, H, W
        N, C, H, W = x.size()

            # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

            # N, W, H*scale, C
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))
            # N, C, H*scale, W
        x_permuted = x_permuted.permute(0, 3, 2, 1)

        x_h = self.conv_H(x_permuted)  # N, C*scale, H*scale,W

        x_h = x_h.permute(0, 2, 3, 1)  # N, H*scale, W, C/scale

        x_h = x_h.contiguous().view(
            (N, H * self.scale, W * self.scale, int(C / (self.scale))))  # N, H*scale, W*scale, C

            # N, C, H*scale, W*scale
        x = x_h.permute(0, 3, 1, 2)

        return x
class convDU2(nn.Module):

    def __init__(self, in_out_channels=2048, kernel_size=(9,1), stride=1):
        super(convDU2, self).__init__()
        self.stride=stride

        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        fea_emp = []
        fea_conv =[]
        if self.stride<=1:
            for i in range(h):
                i_fea = fea.select(2, i).resize(n, c, 1, w)
                if i == 0:
                    fea_stack.append(i_fea)
                    continue
                fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)
                # pdb.set_trace()
                # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)

            for i in range(h):
                pos = h - i - 1
                if pos == h - 1:
                    continue
                fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]
        else:
            for i in range(h):
                i_fea = fea.select(2, i).resize(n,c,1,w)
                if i ==0 or i % self.stride == 0:
                    fea_conv = i_fea
                    continue
                fea_conv = torch.cat([fea_conv, i_fea], 2)
                if (i+1) % self.stride == 0:
                    fea_stack.append(self.conv(fea_conv))
                    fea_conv = []
            if fea_conv != fea_emp:
                fea_stack.append(self.conv(fea_conv))
                fea_conv = []
                # pdb.set_trace()
                # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)
            n = len(fea_stack)
            for i in range(n):
                pos = n-i-1
                i_fea = fea_stack[pos]
                fea_stack[pos]=self.conv(i_fea)

        fea = torch.cat(fea_stack, 2)
        return fea
class convLR2(nn.Module):

    def __init__(self, in_out_channels=2048, kernel_size=(1,9), stride=1):
        super(convLR2, self).__init__()
        self.stride=stride
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )


    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        fea_emp = []
        fea_conv =[]

        if self.stride<=1:
            for i in range(w):
                i_fea = fea.select(3, i).resize(n, c, h, 1)
                if i == 0:
                    fea_stack.append(i_fea)
                    continue
                fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)

            for i in range(w):
                pos = w - i - 1
                if pos == w - 1:
                    continue
                fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]
        else:
            for i in range(w):
                i_fea = fea.select(3, i).resize(n,c,h,1)
                if i == 0 or i % self.stride == 0:
                    fea_conv = i_fea
                    continue
                fea_conv = torch.cat([fea_conv, i_fea], 3)
                if (i + 1) % self.stride == 0:
                    fea_stack.append(self.conv(fea_conv))
                    fea_conv = []
            if fea_conv != fea_emp:
                fea_stack.append(self.conv(fea_conv))
                fea_conv = []
                # pdb.set_trace()
                # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)
            n = len(fea_stack)
            for i in range(n):
                pos = n-i-1
                i_fea = fea_stack[pos]
                fea_stack[pos]=self.conv(i_fea)
        # pdb.set_trace()

        fea = torch.cat(fea_stack, 3)
        return fea

class convDU3(nn.Module):

    def __init__(self, in_channels=2048, out_channels=512, kernel_size=(9,1), stride=1):
        super(convDU3, self).__init__()
        self.stride=stride

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        fea_emp = []
        fea_conv =[]
        if self.stride<=1:
            for i in range(h):
                i_fea = fea.select(2, i).resize(n, c, 1, w)

                fea_stack.append(self.conv(i_fea))

        else:
            for i in range(h):
                i_fea = fea.select(2, i).resize(n,c,1,w)
                if i ==0 or i % self.stride == 0:
                    fea_conv = i_fea
                    continue
                fea_conv = torch.cat([fea_conv, i_fea], 2)
                if (i+1) % self.stride == 0:
                    fea_stack.append(self.conv(fea_conv))
                    fea_conv = []
            if fea_conv != fea_emp:
                fea_stack.append(self.conv(fea_conv))


        fea = torch.cat(fea_stack, 2)
        return fea
class convLR3(nn.Module):

    def __init__(self, in_channels=2048,out_channels=512, kernel_size=(1,9), stride=1):
        super(convLR3, self).__init__()
        self.stride=stride
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )


    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        fea_emp = []
        fea_conv =[]

        if self.stride<=1:
            for i in range(w):
                i_fea = fea.select(3, i).resize(n, c, h, 1)

                fea_stack.append(self.conv(i_fea))

        else:
            for i in range(w):
                i_fea = fea.select(3, i).resize(n,c,h,1)
                if i == 0 or i % self.stride == 0:
                    fea_conv = i_fea
                    continue
                fea_conv = torch.cat([fea_conv, i_fea], 3)
                if (i + 1) % self.stride == 0:
                    fea_stack.append(self.conv(fea_conv))
                    fea_conv = []
            if fea_conv != fea_emp:
                fea_stack.append(self.conv(fea_conv))
                fea_conv = []


        fea = torch.cat(fea_stack, 3)
        return fea

class convDU4(nn.Module):
# 用于4 4GAN
    def __init__(self, in_channels=2048, out_channels=512, kernel_size=(9,1), stride=1):
        super(convDU4, self).__init__()
        self.stride=stride

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.LeakyReLU(0.2)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        fea_emp = []
        fea_conv =[]
        if self.stride<=1:
            for i in range(h):
                i_fea = fea.select(2, i).resize(n, c, 1, w)

                fea_stack.append(self.conv(i_fea))

        else:
            for i in range(h):
                i_fea = fea.select(2, i).resize(n,c,1,w)
                if i ==0 or i % self.stride == 0:
                    fea_conv = i_fea
                    continue
                fea_conv = torch.cat([fea_conv, i_fea], 2)
                if (i+1) % self.stride == 0:
                    fea_stack.append(self.conv(fea_conv))
                    fea_conv = []
            if fea_conv != fea_emp:
                fea_stack.append(self.conv(fea_conv))


        fea = torch.cat(fea_stack, 2)
        return fea
class convLR4(nn.Module):

    def __init__(self, in_channels=2048,out_channels=512, kernel_size=(1,9), stride=1):
        super(convLR4, self).__init__()
        self.stride=stride
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.LeakyReLU(0.2)
            )


    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        fea_emp = []
        fea_conv =[]

        if self.stride<=1:
            for i in range(w):
                i_fea = fea.select(3, i).resize(n, c, h, 1)

                fea_stack.append(self.conv(i_fea))

        else:
            for i in range(w):
                i_fea = fea.select(3, i).resize(n,c,h,1)
                if i == 0 or i % self.stride == 0:
                    fea_conv = i_fea
                    continue
                fea_conv = torch.cat([fea_conv, i_fea], 3)
                if (i + 1) % self.stride == 0:
                    fea_stack.append(self.conv(fea_conv))
                    fea_conv = []
            if fea_conv != fea_emp:
                fea_stack.append(self.conv(fea_conv))
                fea_conv = []


        fea = torch.cat(fea_stack, 3)
        return fea
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
