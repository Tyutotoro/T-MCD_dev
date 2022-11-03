import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(  #NNのconv構造作成
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

#↑↓何が違うんやろ?

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mp_conv(x)
        return x


# class Up(nn.Module):
#     def __init__(self, in_size, out_size, up_mode='upconv'):
#         super(Up, self).__init__()
#         if up_mode == 'upconv':
#             self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
#         elif up_mode == 'upsample':
#             self.up = nn.Upsample(mode='nearest', scale_factor=2)
#         self.conv = DoubleConv(in_size, out_size)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         """diff_x = x1.size()[2] - x2.size()[2]
#         diff_y = x1.size()[3] - x2.size()[3]
#         x2 = f.pad(x2, [diff_x // 2, diff_x // 2, diff_y // 2, diff_y // 2])"""
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x




class ArcfaceCoDetectionBase(nn.Module):
    def __init__(self, n_channels, up_mode='upconv'):
        super(ArcfaceCoDetectionBase, self).__init__()
        factor = 2 if up_mode == 'upsample' else 1
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.fc = nn.Linear(in_features=16,out_features=4)
        # self.up1 = Up(1024, 512 // factor, up_mode=up_mode)
        # self.up2 = Up(512, 256 // factor, up_mode=up_mode)

    def forward(self, x1, x2):
        x2_t1 = self.inc(x1)
        x2_t2 = self.inc(x2)

        x3_t1 = self.down1(x2_t1)
        x3_t2 = self.down1(x2_t2)

        x4 = torch.cat([x3_t1, x3_t2], dim=1)

        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x7 = self.down4(x6)
        xf = self.fc(x7)
        # x8 = self.up1(x7, x6)
        # x9 = self.up2(x8, x5)
        return xf
        # return (x7, x3_t2, x2_t1), (x7, x3_t2, x2_t2),(x7),(xf)