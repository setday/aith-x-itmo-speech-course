import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
            ),
            nn.LeakyReLU(0.1),
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, dilation=1)
            )
        )

    def forward(self, x):
        return x + self.block(x)

class PWGGenerator(nn.Module):
    def __init__(self, in_channels=80, out_channels=1, upsample_initial_channel=512, upsample_scales=[4,4,4,4], resblock_kernel_sizes=[3,7,11], resblock_dilations=[1,3,9]):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels, upsample_initial_channel, 1)
        self.upsample_layers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, scale in enumerate(upsample_scales):
            in_ch = upsample_initial_channel
            out_ch = upsample_initial_channel

            self.upsample_layers.append(
                nn.ConvTranspose1d(in_ch, out_ch, scale*2, stride=scale, padding=scale//2)
            )
            for k, d in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResidualBlock(out_ch, k, d))
        self.output_conv = nn.Conv1d(out_ch, out_channels, 1)

    def forward(self, x):
        x = self.input_conv(x)
        for up, res in zip(self.upsample_layers, self.resblocks):
            x = F.leaky_relu(up(x), 0.1)
            x = res(x)
        x = torch.tanh(self.output_conv(x))
        return x

class PWGDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 64, 41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 256, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1024, 41, stride=4, padding=20, groups=64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, 41, stride=4, padding=20, groups=256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)
