import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.utils.spectral_norm(nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, upsample=False):
        super(Generator, self).__init__()
        self.upsample = upsample
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0, upsample=False),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1, self.upsample),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1, self.upsample),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1, self.upsample),  # img: 32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        if self.upsample:
            self.last_conv = nn.Conv2d(features_g * 2, channels_img, 3, 1, 1)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, upsample):
        if upsample:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

    def forward(self, x):
        if self.upsample:
            x = self.net[0](x)
            size = 8
            for idx, layer in enumerate(self.net[1:4]):
                x = torch.nn.functional.interpolate(x, size=(size, size), mode='nearest')
                x = layer(x)
                size *= 2
            x = self.net[-2](x)
            x = self.net[-1](x)  # Tanh
            return x
        else:
            return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    crit = Critic(in_channels, 8)
    assert crit(x).shape == (N, 1, 1, 1), "Critic test failed"
    gen = Generator(noise_dim, in_channels, 8, upsample=True)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


# test()
