import torch.nn as nn
import torch
from encoder import Encoder_v, Decoder_V, Decoder_S


class VSNet(nn.Module):
    def __init__(self, in_channels, out_channels_v=1, out_channels_s=5, dd_v=1, dd_s=5, dh_v=70, dh_s=1000, dw=70,
                 vit_latent_dim=128, dim5 = 512):
        super(VSNet, self).__init__()

        self.out_channels_v = out_channels_v
        self.out_channels_s = out_channels_s
        self.vit_latent_dim = vit_latent_dim

        self.encoder = Encoder_v(in_channels, dim5=dim5)

        in_fea = self.encoder.dim5 * 1 * 1

        self.fc_v = nn.Linear(in_features=in_fea, out_features=dd_v * vit_latent_dim)
        self.fc_s = nn.Linear(in_features=in_fea, out_features=dd_s * vit_latent_dim)

        self.batch_norm_v = nn.BatchNorm2d(dd_v * vit_latent_dim)
        self.batch_norm_s = nn.BatchNorm2d(dd_s * vit_latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.decoder_v = Decoder_V(dim5=dd_v * vit_latent_dim, out_channels=out_channels_v)
        self.decoder_s = Decoder_S(dim5=dd_s * vit_latent_dim, out_channels=out_channels_s)

    def forward(self, x):
        z = self.encoder(x)

        z = z.view(z.shape[0], -1)

        z_v = self.fc_v(z)
        z_s = self.fc_s(z)

        z_v = z_v.view(z.shape[0], self.out_channels_v * self.vit_latent_dim, 1, 1)
        z_s = z_s.view(z.shape[0], self.out_channels_s * self.vit_latent_dim, 1, 1)

        z_v = self.batch_norm_v(z_v)
        z_v = self.leaky_relu(z_v)
        z_s = self.batch_norm_s(z_s)
        z_s = self.leaky_relu(z_s)

        v = self.decoder_v(z_v)
        s = self.decoder_s(z_s)

        return s, v

    def freeze_net(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("net froze")


if __name__ == '__main__':
    model = VSNet(1)
    inp = torch.randn((2,1,70,70))

    seis, vel = model(inp)

    print(seis.shape, vel.shape)