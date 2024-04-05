# %%
import torch.nn as nn
import torch

# %%
class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        final = [
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
            ]
        self.model = nn.Sequential(
            *Discriminator.__get_layer__(input=2*num_channels, out=64, batch_norm=False),
            *Discriminator.__get_layer__(input=64, out=128),
            *Discriminator.__get_layer__(input=128, out=256),
            *Discriminator.__get_layer__(input=256, out=512),
            *final
        )
    @staticmethod
    def __get_layer__(input, out, batch_norm = True):
        layer = [
            nn.Conv2d(in_channels=input, out_channels=out, kernel_size=4 , stride=2, padding=1)
        ]
        if batch_norm:
            layer.append(nn.BatchNorm2d(out))
        layer.append(nn.LeakyReLU(0.2, True))
        return layer
    
    def forward(self, x, y):
        img = torch.cat((x, y), 1)
        return self.model(img)
