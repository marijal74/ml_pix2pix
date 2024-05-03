# %%
import torch.nn as nn
import torch

# %%
class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            *Discriminator.__get_layer__(input=2*num_channels, out=64, batch_norm=False),
            *Discriminator.__get_layer__(input=64, out=128),
            *Discriminator.__get_layer__(input=128, out=256),
            *Discriminator.__get_layer__(input=256, out=512),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
    @staticmethod
    def __get_layer__(input, out, batch_norm = True):
        layer = [
            nn.Conv2d(in_channels=input, out_channels=out, kernel_size=4 , stride=2, padding=1, bias=False)
        ]
        if batch_norm:
            layer.append(nn.BatchNorm2d(out))
        layer.append(nn.LeakyReLU(0.2, True))
        return layer
    
    def forward(self, x, y):
        img = torch.cat((x, y), 1)
        return self.model(img)
