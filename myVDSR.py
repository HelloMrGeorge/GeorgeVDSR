import math
from torch import nn

class VDSR(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.mid_layer = []
        mid_single_layer = [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True)]
        for i in range(d):
            self.mid_layer.extend(mid_single_layer)
        self.mid_layer = nn.Sequential(*self.mid_layer)
        
        self.last_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for x in self.modules():
            if isinstance(x, nn.Conv2d):
                n = x.kernel_size[0] * x.kernel_size[1] * x.out_channels
                nn.init.normal_(x.weight.data, mean=0.0, std=math.sqrt(2.0/n))

    def forward(self, x):
        out = self.first_layer(x)
        out = self.mid_layer(out)
        out = self.last_layer(out)
        return out


if __name__ == '__main__':
    model = VDSR(12)
    print(model.modules)