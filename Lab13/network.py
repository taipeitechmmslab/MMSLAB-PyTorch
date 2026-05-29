import torch
import torch.nn as nn
import torchvision


class SegNet(nn.Module):
    def __init__(self, num_class):
        super(SegNet, self).__init__()
        # Encoder 將 64x64 影像逐步下採樣到 4x4 特徵圖
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # Decoder 透過轉置卷積恢復到原圖大小，最後輸出每個像素的類別 logits
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        feature_enc = self.encoder(x)
        dec_out = self.decoder(feature_enc)
        return dec_out


class SegUNet(SegNet):
    def __init__(self, num_class):
        super(SegUNet, self).__init__(num_class)
        # U-Net skip connection 會在通道維度串接 encoder 特徵，因此下一層輸入通道需加倍
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # 保留不同解析度的 encoder 特徵，讓 decoder 用 skip connection 補回空間細節
        x1 = self.encoder[:6](x)
        x2 = self.encoder[6:13](x1)
        x3 = self.encoder[13:20](x2)
        x = self.encoder[20:](x3)

        x = self.decoder[:3](x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder[3:6](x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder[6:9](x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder[9:](x)
        return x


class SegMobileUNet(nn.Module):
    def __init__(self, num_class):
        super(SegMobileUNet, self).__init__()
        # 使用 ImageNet 預訓練 MobileNetV2 作為 encoder
        self.mobile_net = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        ).features
        self.hooks = []

        # 取出 32x32、16x16、8x8、4x4 特徵，對應 decoder 的 skip connection
        target_layer_idx = [0, 3, 6, 13]
        self.intermediates = [None for _ in range(len(target_layer_idx))]
        for i in range(len(target_layer_idx)):
            layer_idx = target_layer_idx[i]
            hook = self.mobile_net[layer_idx].register_forward_hook(self.hook_intermediate(i))
            self.hooks.append(hook)

        # 每次上採樣後與對應特徵串接，下一段 decoder 需接收合併後的通道數
        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(1280, 96, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(192, 32, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 24, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(24),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1),
            )
        )

    def hook_intermediate(self, hook_idx):
        def hook_fn(module, input, output):
            self.intermediates[hook_idx] = output
        return hook_fn

    def __del__(self):
        if hasattr(self, 'hooks') and len(self.hooks) > 0:
            for hook in self.hooks:
                hook.remove()

    def forward(self, x):
        x = self.mobile_net(x)

        # hook 取得的特徵由淺到深排列，decoder 使用時需反向對齊
        for i in range(len(self.intermediates)):
            x = self.decoder[i](x)
            x = torch.cat([x, self.intermediates[len(self.intermediates) - i - 1]], dim=1)

        x = self.decoder[-1](x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    nets = [SegMobileUNet(3), SegNet(3), SegUNet(3)]
    for net in nets:
        summary(net, (1, 3, 64, 64))


