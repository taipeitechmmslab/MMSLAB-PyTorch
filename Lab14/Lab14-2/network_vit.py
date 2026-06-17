import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        channels=1280,
        image_size=64,
        num_heads=8,
        depth=2,
        dim_feedforward=None,
        dropout=0.1,
    ):
        super(TransformerBottleneck, self).__init__()
        # MobileNetV2 最深層特徵圖會比輸入縮小 32 倍，因此輸入大小需可整除 32
        if image_size % 32 != 0:
            raise ValueError('image_size must be divisible by 32')

        # 記錄 bottleneck 特徵圖大小，作為 learnable positional embedding 的基準解析度
        self.base_grid = image_size // 32
        self.channels = channels
        # 位置編碼會和每個空間 token 相加，讓 Transformer 保留位置資訊
        self.pos_embed = nn.Parameter(torch.zeros(1, self.base_grid * self.base_grid, channels))

        # bottleneck token 數少，主要用 attention 補強最深層的全域語意關係
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or channels * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def _pos_embed_for_grid(self, grid_size):
        # 若輸入解析度與初始化時相同，直接使用原本的位置編碼
        if grid_size == (self.base_grid, self.base_grid):
            return self.pos_embed

        # 若輸入解析度不同，將位置編碼還原成 2D 後插值到目前特徵圖大小
        pos_embed = self.pos_embed.transpose(1, 2).reshape(
            1,
            self.channels,
            self.base_grid,
            self.base_grid,
        )
        pos_embed = F.interpolate(pos_embed, size=grid_size, mode='bicubic', align_corners=False)
        # 轉回 token 格式，讓每個空間位置對應一個 positional embedding
        return pos_embed.flatten(2).transpose(1, 2)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # [B, C, H, W] -> [B, H*W, C]，讓 Transformer 處理空間 token
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self._pos_embed_for_grid((height, width))
        tokens = self.encoder(tokens)

        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class SegMobileUNetTransformer(nn.Module):
    def __init__(
        self,
        num_class,
        image_size=64,
        transformer_depth=2,
        transformer_heads=8,
        transformer_dropout=0.1,
    ):
        super(SegMobileUNetTransformer, self).__init__()
        # 使用 ImageNet 預訓練 MobileNetV2 作為 encoder，decoder 對齊 SegMobileUNet
        self.mobile_net = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        ).features
        self.hooks = []

        # 取出不同解析度的 encoder 特徵，後續提供給 U-Net skip connection 使用
        target_layer_idx = [0, 3, 6, 13]
        self.intermediates = [None for _ in range(len(target_layer_idx))]
        for i, layer_idx in enumerate(target_layer_idx):
            # 透過 forward hook 儲存指定 layer 的輸出，不需要修改 MobileNetV2 結構
            hook = self.mobile_net[layer_idx].register_forward_hook(self.hook_intermediate(i))
            self.hooks.append(hook)

        # 在 encoder 最深層加入 Transformer，補強 bottleneck 的全域關係建模
        self.transformer = TransformerBottleneck(
            channels=1280,
            image_size=image_size,
            num_heads=transformer_heads,
            depth=transformer_depth,
            dropout=transformer_dropout,
        )

        # Decoder 逐層上採樣，並在 forward 中和對應的 encoder 特徵串接
        # 通道數需配合 skip connection 後的 concat 結果
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
            ),
        )

    def hook_intermediate(self, hook_idx):
        def hook_fn(module, input, output):
            # 將 hook 抓到的中間特徵存入固定位置，forward 時再依序取用
            self.intermediates[hook_idx] = output
        return hook_fn

    def __del__(self):
        # 移除 forward hook，避免模型釋放後仍保留 hook 參考
        if hasattr(self, 'hooks') and len(self.hooks) > 0:
            for hook in self.hooks:
                hook.remove()

    def forward(self, x):
        # 每次 forward 前清空舊的中間特徵，避免重複使用前一次輸入的結果
        for i in range(len(self.intermediates)):
            self.intermediates[i] = None

        # MobileNetV2 產生最深層特徵後，先經過 Transformer bottleneck
        x = self.mobile_net(x)
        x = self.transformer(x)

        # hook 特徵由淺到深排列，decoder 上採樣時反向接回 skip connection
        for i in range(len(self.intermediates)):
            x = self.decoder[i](x)
            x = torch.cat([x, self.intermediates[len(self.intermediates) - i - 1]], dim=1)

        # 最後一層輸出每個像素對應的類別 logits
        return self.decoder[-1](x)


if __name__ == '__main__':
    from torchinfo import summary

    # 簡單檢查模型輸入輸出尺寸與參數量
    net = SegMobileUNetTransformer(3)
    summary(net, (1, 3, 64, 64))
