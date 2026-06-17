import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class PetSegmentationTransform:
    def __init__(self, image_size=(64, 64), train=False):
        self.image_size = image_size
        self.train = train

    def __call__(self, image, trimap):
        if self.train:
            # 影像與 trimap 共用同一組幾何增強參數，避免標註與圖片錯位
            if torch.rand(()) < 0.5:
                image = F.hflip(image)
                trimap = F.hflip(trimap)

            _, height, width = F.get_dimensions(image)
            angle, translations, scale, shear = transforms.RandomAffine.get_params(
                degrees=(-10, 10),
                translate=(0.1, 0.1),
                scale_ranges=None,
                shears=None,
                img_size=[height, width],
            )
            image = F.affine(image, angle, translations, scale, shear, interpolation=InterpolationMode.BILINEAR)
            trimap = F.affine(trimap, angle, translations, scale, shear, interpolation=InterpolationMode.NEAREST)

        # trimap 必須使用 nearest，否則類別標籤會被插值成無效數值
        image = F.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        trimap = F.resize(trimap, self.image_size, interpolation=InterpolationMode.NEAREST)
        return image, trimap


class OxfordPetSegmentationDataset(Dataset):
    def __init__(self, root, split, transform=None, transform_color=None, download=True):
        # torchvision 會處理下載與 split，這裡只包成原本 training loop 使用的輸出格式
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types=['category', 'binary-category', 'segmentation'],
            download=download,
        )
        self.transform = transform
        self.transform_color = transform_color
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, (_, _, trimap) = self.dataset[idx]
        image = image.convert('RGB')
        trimap = trimap.convert('L')

        if self.transform_color:
            image = self.transform_color(image)
        if self.transform:
            image, trimap = self.transform(image, trimap)

        image = self.post_transform(image)

        # Oxford-IIIT Pet trimap 原始值為 1/2/3，模型訓練使用 0/1/2
        trimap = np.asarray(trimap).copy()
        trimap[trimap == 0] = 2
        trimap = torch.from_numpy(trimap - 1).to(torch.long)

        return image, trimap

    def denorm(self, t):
        mean = torch.tensor(self.norm_mean).unsqueeze(1).unsqueeze(2).to(t.device)
        std = torch.tensor(self.norm_std).unsqueeze(1).unsqueeze(2).to(t.device)
        return t * std + mean


def map_trimap(arr, map_forward=True):
    """
    將 0/1/2 mask 與可視化灰階值互相轉換。
    """
    if map_forward:
        arr = (arr < 86) * 0 + (arr >= 86) * (arr < 171) * 1 + (arr >= 171) * 2
    else:
        arr = (arr == 0) * 86 + (arr == 1) * 172 + (arr == 2) * 255
    return arr


def imshow_segmentation(img, mask, title=None):
    mask = map_trimap(mask, map_forward=False)
    fig, ax = plt.subplots(1, 2, dpi=200)

    img = img.numpy().transpose((1, 2, 0))
    mask = mask.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean

    ax[0].imshow(img)
    ax[1].imshow(mask)
    if title is not None:
        fig.suptitle(title)
    plt.show()


if __name__ == '__main__':
    dataset_path = './data'
    train_transform = PetSegmentationTransform((64, 64), train=True)
    train_transform_color = transforms.ColorJitter(contrast=0.2)
    train_dataset = OxfordPetSegmentationDataset(dataset_path, 'trainval', train_transform, train_transform_color)
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)

    for batch_img, batch_trimap in train_loader:
        batch_trimap = batch_trimap.unsqueeze(1)
        debug_img = torchvision.utils.make_grid(batch_img[:25])
        debug_mask = torchvision.utils.make_grid(batch_trimap[:25])
        debug_mask = debug_mask.to(torch.float32) / debug_mask.max()
        imshow_segmentation(debug_img, debug_mask)
        break
