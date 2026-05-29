import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset


class ImageFilesDataset(Dataset):
    # 支援資料夾中直接放圖片的情境，回傳固定 label 以符合 ImageFolder 介面
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, name))
            and name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def plot_images(images):
    """將一個 batch 的圖片橫向拼接顯示。"""
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    """將圖片 batch 排成 grid 後存檔。"""
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    # 圖片正規化到 -1 到 1，對齊 DDPM 訓練與採樣後的輸出範圍
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 有類別子資料夾時使用 ImageFolder，否則直接讀取資料夾內的圖片檔
    has_subdirs = any(
        os.path.isdir(os.path.join(args.dataset_path, name))
        for name in os.listdir(args.dataset_path)
    )
    if has_subdirs:
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    else:
        dataset = ImageFilesDataset(args.dataset_path, transform=transforms)
    if args.num_sample != -1:
        dataset = torch.utils.data.Subset(dataset, range(args.num_sample))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    """建立模型權重與生成結果的輸出目錄。"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
