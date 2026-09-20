import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from utils.data_augmentation import (
    cv_random_flip,
    randomCrop,
    randomRotation,
    randomPeper,
    colorEnhance,
)


# =========================
# Train Dataset
# =========================
class TrainDataset(Dataset):
    """
    RGB-D COD 训练集
    返回：
        - 若同时有 depth 和 edge：
            img    : (3, H, W), float, 已归一化
            depth  : (1, H, W), float, [0,1]
            gt     : (1, H, W), float, [0,1]
            edge   : (1, H, W), float, [0,1]
        - 若无 edge：
            img, depth, gt
    """

    def __init__(
        self,
        image_root: str,
        gt_root: str,
        depth_root: str,
        trainsize: int,
        edge_root: Optional[str] = None,
        rVFlip: bool = True,
        rCrop: bool = True,
        rRotate: bool = True,
        do_color_enhance: bool = True,
        rPeper: bool = False,
    ):
        self.edge_root = edge_root
        self.depth_root = depth_root
        self.trainsize = trainsize

        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.do_color_enhance = do_color_enhance
        self.rPeper = rPeper

        # 收集文件路径
        self.imgs = [
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.gts = [
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.endswith(".png")
        ]
        self.depths = [
            os.path.join(depth_root, f)
            for f in os.listdir(depth_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        if edge_root is not None:
            self.edges = [
                os.path.join(edge_root, f)
                for f in os.listdir(edge_root)
                if f.endswith(".jpg") or f.endswith(".png")
            ]

        # 排序保证一一对应
        self.imgs = sorted(self.imgs)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        if edge_root is not None:
            self.edges = sorted(self.edges)

        # 保证数量一致
        assert len(self.imgs) == len(self.gts) == len(self.depths), \
            "imgs / gts / depths 数量不一致，请检查数据集文件名是否对应"
        if edge_root is not None:
            assert len(self.edges) == len(self.imgs), \
                "edge 文件数量与 imgs 不一致"

        # 过滤掉尺寸不匹配的样本
        self.filter_files()

        # transforms
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),  # [0,1]
            ]
        )
        self.depth_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),  # [0,1] 单通道
            ]
        )

        self.size = len(self.imgs)
        print(f">>> training with {self.size} samples")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index):
        img = self.rgb_loader(self.imgs[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])  # ★ 这里改成 rgb_loader
        if self.edge_root is not None:
            edge = self.binary_loader(self.edges[index])

        # 数据增强（保持和之前一样，只是列表里有 depth）
        if self.edge_root is not None:
            if self.rVFlip:
                img, gt, depth, edge = cv_random_flip([img, gt, depth, edge])
            if self.rCrop:
                img, gt, depth, edge = randomCrop([img, gt, depth, edge])
            if self.rRotate:
                img, gt, depth, edge = randomRotation([img, gt, depth, edge])
        else:
            if self.rVFlip:
                img, gt, depth = cv_random_flip([img, gt, depth])
            if self.rCrop:
                img, gt, depth = randomCrop([img, gt, depth])
            if self.rRotate:
                img, gt, depth = randomRotation([img, gt, depth])

        if self.do_color_enhance:
            img = colorEnhance(img)
        if self.rPeper:
            gt = randomPeper(gt)

        img = self.img_transform(img)  # (3, H, W)
        gt = self.mask_transform(gt)  # (1, H, W)
        depth = self.depth_transform(depth)  # ★ 这里会得到 (3, H, W)，因为是彩色图

        if self.edge_root is not None:
            edge = self.mask_transform(edge)  # (1, H, W)
            return img, depth, gt, edge
        else:
            return img, depth, gt

    def filter_files(self):
        """过滤掉尺寸不一致的样本，保证 img/gt/depth(/edge) size 一样"""
        images = []
        gts = []
        depths = []
        edges = [] if self.edge_root is not None else None

        for i, (img_path, gt_path, depth_path) in enumerate(
            zip(self.imgs, self.gts, self.depths)
        ):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)

            ok = (img.size == gt.size) and (img.size == depth.size)

            if self.edge_root is not None:
                edge_path = self.edges[i]
                edge = Image.open(edge_path)
                ok = ok and (edge.size == img.size)

            if ok:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
                if self.edge_root is not None:
                    edges.append(edge_path)

        self.imgs = images
        self.gts = gts
        self.depths = depths
        if self.edge_root is not None:
            self.edges = edges

    @staticmethod
    def rgb_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    @staticmethod
    def binary_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")


# =========================
# Test Dataset
# =========================
class TestDataset(Dataset):
    """
    RGB-D COD 测试集
    返回：
        若有 edge：
            image        : (3, H, W)  归一化
            depth        : (1, H, W)
            gt           : (1, H, W)
            gt_origin    : (1, H0, W0) 原始尺寸
            edge         : (1, H, W)
            edge_origin  : (1, H0, W0)
            depth_origin : (1, H0, W0)
            name         : 文件名（统一为 .png）
        若无 edge：
            image, depth, gt, gt_origin, depth_origin, name
    """

    def __init__(
        self,
        image_root: str,
        gt_root: str,
        depth_root: str,
        testsize: int,
        edge_root: Optional[str] = None,
    ):
        self.testsize = testsize
        self.edge_root = edge_root
        self.depth_root = depth_root

        self.images = [
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.gts = [
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.endswith(".tif") or f.endswith(".png")
        ]
        self.depths = [
            os.path.join(depth_root, f)
            for f in os.listdir(depth_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        if edge_root is not None:
            self.edges = [
                os.path.join(edge_root, f)
                for f in os.listdir(edge_root)
                if f.endswith(".jpg") or f.endswith(".png")
            ]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        if edge_root is not None:
            self.edges = sorted(self.edges)

        assert len(self.images) == len(self.gts) == len(self.depths), \
            "test: imgs / gts / depths 数量不一致"
        if edge_root is not None:
            assert len(self.edges) == len(self.images), \
                "test: edges 数量与 imgs 不一致"

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
            ]
        )
        self.depth_transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])  # ★ 读彩色深度图
        if self.edge_root is not None:
            edge = self.binary_loader(self.edges[index])

        # 原始尺寸的 GT / edge / depth
        gt_origin = transforms.PILToTensor()(gt)        # (1, H0, W0)
        depth_origin = transforms.PILToTensor()(depth)  # (1, H0, W0)
        if self.edge_root is not None:
            edge_origin = transforms.PILToTensor()(edge)

        # resize + 归一化
        image = self.img_transform(image)
        gt = self.mask_transform(gt)
        depth = self.depth_transform(depth)  # ★ 得到 (3, H, W)
        if self.edge_root is not None:
            edge = self.mask_transform(edge)

        # 统一名字为 .png
        name = os.path.basename(self.images[index])
        if name.endswith(".jpg"):
            name = name.replace(".jpg", ".png")

        if self.edge_root is not None:
            return (
                image,
                depth,
                # gt,
                gt_origin,
                edge,
                # edge_origin,
                # depth_origin,
                name,
            )
        else:
            return image, depth, gt, gt_origin, depth_origin, name

    @staticmethod
    def rgb_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    @staticmethod
    def binary_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")


# =========================
# 示例：根据 Config 构建 DataLoader
# =========================
if __name__ == "__main__":
    from config import Config  # 假设你的 Config 定义在 config.py 中

    cfg = Config()

    # 训练集
    train_dataset = TrainDataset(
        image_root=cfg.dp.train_imgs,
        gt_root=cfg.dp.train_masks,
        depth_root=cfg.dp.train_depth,
        trainsize=cfg.trainsize,
        edge_root=cfg.dp.train_edges,   # 如果不想用 edge，可传 None
    )
    train_datald = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # 简单测试一下 batch 结构
    batch = next(iter(train_datald))
    if len(batch) == 4:
        img, depth, gt, edge = batch
        print("train batch shapes:", img.shape, depth.shape, gt.shape, edge.shape)
    else:
        img, depth, gt = batch
        print("train batch shapes:", img.shape, depth.shape, gt.shape)