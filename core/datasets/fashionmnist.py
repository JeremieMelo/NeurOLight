"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-09-26 00:48:23
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 00:50:02
"""
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-09-26 00:29:22
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 00:31:58
"""

import torch

from torchvision import datasets, transforms
from typing import List
from pyutils.general import logger
from torchvision.transforms import InterpolationMode


__all__ = ["FashionMNISTDataset"]


resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


class FashionMNISTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        train_valid_split_ratio: List[float] = (0.9, 0.1),
        center_crop: bool = True,
        resize: bool = True,
        resize_mode: str = "bicubic",
        binarize: bool = False,
        binarize_threshold: float = 0.1307,
        digits_of_interest: List[float] = list(range(10)),
        n_test_samples: int = 10000,
        n_valid_samples: int = 5000,
    ):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_modes[resize_mode]
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.digits_of_interest = digits_of_interest
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        if not self.center_crop == 28:
            tran.append(transforms.CenterCrop(self.center_crop))
        if not self.resize == 28:
            tran.append(transforms.Resize(self.resize, interpolation=self.resize_mode))
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = datasets.FashionMNIST(self.root, train=True, download=True, transform=transform)

            idx, _ = torch.stack([train_valid.targets == number for number in self.digits_of_interest]).max(
                dim=0
            )
            train_valid.targets = train_valid.targets[idx]
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                if self.n_valid_samples is None:
                    # use all samples in valid set
                    self.data = valid_subset
                else:
                    # use a subset of valid set, useful to speedup evo search
                    valid_subset.indices = valid_subset.indices[: self.n_valid_samples]
                    self.data = valid_subset
                    logger.warning(f"Only use the front " f"{self.n_valid_samples} images as " f"VALID set.")

        else:
            test = datasets.FashionMNIST(self.root, train=False, transform=transform)
            idx, _ = torch.stack([test.targets == number for number in self.digits_of_interest]).max(dim=0)
            test.targets = test.targets[idx]
            test.data = test.data[idx]
            if self.n_test_samples is None:
                # use all samples as test set
                self.data = test
            else:
                # use a subset as test set
                test.targets = test.targets[: self.n_test_samples]
                test.data = test.data[: self.n_test_samples]
                self.data = test
                logger.warning(f"Only use the front {self.n_test_samples} " f"images as TEST set.")

    def __getitem__(self, index: int):
        img = self.data[index][0]
        if self.binarize:
            img = 1.0 * (img > self.binarize_threshold) + -1.0 * (img <= self.binarize_threshold)
        digit = self.digits_of_interest.index(self.data[index][1])
        return img, torch.tensor(digit).long()
        # instance = {'image': img, 'digit': digit}
        # return instance

    def __len__(self) -> int:
        return self.n_instance


def test():
    mnist = FashionMNISTDataset(
        root="../../data",
        split="train",
        train_valid_split_ratio=[0.9, 0.1],
        center_crop=28,
        resize=28,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=(3, 6),
        n_test_samples=100,
        n_valid_samples=1000,
        fashion=True,
    )
    data, label = mnist.__getitem__(20)
    print(data.size(), label.size())
    print("finish")


if __name__ == "__main__":
    test()
