"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-24 23:27:31
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-01 16:22:32
"""
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-04 13:38:43
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-04-04 15:16:55
"""

import os
import numpy as np
import torch

from torch import Tensor
from torchpack.datasets.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.transforms import InterpolationMode

resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["MMI", "MMIDataset"]


class MMI(VisionDataset):
    url = None
    filename_suffix = "fields_epsilon_mode.pt"
    train_filename = "training"
    test_filename = "test"
    folder = "mmi"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.7,
        pol_list: List[str] = ["Hz"],
        processed_dir: str = "processed",
        download: bool = False,
    ) -> None:
        self.processed_dir = processed_dir
        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio
        self.pol_list = sorted(pol_list)
        self.filenames = [f"{pol}_{self.filename_suffix}" for pol in self.pol_list]
        self.train_filename = self.train_filename
        self.test_filename = self.test_filename

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.wavelength: Any = []
        self.grid_step: Any = []
        self.data: Any = []
        self.targets = []
        self.eps_min = self.eps_max = None

        self.process_raw_data()
        self.wavelength, self.grid_step, self.data, self.targets = self.load(train=train)

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, self.processed_dir)
        processed_training_file = os.path.join(processed_dir, f"{self.train_filename}.pt")
        processed_test_file = os.path.join(processed_dir, f"{self.test_filename}.pt")
        if os.path.exists(processed_training_file) and os.path.exists(processed_test_file):
            with open(os.path.join(self.root, self.processed_dir, f"{self.test_filename}.pt"), "rb") as f:
                wavelength, grid_step, data, targets, _, _ = torch.load(f)
                if data.shape[0:2] == targets.shape[0:2]:
                    print("Data already processed")
                    return
        wavelength, grid_step, data, targets, eps_min, eps_max = self._load_dataset()
        (
            wavelength_train,
            grid_step_train,
            data_train,
            targets_train,
            wavelength_test,
            grid_step_test,
            data_test,
            targets_test,
        ) = self._split_dataset(wavelength, grid_step, data, targets)
        data_train, data_test = self._preprocess_dataset(data_train, data_test)
        self._save_dataset(
            wavelength_train,
            grid_step_train,
            data_train,
            targets_train,
            wavelength_test,
            grid_step_test,
            data_test,
            targets_test,
            eps_min,
            eps_max,
            processed_dir,
            self.train_filename,
            self.test_filename,
        )

    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_data = [torch.load(os.path.join(self.root, "raw", filename)) for filename in self.filenames]
        data = torch.cat([rd["eps"] for rd in raw_data], dim=0)  # use Ex Ey Ez fields for now
        wavelength = torch.cat([rd["wavelength"] for rd in raw_data], dim=0)
        grid_step = torch.cat([rd["grid_step"] for rd in raw_data], dim=0)
        targets = torch.cat([rd["fields"] for rd in raw_data], dim=0)  # use Ex Ey Ez field for now
        self.eps_min = raw_data[0]["eps_min"]
        self.eps_max = raw_data[0]["eps_max"]

        return wavelength, grid_step, data, targets, self.eps_min, self.eps_max

    def _split_dataset(
        self, wavelength: Tensor, grid_step: Tensor, data: Tensor, targets: Tensor
    ) -> Tuple[Tensor, ...]:
        from sklearn.model_selection import train_test_split

        (
            wavelength_train,
            wavelength_test,
            grid_step_train,
            grid_step_test,
            data_train,
            data_test,
            targets_train,
            targets_test,
        ) = train_test_split(
            wavelength, grid_step, data, targets, train_size=self.train_ratio, random_state=42
        )
        print(f"training: {data_train.shape[0]} examples, " f"test: {data_test.shape[0]} examples")
        return (
            wavelength_train,
            grid_step_train,
            data_train,
            targets_train,
            wavelength_test,
            grid_step_test,
            data_test,
            targets_test,
        )

    def _preprocess_dataset(self, data_train: Tensor, data_test: Tensor) -> Tuple[Tensor, Tensor]:
        return data_train, data_test

    @staticmethod
    def _save_dataset(
        wavelength_train: Tensor,
        grid_step_train: Tensor,
        data_train: Tensor,
        targets_train: Tensor,
        wavelength_test: Tensor,
        grid_step_test: Tensor,
        data_test: Tensor,
        targets_test: Tensor,
        eps_min: Tensor,
        eps_max: Tensor,
        processed_dir: str,
        train_filename: str = "training",
        test_filename: str = "test",
    ) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, f"{train_filename}.pt")
        processed_test_file = os.path.join(processed_dir, f"{test_filename}.pt")
        with open(processed_training_file, "wb") as f:
            torch.save((wavelength_train, grid_step_train, data_train, targets_train, eps_min, eps_max), f)

        with open(processed_test_file, "wb") as f:
            torch.save((wavelength_test, grid_step_test, data_test, targets_test, eps_min, eps_max), f)
        print(f"Processed dataset saved")

    def load(self, train: bool = True):
        filename = f"{self.train_filename}.pt" if train else f"{self.test_filename}.pt"
        with open(os.path.join(self.root, self.processed_dir, filename), "rb") as f:
            wavelength, grid_step, data, targets, self.eps_min, self.eps_max = torch.load(f)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            if isinstance(wavelength, np.ndarray):
                wavelength = torch.from_numpy(wavelength)
            if isinstance(grid_step, np.ndarray):
                grid_step = torch.from_numpy(grid_step)
        return wavelength, grid_step, data, targets

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

    def _check_integrity(self) -> bool:
        return all([os.path.exists(os.path.join(self.root, "raw", filename)) for filename in self.filenames])

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        # wave [bs, 2, 1] real
        # data [bs, 2, 6, h, w] real # 224x224 -> 80x520, 96x384
        # bs = #Wavelength * #epsilon combinatio
        # N = #ports/wavelenth/epsilon combination
        # data [bs, 2, 6, H, W] -> [bs, 2, 4, H, W]
        # target [bs, 2, 4, h, w] real grid_step ->
        return self.wavelength[item], self.grid_step[item], self.data[item], self.targets[item]

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class MMIDataset:
    def __init__(
        self,
        root: str,
        split: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        pol_list: List[str],
        processed_dir: str = "processed",
    ):
        self.root = root
        self.split = split
        self.test_ratio = test_ratio
        assert 0 < test_ratio < 1, print(f"Only support test_ratio from (0, 1), but got {test_ratio}")
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.eps_min = None
        self.eps_max = None
        self.pol_list = sorted(pol_list)
        self.processed_dir = processed_dir

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = MMI(
                self.root,
                train=True,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                pol_list=self.pol_list,
                processed_dir=self.processed_dir,
            )
            self.eps_min = train_valid.eps_min
            self.eps_max = train_valid.eps_max

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            if self.train_valid_split_ratio[0] + self.train_valid_split_ratio[1] > 0.99999:
                valid_len = len(train_valid) - train_len
            else:
                valid_len = int(self.train_valid_split_ratio[1] * len(train_valid))
                train_valid.data = train_valid.data[:train_len+valid_len]
                train_valid.wavelength = train_valid.wavelength[:train_len+valid_len]
                train_valid.grid_step = train_valid.grid_step[:train_len+valid_len]
                train_valid.targets = train_valid.targets[:train_len+valid_len]

            split = [train_len, valid_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = MMI(
                self.root,
                train=False,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                pol_list=self.pol_list,
                processed_dir=self.processed_dir,
            )

            self.data = test
            self.eps_min = test.eps_min
            self.eps_max = test.eps_max

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        return self.__getitem__(index)


def test_mmi():
    import pdb

    # pdb.set_trace()
    mmi = MMI(root="../../data", download=True, processed_dir="random_size5")
    print(mmi.data.size(), mmi.targets.size())
    mmi = MMI(root="../../data", train=False, download=True, processed_dir="random_size5")
    print(mmi.data.size(), mmi.targets.size())
    mmi = MMIDataset(
        root="../../data",
        split="train",
        test_ratio=0.1,
        train_valid_split_ratio=[0.9, 0.1],
        pol_list=[f"rHz_{i}" for i in range(5)],
    )
    print(len(mmi))


if __name__ == "__main__":
    test_mmi()
