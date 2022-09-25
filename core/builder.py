"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-03-31 17:48:41
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 00:51:50
"""

from typing import Dict, Tuple
import numpy as np
import os
import torch
import torch.nn as nn
from pyutils.datasets import get_dataset
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from pyutils.config import configs
from pyutils.optimizer.sam import SAM
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.loss import AdaptiveLossSoft, KLLossMixed
from .utils import (
    ComplexL1Loss,
    ComplexMSELoss,
    ComplexTVLoss,
    CurlLoss,
    DivergenceLoss,
    MatReader,
    PoyntingLoss,
    UnitGaussianNormalizer,
)

from core.models import *
from core.datasets import (
    MNISTDataset,
    FashionMNISTDataset,
    CIFAR10Dataset,
    SVHNDataset,
    MMIDataset,
)

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_dataloader(name: str = None, splits=["train", "valid", "test"]) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    if name == "fashionmnist":
        train_dataset, validation_dataset, test_dataset = (
            FashionMNISTDataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar10":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR10Dataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "svhn":
        train_dataset, validation_dataset, test_dataset = (
            SVHNDataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "mmi":
        train_dataset, validation_dataset, test_dataset = (
            MMIDataset(
                root=configs.dataset.root,
                split=split,
                test_ratio=configs.dataset.test_ratio,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                pol_list=configs.dataset.pol_list,
                processed_dir=configs.dataset.processed_dir,
            )
            if split in splits
            else None
            for split in ["train", "valid", "test"]
        )
    elif name == "ns_2d":
        ntrain, ntest = 10, 10
        r = 5
        h = int(((256 - 1) / r) + 1)
        s = h
        reader = MatReader(os.path.join(configs.dataset.root, "navier_stokes/ns_data.mat"))
        x_train = reader.read_field("a")[:ntrain, ::r, ::r][:, :s, :s]
        y_train = reader.read_field("u")[:ntrain, ::r, ::r][:, :s, :s]

        x_test = reader.read_field("a")[-ntest:, ::r, ::r][:, :s, :s]
        y_test = reader.read_field("u")[-ntest:, ::r, ::r][:, :s, :s]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)
        y_train = y_train[..., -1][:, np.newaxis, ...]
        y_test = y_test[..., -1][:, np.newaxis, ...]
        x_train = x_train.reshape(ntrain, 1, s, s)
        x_test = x_test.reshape(ntest, 1, s, s)

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        validation_dataset = None
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            configs.dataset.img_height,
            configs.dataset.img_width,
            dataset_dir=configs.dataset.root,
            transform=configs.dataset.transform,
        )
        validation_dataset = None

    train_loader = (
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=configs.run.batch_size,
            shuffle=int(configs.dataset.shuffle),
            pin_memory=True,
            num_workers=configs.dataset.num_workers,
        )
        if train_dataset is not None
        else None
    )

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=configs.dataset.num_workers,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = (
        torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=configs.dataset.num_workers,
        )
        if test_dataset is not None
        else None
    )

    return train_loader, validation_loader, test_loader


def make_model(device: Device, random_state: int = None, **kwargs) -> nn.Module:
    if "mlp" in configs.model.name.lower():
        model = eval(configs.model.name)(
            n_feat=configs.dataset.img_height * configs.dataset.img_width,
            n_class=configs.dataset.n_class,
            hidden_list=configs.model.hidden_list,
            block_list=[int(i) for i in configs.model.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "cnn" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            pool_out_size=configs.model.pool_out_size,
            stride_list=configs.model.stride_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            block_list=[int(i) for i in configs.model.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            v_max=configs.quantize.v_max,
            # v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=configs.model.photodetect,
            bias=False,
            device=device,
            super_layer_name=configs.super_layer.name,
            super_layer_config=configs.super_layer.arch,
            bn_affine=configs.model.bn_affine,
        ).to(device)
        model.reset_parameters(random_state)
        # model.super_layer.set_sample_arch(configs.super_layer.sample_arch)
    elif "vgg" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            pool_out_size=configs.model.pool_out_size,
            stride_list=configs.model.stride_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            block_list=[int(i) for i in configs.model.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            v_max=configs.quantize.v_max,
            # v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=configs.model.photodetect,
            bias=False,
            device=device,
            super_layer_name=configs.super_layer.name,
            super_layer_config=configs.super_layer.arch,
            bn_affine=configs.model.bn_affine,
        ).to(device)
        model.reset_parameters(random_state)
    elif "resnet" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channel=configs.dataset.in_channel,
            n_class=configs.dataset.n_class,
            block_list=[int(i) for i in configs.model.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "unet" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.dataset.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            act_func=configs.model.act_func,
            domain_size=configs.model.domain_size,
            grid_step=configs.model.grid_step,
            buffer_width=configs.model.buffer_width,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding=configs.model.pos_encoding,
            device=device,
            **kwargs,
        ).to(device)
    elif "neurolight" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.dataset.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            domain_size=configs.model.domain_size,
            grid_step=configs.model.grid_step,
            buffer_width=configs.model.buffer_width,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding=configs.model.pos_encoding,
            with_cp=configs.model.with_cp,
            device=device,
            conv_stem=configs.model.conv_stem,
            aug_path=configs.model.aug_path,
            ffn=configs.model.ffn,
            ffn_dwconv=configs.model.ffn_dwconv,
            **kwargs,
        ).to(device)
    elif "fno" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.dataset.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            domain_size=configs.model.domain_size,
            grid_step=configs.model.grid_step,
            buffer_width=configs.model.buffer_width,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding=configs.model.pos_encoding,
            with_cp=configs.model.with_cp,
            device=device,
            **kwargs,
        ).to(device)
        # model.reset_parameters(random_state)
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")

    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(configs.run.n_epochs), eta_min=float(configs.scheduler.lr_min)
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.scheduler.lr_gamma)
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "cmse":
        criterion = ComplexMSELoss(norm=cfg.norm)
    elif name == "cmae":
        criterion = ComplexL1Loss(norm=cfg.norm)
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif name == "adaptive":
        criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    elif name == "mixed_kl":
        criterion = KLLossMixed(
            T=getattr(cfg, "T", 3),
            alpha=getattr(cfg, "alpha", 0.9),
        )
    elif name == "curl_loss":
        criterion = CurlLoss(configs.model.grid_step, configs.model.wavelength)
    elif name == "tv_loss":
        criterion = ComplexTVLoss(norm=cfg.norm)
    elif name == "div_loss":
        criterion = DivergenceLoss()
    elif name == "poynting_loss":
        criterion = PoyntingLoss(configs.model.grid_step, configs.model.wavelength)
    else:
        raise NotImplementedError(name)
    return criterion
