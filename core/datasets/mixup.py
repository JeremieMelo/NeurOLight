"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-09 02:02:22
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-09 02:27:11
"""
import torch
import numpy as np
import torchvision.transforms.functional as tf

__all__ = ["Mixup", "MixupAll"]


class Mixup:
    """Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        correct_lam=True,
        random_vflip_ratio: float = 0.0,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
        self.random_vflip_ratio = random_vflip_ratio

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size),
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            phase = np.exp(1j * np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size) * 2 * np.pi)
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)

        return lam, phase, use_cutmix

    def _params_per_batch(self):
        lam = 0.0
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
            phase = np.exp(1j * np.random.beta(self.mixup_alpha, self.mixup_alpha) * 2 * np.pi).item()
        return lam, phase, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, phase_batch, use_cutmix = self._params_per_elem(batch_size * x.size(1))
        # x [bs, mode, inc, h, w] complex
        # x flip [bs, mode', inc, h, w]
        indices = np.arange(batch_size * x.size(1)).reshape([batch_size, x.size(1)])
        for i in range(batch_size):
            np.random.shuffle(indices[i, :])
        indices = indices.flatten()
        # print(x.shape, indices.shape)
        x_orig = (
            x.flatten(0, 1)[indices].contiguous().view_as(x)
        )  # random shuffle the input modes; an unmodified original for mixing source
        lam_batch = torch.tensor(lam_batch, device=x.device, dtype=x.dtype).view(
            batch_size, -1, *([1] * (x.dim() - 2))
        )  # [bs, mode, 1, 1, 1]
        phase_batch = torch.tensor(phase_batch, device=x.device, dtype=x.dtype).view(
            batch_size, -1, *([1] * (x.dim() - 2))
        )  # [bs, mode, 1, 1, 1]
        x.copy_(x * lam_batch + (1 - lam_batch) * phase_batch * x_orig)
        return lam_batch, phase_batch, indices  # [bs, mode, 1, 1, 1], [batch*mode]

    def _mix_batch(self, x):
        lam, phase, use_cutmix = self._params_per_batch()
        if lam == 1.0 + 0j:
            return 1.0 + 0j
        batch_size = len(x)
        # x [bs, mode, inc, h, w] complex
        # x flip [bs, mode', inc, h, w]
        indices = np.arange(batch_size * x.size(1)).reshape([batch_size, x.size(1)])
        for i in range(batch_size):
            np.random.shuffle(indices[i, :])
        indices = indices.flatten()
        x_orig = (
            x.flatten(0, 1)[indices].contiguous().view_as(x)
        )  # random shuffle the input modes; an unmodified original for mixing source
        x.mul_(lam).add_(x_orig.mul_((1.0 - lam) * phase))
        return lam, phase, indices

    def random_vflip(self, x, target):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if np.random.rand() < self.random_vflip_ratio:
                    x.data[i, j].copy_(tf.vflip(x[i, j]))
                    target.data[i, j].copy_(tf.vflip(target[i, j]))

    def __call__(self, x, target):
        # assert len(x) % 2 == 0, "Batch size should be even when using this"
        mode = x.data[:, :, 1:]  # inplace modified
        if self.mode == "elem":
            lam, phase, indices = self._mix_elem(mode)
        elif self.mode == "pair":
            raise NotImplementedError
        else:
            lam, phase, indices = self._mix_batch(mode)
        target = mixup_target(target, lam, phase, indices)
        self.random_vflip(x, target)
        return x, target


def mixup_target(target, lam, phase, indices):
    return target * lam + target.flatten(0, 1)[indices].contiguous().view_as(target).mul((1 - lam) * phase)


class MixupAll:
    """Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        correct_lam=True,
        random_vflip_ratio: float = 0.0,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
        self.random_vflip_ratio = random_vflip_ratio

    def _params_per_elem(self, batch_size, n_ports, device, random_state=None):
        if random_state is not None:
            old_seed = np.random.get_state()
            np.random.seed(random_state)
        lam = np.random.uniform(0, 1, [batch_size, n_ports, n_ports]).astype(np.float32)
        I = np.eye(n_ports, dtype=np.complex64)[np.newaxis, ...]  # [1,mode, mode]

        if self.mixup_enabled:
            lam /= np.sum(lam**2, axis=1, keepdims=True)  # normalize total energy
            phase = np.random.uniform(0, 2 * np.pi, size=[batch_size, n_ports, n_ports]).astype(
                np.float32
            )  # random phase shift
            phase[:, 0, :] = 0  # remove global phase, up to the phase of the first port
            lam = lam * np.exp(1j * phase)
            lam = np.where(np.random.rand(batch_size, 1, 1) < self.mix_prob, lam, I)
        else:
            lam = I
        if random_state is not None:
            np.random.set_state(old_seed)

        return torch.from_numpy(lam).to(device)

    def _params_per_batch(self):
        lam = 0.0
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
            phase = np.exp(1j * np.random.beta(self.mixup_alpha, self.mixup_alpha) * 2 * np.pi).item()
        return lam, phase, use_cutmix

    def _mix_elem(self, x, random_state=None):
        batch_size = len(x)
        lam_batch = self._params_per_elem(batch_size, x.shape[1], x.device, random_state)  # [bs, mode, mode] complex
        # x [bs, mode, inc, h, w] complex
        x.copy_(torch.einsum("bnihw,bnm->bmihw", x, lam_batch))
        return lam_batch

    def random_vflip(self, x, target, random_state):
        if random_state is not None:
            old_seed = np.random.get_state()
            np.random.seed(random_state)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if np.random.rand() < self.random_vflip_ratio:
                    x.data[i, j].copy_(tf.vflip(x[i, j]))
                    target.data[i, j].copy_(tf.vflip(target[i, j]))
        if random_state is not None:
            np.random.set_state(old_seed)

    def __call__(self, x, target, random_state=None, vflip=True, mode_dim=1):
        # assert len(x) % 2 == 0, "Batch size should be even when using this"
        mode = x.data[:, :, mode_dim : mode_dim + 1]  # inplace modified
        if self.mode == "elem":
            lam = self._mix_elem(mode, random_state)
        elif self.mode == "pair":
            raise NotImplementedError
        else:
            raise NotImplementedError
        target = mixup_target_all(target, lam)

        if vflip:
            self.random_vflip(x, target, random_state)
        return x, target


def mixup_target_all(target, lam):
    return torch.einsum("bnihw,bnm->bmihw", target, lam)
