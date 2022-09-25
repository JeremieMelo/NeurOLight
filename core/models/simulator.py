import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import torch
import torch.nn.functional as F
from angler import Simulation
from angler.derivatives import unpack_derivs

from .constant import *


class Simulation1D:
    """FDFD simulation of a 1-dimensional system"""

    def __init__(
        self,
        mode="Ez",
        device_length=DEVICE_LENGTH,
        npml=0,
        buffer_length=BUFFER_LENGTH,
        buffer_permittivity=BUFFER_PERMITTIVITY,
        dl=dL,
        l0=L0,
        use_dirichlet_bcs=False,
    ):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.buffer_length = buffer_length
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = l0
        self.use_dirichlet_bcs = use_dirichlet_bcs

    def solve(self, epsilons: np.array, omega=OMEGA_1550, src_x=None, clip_buffers=False):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml
        start = self.npml + self.buffer_length
        end = start + self.device_length

        permittivities = np.ones(total_length, dtype=np.float64)

        # set permittivity and reflection zone
        permittivities[:start] = self.buffer_permittivity
        permittivities[start:end] = epsilons
        permittivities[end:] = self.buffer_permittivity

        if src_x is None:
            src_x = int(self.device_length / 2)

        sim = Simulation(
            omega,
            permittivities,
            self.dl,
            [self.npml, 0],
            self.mode,
            L0=self.L0,
            use_dirichlet_bcs=self.use_dirichlet_bcs,
        )
        sim.src[src_x + self.npml + self.buffer_length] = 1j

        if clip_buffers:
            clip0 = self.npml + self.buffer_length
            clip1 = -(self.npml + self.buffer_length)
        else:
            clip0 = None
            clip1 = None

        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            permittivities = permittivities[clip0:clip1]
            Hx = Hx[clip0:clip1]
            Hy = Hy[clip0:clip1]
            Ez = Ez[clip0:clip1]
            return permittivities, src_x, Hx, Hy, Ez

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            permittivities = permittivities[clip0:clip1]
            Ex = Ex[clip0:clip1]
            Ey = Ey[clip0:clip1]
            Hz = Hz[clip0:clip1]
            return permittivities, src_x, Ex, Ey, Hz

        else:
            raise ValueError("Polarization must be Ez or Hz!")

    def get_operators(self, omega=OMEGA_1550):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml

        perms = np.ones(total_length, dtype=np.float64)

        start = self.npml + self.buffer_length
        end = start + self.device_length

        perms[:start] = self.buffer_permittivity
        perms[end:] = self.buffer_permittivity

        sim = Simulation(
            omega,
            perms,
            self.dl,
            [self.npml, 0],
            self.mode,
            L0=self.L0,
            use_dirichlet_bcs=self.use_dirichlet_bcs,
        )

        Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)

        N = np.asarray(perms.shape)
        M = np.prod(N)

        vector_eps_z = EPSILON0 * self.L0 * perms.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format="csr")

        curl_curl = Dxf @ Dxb + Dyf @ Dyb

        other = omega**2 * MU0 * self.L0 * T_eps_z

        return curl_curl.todense(), other.todense()


class Simulation2D:
    """FDFD simulation of a 2-dimensional  system"""

    def __init__(
        self,
        mode="Ez",
        device_length=DEVICE_LENGTH_2D,
        npml=0,
        buffer_length=BUFFER_LENGTH,
        buffer_permittivity=BUFFER_PERMITTIVITY,
        dl=dL,
        l0=L0,
        use_dirichlet_bcs=False,
    ):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.buffer_length = buffer_length
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = l0
        self.use_dirichlet_bcs = use_dirichlet_bcs

    def solve(self, epsilons: np.array, omega=OMEGA_1550, src_x=None, src_y=None, clip_buffers=False):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml
        start = self.npml + self.buffer_length
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        permittivities = np.ones((total_length, total_length), dtype=np.float64)

        # set permittivity and reflection zone
        permittivities[:, :start] = self.buffer_permittivity
        permittivities[:start, :] = self.buffer_permittivity

        permittivities[start:end, start:end] = epsilons

        permittivities[:, end:] = self.buffer_permittivity
        permittivities[end:, :] = self.buffer_permittivity

        if src_x is None:
            src_x = self.device_length // 2
        if src_y is None:
            src_y = self.device_length // 2

        sim = Simulation(
            omega,
            permittivities,
            self.dl,
            [self.npml, self.npml],
            self.mode,
            L0=self.L0,
            # use_dirichlet_bcs=self.use_dirichlet_bcs,
        )
        sim.src[src_y + self.npml + self.buffer_length, src_x + self.npml + self.buffer_length] = 1j

        if clip_buffers:
            clip0 = self.npml + self.buffer_length
            clip1 = -(self.npml + self.buffer_length)
        else:
            clip0 = None
            clip1 = None

        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            permittivities = permittivities[clip0:clip1, clip0:clip1]
            Hx = Hx[clip0:clip1, clip0:clip1]
            Hy = Hy[clip0:clip1, clip0:clip1]
            Ez = Ez[clip0:clip1, clip0:clip1]
            return permittivities, src_x, src_y, Hx, Hy, Ez

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            permittivities = permittivities[clip0:clip1, clip0:clip1]
            Ex = Ex[clip0:clip1, clip0:clip1]
            Ey = Ey[clip0:clip1, clip0:clip1]
            Hz = Hz[clip0:clip1, clip0:clip1]
            return permittivities, src_x, src_y, Ex, Ey, Hz

        else:
            raise ValueError("Polarization must be Ez or Hz!")

    def solve2(self, epsilons: np.array, omega=OMEGA_1550, src_x=None, src_y=None, clip_buffers=False):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml
        start = self.npml + self.buffer_length
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        permittivities = np.ones((total_length, total_length), dtype=np.float64)

        # set permittivity and reflection zone
        permittivities[:, :start] = self.buffer_permittivity
        permittivities[:start, :] = self.buffer_permittivity

        permittivities[start:end, start:end] = epsilons

        permittivities[:, end:] = self.buffer_permittivity
        permittivities[end:, :] = self.buffer_permittivity

        sim = Simulation(
            omega,
            permittivities,
            self.dl,
            [self.npml, self.npml],
            self.mode,
            L0=self.L0,
            # use_dirichlet_bcs=self.use_dirichlet_bcs,
        )
        sim.src[
            src_y[0] + self.npml + self.buffer_length : src_y[1] + self.npml + self.buffer_length,
            src_x[0] + self.npml + self.buffer_length : src_x[1] + self.npml + self.buffer_length,
        ] = 1j

        if clip_buffers:
            clip0 = self.npml + self.buffer_length
            clip1 = -(self.npml + self.buffer_length)
        else:
            clip0 = None
            clip1 = None

        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            permittivities = permittivities[clip0:clip1, clip0:clip1]
            Hx = Hx[clip0:clip1, clip0:clip1]
            Hy = Hy[clip0:clip1, clip0:clip1]
            Ez = Ez[clip0:clip1, clip0:clip1]
            return permittivities, src_x, src_y, Hx, Hy, Ez

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            permittivities = permittivities[clip0:clip1, clip0:clip1]
            Ex = Ex[clip0:clip1, clip0:clip1]
            Ey = Ey[clip0:clip1, clip0:clip1]
            Hz = Hz[clip0:clip1, clip0:clip1]
            return permittivities, src_x, src_y, Ex, Ey, Hz

        else:
            raise ValueError("Polarization must be Ez or Hz!")

    def get_operators(self, omega=OMEGA_1550):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml

        perms = np.ones((total_length, total_length), dtype=np.float64)

        start = self.npml + self.buffer_length
        end = start + self.device_length

        # set permittivity and reflection zone
        perms[:, :start] = self.buffer_permittivity
        perms[:start, :] = self.buffer_permittivity
        perms[:, end:] = self.buffer_permittivity
        perms[end:, :] = self.buffer_permittivity

        sim = Simulation(
            omega,
            perms,
            self.dl,
            [self.npml, self.npml],
            self.mode,
            L0=self.L0,
            # use_dirichlet_bcs=self.use_dirichlet_bcs,
        )

        Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)

        N = np.asarray(perms.shape)
        M = np.prod(N)

        vector_eps_z = EPSILON0 * self.L0 * perms.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format="csr")

        curl_curl = Dxf @ Dxb + Dyf @ Dyb

        other = omega**2 * MU0 * self.L0 * T_eps_z

        return curl_curl.todense(), other.todense()
