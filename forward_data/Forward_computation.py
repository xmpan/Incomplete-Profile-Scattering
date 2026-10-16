import numpy as np
import torch

from _config import torch_precision
from _config import torch_complex_precision
from _config import precision
from _config import complex_precision
from scipy import special
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres




class Transmitter:
    def __init__(self, n_pts, angle_strat=360., angle_end=0., radius=1.0):
        self.n_pts = n_pts
        self.radius = radius
        self.ans = angle_strat
        self.ane = angle_end
        self._uniform_sampler()

    def _uniform_sampler(self):
        self.angle = np.deg2rad(
            np.flipud(np.linspace(self.ans, self.ane, self.n_pts, endpoint=False)))
        self.x = self.radius * np.cos(self.angle)
        self.y = self.radius * np.sin(self.angle)


class Receiver:
    def __init__(self, n_pts, radius, sa=0., st=360.):
        self.n_pts = n_pts
        self.radius = radius
        self.sa = sa
        self.st = st
        self._uniform_sampler()

    def _uniform_sampler(self):
        self.angle = np.deg2rad(np.linspace(
            self.sa, self.st, self.n_pts, endpoint=False))
        self.x = self.radius * np.cos(self.angle)
        self.y = self.radius * np.sin(self.angle)


class DoI:
    def __init__(self, center, length, resolution):
        self.space_dim = 2
        self.center = center
        self.length = length
        self.resolution = resolution
        self._uniform_sampler()

    def _uniform_sampler(self):
        self.Lx, self.Ly = self.length[0], self.length[1]
        self.dx, self.dy = self.resolution[0], self.resolution[1]
        self.Nx = int(np.ceil(self.Lx/self.dx))
        self.Ny = int(np.ceil(self.Ly/self.dy))
        self.shape = (self.Nx, self.Ny)
        self.size = np.prod(self.shape)
        x = np.arange(1, self.Nx + 1, dtype=precision) * self.dx - self.dx * \
            0.5 - 0.5 * self.Lx + self.center[0]
        y = np.arange(1, self.Ny + 1, dtype=precision) * self.dy - self.dy * \
            0.5 - 0.5 * self.Ly + self.center[1]
        self.X, self.Y = np.meshgrid(x, y)
        # self.Y, self.X = np.meshgrid(y, x)

class Ez_inc:
    def __init__(self, k0, ri, doi):
        self.k0 = k0
        self.ri = ri
        self.doi = doi

        self.Ni = ri.n_pts
        self.Nv = np.prod(doi.shape)
        self.shape = (self.Ni, self.Nv)
        self.data = np.empty(self.shape, dtype=complex_precision)

    def data(self):
        return self.data

    def at(self, i):
        return self.data[i, :]


class Planewave(Ez_inc):
    def __init__(self, k0, ri, doi):
        super(Planewave, self).__init__(k0, ri, doi)
        self._calculate()

    def _calculate(self):
        self.k_inc_x = np.cos(self.ri.angle)
        self.k_inc_y = np.sin(self.ri.angle)

        for i in range(self.Ni):
            # self.data[i, :] = np.exp(
            #     1j * self.k0 * (self.k_inc_x[i] * self.doi.X.ravel() + self.k_inc_y[i] * self.doi.Y.ravel()))
            self.data[i, :] = np.exp(
                1j * self.k0 * (self.k_inc_x[i]*(self.doi.X.ravel()-self.ri.x)+self.k_inc_y[i]*(self.doi.Y.ravel()--self.ri.y)))

class Dipole(Ez_inc):
    def __init__(self, k0, ri, doi):
        super(Dipole, self).__init__(k0, ri, doi)
        self._calculate()

    def _calculate(self):
        self.radius = self.ri.radius
        self.x = self.radius * np.cos(self.ri.angle)
        self.y = self.radius * np.sin(self.ri.angle)

        for i in range(self.Ni):
            self.data[i, :] = (0.25j*self.k0**2) *\
                special.hankel1(
                    0, self.k0 * np.sqrt((self.x[i]-self.doi.X.ravel())**2+(self.y[i]-self.doi.Y.ravel())**2))

def TorchGd_dot(Z_mtx, other):
    N, C = other.shape[0], other.shape[-1]
    H, W = Z_mtx.shape[-3], Z_mtx.shape[-2]
    Nx, Ny = (H+1)//2, (W+1)//2
    y = torch.fft.ifft2(Z_mtx * torch.fft.fft2(other.reshape([N, Nx, Ny, C]), s=[
                        H, W], dim=(-3, -2)), dim=(-3, -2))[:, (Nx-1):, (Ny-1):, :]
    # y = torch.reshape(y, other.shape)
    return y

class TorchGd:
    def __init__(self, Gd, ri):
        self._Nx, self._Ny = Gd.Nx, Gd.Ny
        self._shape = (2*Gd.Nx-1, 2*Gd.Ny-1)

        if (not Gd.is_already_fft()):
            Gd._fft()
        self.Z_mtx_fft = (torch.tensor(
            Gd.Z_mtx_fft, dtype=torch_complex_precision,
            requires_grad=False, device='cuda').reshape(
                [1, self._shape[0], self._shape[1], 1])).repeat(ri.n_pts*10, 1, 1, 1)

        if (not Gd.is_already_conj_fft()):
            Gd._conj_fft()
        self.Z_mtx_conj_fft = (torch.tensor(
            Gd.Z_mtx_conj_fft, dtype=torch_complex_precision,
            requires_grad=False, device='cuda').reshape(
                [1, self._shape[0], self._shape[1], 1])).repeat(ri.n_pts*10, 1, 1, 1)

    def Z_mtx(self):
        return self.Z_mtx_fft

    def Z_mtx_conj(self):
        return self.Z_mtx_conj_fft

    def __matmul__(self, other):
        return TorchGd_dot(self.Z_mtx_fft, other)

    def conj_dot(self, other):
        return TorchGd_dot(self.Z_mtx_conj_fft, other)

class Gd:
    def __init__(self, k0, Nx, Ny, dx, dy):
        self.k0 = k0
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        self._is_already_fft = False
        self._is_already_conj_fft = False
        self.shape = (self.Nx*self.Ny, self.Nx*self.Ny)
        self._fillin()

    def _fillin(self):
        cellrad = np.sqrt(self.dx*self.dy/np.pi)
        x_diff = np.arange(1, self.Nx) * self.dx
        y_diff = np.arange(1, self.Ny) * self.dy
        x_diff = np.concatenate((-np.flipud(x_diff), np.array([0.0]), x_diff))
        y_diff = np.concatenate((-np.flipud(y_diff), np.array([0.0]), y_diff))

        X_diff, Y_diff = np.meshgrid(x_diff, y_diff)
        # Y_diff,X_diff = np.meshgrid(y_diff, x_diff)
        R = np.sqrt(X_diff**2 + Y_diff**2)
        R[self.Nx - 1, self.Ny - 1] = 1.0  # avoid singular

        self.Z_mtx = (1j*self.k0*np.pi*cellrad*0.5*special.jv(1,
                      self.k0*cellrad)) * special.hankel1(0, self.k0*R)
        self.Z_mtx[self.Nx - 1, self.Ny - 1] = (
            1j*self.k0*np.pi*cellrad*0.5) * special.hankel1(1, self.k0*cellrad) - 1.0

    def _fft(self):
        self.Z_mtx_fft = np.fft.fft2(self.Z_mtx)
        self._is_already_fft = True

    def is_already_fft(self):
        return self._is_already_fft

    def _conj_fft(self):
        self.Z_mtx_conj_fft = np.fft.fft2(self.Z_mtx.conj())
        self._is_already_conj_fft = True

    def is_already_conj_fft(self):
        return self._is_already_conj_fft

    def __matmul__(self, other):
        Nx, Ny = self.Nx, self.Ny

        if self.is_already_fft() == False:
            self._fft()

        y = np.fft.ifft2(self.Z_mtx_fft * np.fft.fft2(
            other.reshape([Nx, Ny]), [2*Nx-1, 2*Ny-1]
        ))[(Nx-1):, (Ny-1):]

        return y.ravel()

    def conj_dot(self, other):
        Nx, Ny = self.Nx, self.Ny

        if self.is_already_conj_fft() == False:
            self._conj_fft()

        y = np.fft.ifft2(self.Z_mtx_conj_fft * np.fft.fft2(
            other.reshape([Nx, Ny]), [2*Nx-1, 2*Ny-1]
        ))[(Nx-1):, (Ny-1):]

        return y.ravel()


class Gs:

    def __init__(self, k0, rs, doi):
        self.k0 = k0
        self.rs = rs
        self.doi = doi

        self.Ns = rs.n_pts
        self.Nx, self.Ny = doi.Nx, doi.Ny
        self.dim = (self.Ns, self.Nx * self.Ny,)
        self._is_conj_available = False

        self._fillin()
        self._conj()

    def data(self):
        return self.Z_mtx

    def _fillin(self):
        cellrad = np.sqrt(self.doi.dx*self.doi.dy/np.pi)
        self.Z_mtx = np.empty(self.dim, dtype=complex_precision)

        for i in range(self.Ns):
            self.Z_mtx[i, :] = np.sqrt(
                (self.rs.x[i]-self.doi.X)**2+(self.rs.y[i]-self.doi.Y)**2).ravel()
            self.Z_mtx[i, :] = (1j*self.k0*np.pi*cellrad*0.5*special.jv(1,
                                                                             self.k0*cellrad))*special.hankel1(0, self.k0*self.Z_mtx[i, :])

    def _conj(self):
        self.Z_mtx_conj = self.Z_mtx.conj().T
        self._is_conj_available = True

    def is_conj_available(self):
        return self._is_conj_available

    def __matmul__(self, other):
        return np.dot(self.Z_mtx, other.ravel())

    def conj_dot(self, other):
        if self.is_conj_available() == False:
            self._conj()

        return np.dot(self.Z_mtx_conj, other.ravel())

class Gss:
    def __init__(self, k0, doi_S, doi_D):
        self.k0 = k0
        self.doi_D = doi_D
        self.doi_S = doi_S
        self.Nx_D, self.Ny_D = doi_D.Nx, doi_D.Ny
        self.Nx_S, self.Ny_S = doi_S.Nx, doi_S.Ny
        self.dim = (self.Nx_S*self.Ny_S, self.Nx_D * self.Ny_D,)
        self._fillin()
    def data(self):
        return self.Z_mtx
    def _fillin(self):
        cellrad = np.sqrt(self.doi_D.dx * self.doi_D.dy / np.pi)
        self.Z_mtx = np.empty(self.dim, dtype=complex_precision)
        X_trunk = (self.doi_S.X).reshape(self.doi_S.size, 1)
        Y_trunk = (self.doi_S.Y).reshape(self.doi_S.size, 1)
        for i in range(self.doi_S.size):
            self.Z_mtx[i, :] = np.sqrt(
                (X_trunk[i, 0] - self.doi_D.X) ** 2 + (Y_trunk[i, 0] - self.doi_D.Y) ** 2).ravel()
            self.Z_mtx[i, :] = (1j * self.k0 * np.pi * cellrad * 0.5 * special.jv(1,
                                                                                       self.k0 * cellrad)) * special.hankel1(
                0, self.k0 * self.Z_mtx[i, :])
    def __matmul__(self, other):
        return np.dot(self.Z_mtx, other.ravel())

class ForwardSolver:
    def __init__(self, toler, max_iter, n_restart):
        self._toler = toler
        self._max_iter = max_iter
        self._n_restart = n_restart


class JVIE_Solver(ForwardSolver):
    def __init__(self, toler=1e-3, max_iter=20, n_restart=50):
        super(JVIE_Solver, self).__init__(toler, max_iter, n_restart)

    def solve(self, Gd, coeff, b):
        rhs = coeff * b  # element-wise product

        def matvec(x):
            return x - coeff * (Gd@x)

        A = LinearOperator(dtype=complex_precision,
                           shape=Gd.shape, matvec=matvec)
        x, exitcode = gmres(A, rhs, restart=self._n_restart,
                            tol=self._toler, maxiter=self._max_iter, atol='legacy')
        return x

def evaluate_J_po(Es, U, S, Vh):
    return np.conj(((np.conj(Es)@U)*np.conj(1.0/S))@Vh)  #  返回Ni*Nv大小的J+

def AWGN(x, snr):
    """
    Additive White Gaussian Noise
    :param x: Raw Signal (vector)
    :param snr: SNR (decibel)
    :return: Received Signal With AWGN
    """
    size_1 = x.shape[0]
    size_2 = x.shape[1]
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / size_1 / size_2
    npower = xpower / snr
    noise = (np.random.randn(size_1, size_2) + 1j * np.random.randn(size_1, size_2)) * np.sqrt(0.5 * npower)  # Complex Number
    noise_level = np.linalg.norm(noise, ord='fro')/np.linalg.norm(x, ord='fro')
    print('噪声水平为：', noise_level*100, '%')
    return x + noise