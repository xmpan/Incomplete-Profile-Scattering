import time
import torch
from torch import nn
# import seaborn as sns
import torchvision
# import cv2
from torch.nn.functional import relu
import math
import scipy.io as io
from torch.autograd import Variable
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
from matplotlib.pyplot import MultipleLocator
import sys
from forward_data.Forward_computation import *
from torch.utils.tensorboard import SummaryWriter
import os
sys.path.append(os.path.dirname(sys.path[0]))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from matplotlib.gridspec import GridSpec
precision = np.float32
complex_precision = np.complex64
torch_precision = torch.float32
torch_complex_precision = torch.complex64
sys.path.append("data")
activations = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU
}

def mre_pixel(xi_pred, xi_pinn, doi_pinn):
    delta = 0
    for i in range(doi_pinn.size):
        delta = delta + np.abs((xi_pred[i]-xi_pinn[i])/xi_pinn[i])**2
    delta = np.sqrt(delta/doi_pinn.size)
    return delta


def plot_result(name, Xi_true, Xi_pred, Es_true, Es_pred, Ess_true, Ess_pred, dim=64, length=0.64*2):
    cmap = 'rainbow'
    fontsize=45

    fig=plt.figure(figsize=(75, 15))
    gs = GridSpec(2, 8, figure=fig)

    ax = fig.add_subplot(gs[0, 0:1])
    plt.imshow(np.real(Xi_true).reshape(dim, dim), cmap=cmap, vmin=0, vmax=1.0)
    plt.title(r"True (Real)", fontdict={'size': fontsize})
    a=plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim/2, dim-1])
    ax.set_xticklabels([-length/2, 0, length/2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim/2, dim-1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax=fig.add_subplot(gs[1, 0:1])
    plt.imshow(np.imag(Xi_true).reshape(dim, dim), cmap=cmap, vmin=0, vmax=1.0)
    plt.title(r"True (Imag)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim/2, dim-1])
    ax.set_xticklabels([-length/2, 0, length/2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim/2, dim-1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax=fig.add_subplot(gs[0, 1:2])
    plt.imshow(np.real(Xi_pred).reshape(dim, dim), cmap=cmap, vmin=0, vmax=1.0)
    plt.title("Pred (Real)", fontdict={'size': fontsize})
    a=plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim/2, dim-1])
    ax.set_xticklabels([-length/2, 0, length/2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim/2, dim-1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax=fig.add_subplot(gs[1, 1:2])
    plt.imshow(np.imag(Xi_pred).reshape(dim, dim), cmap=cmap, vmin=0, vmax=1.0)
    plt.title("Pred (Imag)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim/2, dim-1])
    ax.set_xticklabels([-length/2, 0, length/2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim/2, dim-1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax = fig.add_subplot(gs[0, 2:3])
    plt.imshow(np.real(Xi_true-Xi_pred).reshape(dim, dim), cmap=cmap)
    plt.title("Diff (Real)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim / 2, dim - 1])
    ax.set_xticklabels([-length / 2, 0, length / 2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim / 2, dim - 1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax = fig.add_subplot(gs[1, 2:3])
    plt.imshow(np.imag(Xi_true-Xi_pred).reshape(dim, dim), cmap=cmap)
    plt.title("Diff (Imag)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim / 2, dim - 1])
    ax.set_xticklabels([-length / 2, 0, length / 2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim / 2, dim - 1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax = fig.add_subplot(gs[0, 3:4])
    plt.imshow(np.real(Es_true).reshape(dim, dim), cmap=cmap)
    plt.title("True (Real)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim / 2, dim - 1])
    ax.set_xticklabels([-length / 2, 0, length / 2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim / 2, dim - 1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax = fig.add_subplot(gs[1, 3:4])
    plt.imshow(np.imag(Es_true).reshape(dim, dim), cmap=cmap)
    plt.title("True (Imag)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim / 2, dim - 1])
    ax.set_xticklabels([-length / 2, 0, length / 2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim / 2, dim - 1])
    ax.set_yticklabels([-length / 2, 0, length / 2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax = fig.add_subplot(gs[0, 4:5])
    plt.imshow(np.real(Es_pred).reshape(dim, dim), cmap=cmap)
    plt.title("Pred (Real)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim / 2, dim - 1])
    ax.set_xticklabels([-length / 2, 0, length / 2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim / 2, dim - 1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax = fig.add_subplot(gs[1, 4:5])
    plt.imshow(np.imag(Es_pred).reshape(dim, dim), cmap=cmap,)
    plt.title("Pred (Imag)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim / 2, dim - 1])
    ax.set_xticklabels([-length / 2, 0, length / 2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim / 2, dim - 1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})


    ax = fig.add_subplot(gs[0, 5:6])
    plt.imshow(np.real(Es_pred.reshape(dim, dim)-Es_true.reshape(dim, dim)), cmap=cmap)
    plt.title("Diff (Real)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim / 2, dim - 1])
    ax.set_xticklabels([-length / 2, 0, length / 2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim / 2, dim - 1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax = fig.add_subplot(gs[1, 5:6])
    plt.imshow(np.imag(Es_pred.reshape(dim, dim)-Es_true.reshape(dim, dim)), cmap=cmap)
    plt.title("Diff (Imag)", fontdict={'size': fontsize})
    a = plt.colorbar(fraction=0.045, pad=0.05)
    a.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xticks([0, dim / 2, dim - 1])
    ax.set_xticklabels([-length / 2, 0, length / 2])
    ax.set_xlabel('y(m)', fontdict={'size': fontsize})
    ax.set_yticks([0, dim / 2, dim - 1])
    ax.set_yticklabels([-length/2, 0, length/2])
    ax.set_ylabel('x(m)', fontdict={'size': fontsize})

    ax = fig.add_subplot(gs[0, 6:8])
    y = MultipleLocator(0.1)
    x = np.linspace(0, 359, 360)
    plt.plot(np.real(Ess_true[0, 0:360]), color='orange', linewidth=4)
    plt.scatter(x, np.real(Ess_pred[0, 0:360]), marker='*', s=200)
    ax.yaxis.set_major_locator(y)
    ax.yaxis.set_ticks_position('right')
    my_x_ticks = np.arange(0, 359, 90)
    plt.xticks(my_x_ticks, size=fontsize)
    # plt.xticks(fontsize=40)
    plt.yticks(fontsize=fontsize)
    plt.legend(['True', 'Pred'], fontsize=fontsize, frameon=True)
    plt.title("Scattered field(Real)", fontdict={'size': fontsize})
    plt.xlabel('Degrees', fontdict={'size': fontsize})
    plt.axis('tight')

    ax = fig.add_subplot(gs[1, 6:8])
    y = MultipleLocator(0.1)
    x = np.linspace(0, 359, 360)
    plt.plot(np.imag(Ess_true[0, 0:360]), color='orange', linewidth=4)
    plt.scatter(x, np.imag(Ess_pred[0, 0:360]), marker='*', s=200)
    ax.yaxis.set_major_locator(y)
    ax.yaxis.set_ticks_position('right')
    my_x_ticks = np.arange(0, 359, 90)
    plt.xticks(my_x_ticks, size=fontsize)
    # plt.xticks(fontsize=40)
    plt.yticks(fontsize=fontsize)
    plt.legend(['True', 'Pred'], fontsize=fontsize, frameon=True)
    plt.title("Scattered field(Imag)", fontdict={'size': fontsize})
    plt.xlabel('Degrees', fontdict={'size': fontsize})
    plt.axis('tight')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.125,
    #                     bottom=0.1,
    #                     right=0.9,
    #                     top=0.9,
    #                     wspace=0.05,
    #                     hspace=0.35)

    # plt.savefig(name)
    plt.show()

def update_xi_num_den(ri, J, Ei, Gd, doi):
    num_real = np.zeros([doi.size])
    num_imag = np.zeros([doi.size])
    den = np.zeros([doi.size])
    for p in range(ri.n_pts):
        Et = Gd@(J[0, p, :, :].reshape(doi.size)) + Ei.at(p)
        num_real += (np.divide(np.real(np.conj(Et) * (J[0, p, :, :].reshape(doi.size))), np.abs(Et)))**2
        num_imag += (np.divide(np.imag(np.conj(Et) * (J[0, p, :, :].reshape(doi.size))), np.abs(Et))) ** 2
        den += np.abs(Et)**2
    return np.sqrt(num_real / den) + 1j*np.sqrt(num_imag / den)

def Hermitian(A):
    return A.T.conjugate()

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    epochs = 100
    num_test = 1000
    print('loading trained-net...')
    net = torch.load(f'unet_rv_num15000_64rx_pair2.pkl')
    device = 'cuda:0'
    net.eval()

    doi_length = 0.64
    resolution = 64
    doi = DoI(center=(0., 0.),
              length=(doi_length, doi_length),
              resolution=(doi_length / resolution, doi_length / resolution))
    k0 = 2 * np.pi * 1e9 / 3 / 1e8  # 1GHz
    ri = Transmitter(n_pts=1, angle_strat=45., angle_end=45., radius=0.6)  # 320 0
    Ei = Planewave(k0, ri, doi)
    rs = 64
    xi = np.load(f"forward_data/Xi_test_data_1_1_num1000_64times64.npy")
    Et = np.load(f"forward_data/Et_test_data_1_1_num1000_64times64.npy")
    xi_mask = np.load(f"forward_data/Ximask_test_data_1_1_num1000_64times64.npy")
    Et_mask = np.load(f"forward_data/Etmask_test_data_1_1_num1000_64times64.npy")

    xi_in_1 = np.zeros([num_test, 4, doi.shape[0], doi.shape[1]], dtype=precision)
    for j in range(num_test):
        xi_in_1[j:(j + 1), 0:1, :, :] = np.real(xi_mask[j, :].reshape(1, 1, doi.shape[0], doi.shape[1]))
        xi_in_1[j:(j + 1), 1:2, :, :] = np.imag(xi_mask[j, :].reshape(1, 1, doi.shape[0], doi.shape[1]))
        xi_in_1[j:(j + 1), 2:3, :, :] = np.real(Et_mask[j, :].reshape(1, 1, doi.shape[0], doi.shape[1]))
        xi_in_1[j:(j + 1), 3:4, :, :] = np.imag(Et_mask[j, :].reshape(1, 1, doi.shape[0], doi.shape[1]))

    xi_in_1 = torch.tensor(xi_in_1).to(device)

    rel_error_xi_mask = np.zeros([num_test, 1], dtype=precision)
    rel_error_Et = np.zeros([num_test, 1], dtype=precision)
    rel_error_Es = np.zeros([num_test, 1], dtype=precision)
    ssim_xi = np.zeros([num_test, 1], dtype=precision)

    rsss = Receiver(n_pts=360, radius=10.0, sa=0, st=360)
    Gs_inv = Gs(k0, rsss, doi)
    for i in range(0, num_test):
        Et_test = Et[i:(i + 1), :]
        xi_true = xi[i:(i + 1), :]
        with torch.no_grad():
            prediction1, prediction2 = net(xi_in_1[i:i+1, :, :, :])

        prediction_out1 = prediction1.cpu().detach().numpy()
        prediction_out1 = prediction_out1[:, 0:1, :, :] + 1j * prediction_out1[:, 1:2, :, :]
        prediction_out2 = prediction2.cpu().detach().numpy()
        prediction_out2 = prediction_out2[:, 0:1, :, :] + 1j * prediction_out2[:, 1:2, :, :]

        pred_Es = (Gs_inv@(prediction_out1.reshape(1, doi.size)*prediction_out2.reshape(1, doi.size))[0, :]).reshape(1, 360)
        true_Es = (Gs_inv @ (xi_true * Et_test)[0, :]).reshape(1, 360)

        if i<5:
            plot_result(f'pred_result_pair2_{i}.svg',
                        Xi_true=xi_true,
                        Xi_pred=prediction_out1.reshape(1, doi.size),
                        Es_true=Et_test,
                        Es_pred=prediction_out2.reshape(1, doi.size),
                        dim=doi.shape[0],
                        length=doi.length[0],
                        Ess_pred=pred_Es,
                        Ess_true=true_Es)
            rel_error_xi_mask[i, 0] = np.linalg.norm(
                prediction_out1.reshape(1, doi.size) - xi_true.reshape(1, doi.size), ord=2) \
                                      / np.linalg.norm(xi_true.reshape(1, doi.size) + 1, ord=2)
            rel_error_Et[i, 0] = np.linalg.norm(prediction_out2.reshape(1, doi.size) - Et_test.reshape(1, doi.size),
                                                ord=2) / np.linalg.norm(Et_test.reshape(1, doi.size), ord=2)
            rel_error_Es[i, 0] = np.linalg.norm(pred_Es - true_Es, ord=2) / np.linalg.norm(true_Es, ord=2)
            print(rel_error_xi_mask[i, 0], rel_error_Et[i, 0], rel_error_Es[i, 0])

        xi_p = np.concatenate((np.expand_dims(np.real(prediction_out1.reshape(doi.shape)), axis=-1),
                               np.expand_dims(np.imag(prediction_out1.reshape(doi.shape)), axis=-1)),
                              axis=-1)
        xi_t = np.concatenate((np.expand_dims(np.real(xi_true.reshape(doi.shape)), axis=-1),
                                        np.expand_dims(np.imag(xi_true.reshape(doi.shape)), axis=-1)),
                                       axis=-1)
        ssim_xi[i, 0] = ssim(xi_p, xi_t, multichannel=True)

        rel_error_xi_mask[i, 0] = np.linalg.norm(prediction_out1.reshape(1, doi.size)-xi_true.reshape(1, doi.size), ord=2) \
                                  / np.linalg.norm(xi_true.reshape(1, doi.size)+1, ord=2)
        rel_error_Et[i, 0] = np.linalg.norm(prediction_out2.reshape(1, doi.size) - Et_test.reshape(1, doi.size),
                                            ord=2) / np.linalg.norm(Et_test.reshape(1, doi.size), ord=2)
        rel_error_Es[i, 0] = np.linalg.norm(pred_Es - true_Es, ord=2) / np.linalg.norm(true_Es, ord=2)



    print(f"Mean SSIM for scatterer reconstruction:", np.mean(ssim_xi[:, 0]))
    print(f"Mean relative error for scatterer reconstruction:", np.mean(rel_error_xi_mask[:, 0]))
    print(f"Mean relative error for total field prediction:", np.mean(rel_error_Et[:, 0]))
    print(f"Mean relative error for scattered field prediction:", np.mean(rel_error_Es[:, 0]))
    print('%------------------------------------------------------------')
