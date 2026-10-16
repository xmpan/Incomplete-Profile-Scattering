import cv2
import time
import matplotlib.pyplot as plt
import os
import sys
from torchvision.datasets import MNIST, EMNIST, FashionMNIST
sys.path.append(os.path.dirname(sys.path[0]))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from forward_data.Forward_computation import *
precision = np.float32
complex_precision = np.complex64
torch_precision = torch.float32
torch_complex_precision = torch.complex64

def load_mnist(path, kind):
    """Download MNIST to `path` if needed and return NumPy arrays."""
    dataset = MNIST(root=path, train=(kind == 'train'), download=True)
    images = dataset.data.numpy().reshape(-1, 784)
    labels = dataset.targets.numpy().astype(np.uint8)
    return images, labels

def load_emnist(path, kind):
    """Download EMNIST Letters to `path` if needed and return NumPy arrays."""
    dataset = EMNIST(root=path, split='letters', train=(kind == 'train'), download=True)
    images = dataset.data.numpy().reshape(-1, 784)
    labels = dataset.targets.numpy().astype(np.uint8)
    return images, labels

def load_Famnist(path, kind):
    """Download Fashion-MNIST to `path` if needed and return NumPy arrays."""
    dataset = FashionMNIST(root=path, train=(kind == 'train'), download=True)
    images = dataset.data.numpy().reshape(-1, 784)
    labels = dataset.targets.numpy().astype(np.uint8)
    return images, labels

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    train_images, train_labels = load_mnist(data_dir, 'train')
    test_images, test_labels = load_mnist(data_dir, 't10k')
    # train_images, train_labels = load_emnist(data_dir, 'train')
    # test_images, test_labels = load_emnist(data_dir, 'test')
    # train_images, train_labels = load_Famnist(data_dir, 'train')
    # test_images, test_labels = load_Famnist(data_dir, 't10k')

    time_start = time.time()

    k0 = 2 * np.pi * 1e9 / 3 / 1e8 # 1GHz
    ri = Transmitter(n_pts=1, angle_strat=45., angle_end=45., radius=0.6)  #320 0
    print('Incident angle:', ri.angle / np.pi * 180)
    domain_length = 0.64
    doi = DoI(center=(0., 0.), length=(domain_length, domain_length), resolution=(domain_length/64., domain_length/64.))
    Ei = Planewave(k0, ri, doi)
    Gd = Gd(k0, doi.Nx, doi.Ny, doi.dx, doi.dy)

    # generate train_images_data----------------------------------------------------------------------------------------
    num_train = 16000
    Et_train = np.zeros([num_train, doi.size], dtype=complex_precision)
    xi_train = np.zeros([num_train, doi.size], dtype=complex_precision)
    xi_mask_train = np.zeros([num_train, doi.size], dtype=complex_precision)
    Et_mask_train = np.zeros([num_train, doi.size], dtype=complex_precision)
    for i in range(num_train):
        ###### objec1 ##################################################################################################
        xi = train_images[i, :].reshape(1, 784)
        xi_one = xi/255
        xi_one[(xi_one > 0) & (xi_one < 0.1)] = 0.1
        xi_onee = cv2.resize(xi_one.reshape(28, 28), doi.shape,
                             interpolation=cv2.INTER_NEAREST).reshape(1, doi.size)
        xi_ttt1 = xi_onee + 1.j * 0
        ######complete object###########################################################################################
        xi_train[i:(i+1), :] = xi_ttt1
        jvie_solver = JVIE_Solver()
        J = jvie_solver.solve(Gd, xi_train[i, :], Ei.at(0))
        Et_train[i: (i + 1), :] = Ei.data[0:(0 + 1), :] + (Gd @ J).reshape(1, doi.size)
        ######incomplete object#########################################################################################
        xi_tttttttttttt = xi_ttt1.reshape(doi.shape)
        for k in range(int(doi.shape[0]/ 2)):
            for kk in range(int(doi.shape[1])):
                xi_tttttttttttt[k, kk] = 0 + 1.j * 0
        xi_mask_train[i:(i + 1), :] = xi_tttttttttttt.reshape(1, doi.size)

        jvie_solver = JVIE_Solver()
        J = jvie_solver.solve(Gd, xi_mask_train[i, :], Ei.at(0))
        Et_mask_train[i: (i + 1), :] = Ei.data[0:(0 + 1), :] + (Gd @ J).reshape(1, doi.size)
    #
    np.save(f"Xi_training_data_1_1_num{num_train}_64times64.npy", xi_train)
    np.save(f"Ximask_training_data_1_1_num{num_train}_64times64.npy", xi_mask_train)
    np.save(f"Et_training_data_1_1_num{num_train}_64times64.npy", Et_train)
    np.save(f"Etmask_training_data_1_1_num{num_train}_64times64.npy", Et_mask_train)
    print("Training data has been saved successfully!")

    # generate test images data-----------------------------------------------------------------------------------------
    num_test = 1000
    Et_test = np.zeros([num_test, doi.size], dtype=complex_precision)
    xi_test = np.zeros([num_test, doi.size], dtype=complex_precision)
    xi_mask_test = np.zeros([num_test, doi.size], dtype=complex_precision)
    Et_mask_test = np.zeros([num_test, doi.size], dtype=complex_precision)

    for i in range(num_test):
        xi = test_images[i, :].reshape(1, 784)
        xi_one = xi/255
        xi_one[(xi_one > 0) & (xi_one < 0.1)] = 0.1
        xi_onee = cv2.resize(xi_one.reshape(28, 28), doi.shape,
                             interpolation=cv2.INTER_NEAREST).reshape(1, doi.size)
        xi_ttt1 = xi_onee + 1.j * 0
        ######complete object###########################################################################################
        xi_test[i:(i + 1), :] = xi_ttt1
        jvie_solver = JVIE_Solver()
        J = jvie_solver.solve(Gd, xi_test[i, :], Ei.at(0))
        Et_test[i: (i + 1), :] = Ei.data[0:(0 + 1), :] + (Gd @ J).reshape(1, doi.size)

        ######incomplete object#########################################################################################
        xi_tttttttttttt = xi_ttt1.reshape(doi.shape)
        for k in range(int(doi.shape[0]/2)):
            for kk in range(int(doi.shape[1])):
                xi_tttttttttttt[k, kk] = 0 + 1.j * 0
        xi_mask_test[i:(i + 1), :] = xi_tttttttttttt.reshape(1, doi.size)

        jvie_solver = JVIE_Solver()
        J = jvie_solver.solve(Gd, xi_mask_test[i, :], Ei.at(0))
        Et_mask_test[i: (i + 1), :] = Ei.data[0:(0 + 1), :] + (Gd @ J).reshape(1, doi.size)

    np.save(f"Xi_test_data_1_1_num{num_test}_64times64.npy", xi_test)
    np.save(f"Ximask_test_data_1_1_num{num_test}_64times64.npy", xi_mask_test)
    np.save(f"Et_test_data_1_1_num{num_test}_64times64.npy", Et_test)
    np.save(f"Etmask_test_data_1_1_num{num_test}_64times64.npy", Et_mask_test)
    print("Test data has been saved successfully!")

    time_end = time.time()
    time_c = time_end - time_start
    print('time cost:', time_c)
