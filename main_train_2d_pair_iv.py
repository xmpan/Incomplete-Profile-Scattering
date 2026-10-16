import time
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from forward_data.Forward_computation import *
from _network_module import *
import os
import sys
from tqdm import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append("data")

def Hermitian(A):
    return A.T.conjugate()

if __name__ == "__main__":
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    doi = DoI(center=(0., 0.), length=(0.64, 0.64), resolution=(0.64 / 64., 0.64 / 64.))
    k0 = 2 * np.pi * 1e9 / 3 / 1e8  # 1GHz
    ri = Transmitter(n_pts=1, angle_strat=45., angle_end=45., radius=0.6)  # 320 0
    Ei = Planewave(k0, ri, doi)
    # 定义GPU---------------------------------------------------------------------------------------------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # 准备训练数据-----------------------------------------------------------------------------------------------------------
    branch_batch_size = 30
    num_train_picture = 15000
    xi = np.load(f"forward_data/Xi_training_data_1_1_num16000_64times64.npy")
    Et = np.load(f"forward_data/Et_training_data_1_1_num16000_64times64.npy")
    xi_mask = np.load(f"forward_data/Ximask_training_data_1_1_num16000_64times64.npy")
    rs = 64
    Es = np.load(f"forward_data/Es_training_data_1_1_num15000_{rs}rx.npy")

    num_validation_picture = 1000
    Es_validation = np.load(f"forward_data/Es_validation_data_1_1_num1000_{rs}rx.npy")

    Es_input_training = np.zeros([num_train_picture, rs], dtype=complex_precision)
    net_input_training = np.zeros([num_train_picture, 2, doi.shape[0], doi.shape[0]], dtype=complex_precision)
    delta_label_training = np.zeros([num_train_picture, 2, doi.shape[0], doi.shape[0]], dtype=complex_precision)
    for j in range(num_train_picture):
        Es_input_training[j:(j + 1), :] = Es[j:(j + 1), :].reshape(1, rs)
        net_input_training[j:(j + 1), 0:1, :, :] = xi_mask[j, :].reshape(1, 1, doi.shape[0], doi.shape[1])
        net_input_training[j:(j + 1), 1:2, :, :] = Ei.data[0, :].reshape(1, 1, doi.shape[0], doi.shape[1])
        delta_label_training[j:(j + 1), 0:1, :, :] = xi[j, :].reshape(1, 1, doi.shape[0], doi.shape[1])
        delta_label_training[j:(j + 1), 1:2, :, :] = Et[j, :].reshape(1, 1, doi.shape[0], doi.shape[1])
    training_data = cnn_Dataset(net_input_training, delta_label_training, Es_input_training)
    train_dataloader = DataLoader(training_data, batch_size=branch_batch_size, shuffle=True)
    print('Training data shape:', training_data.samples.shape)

    Es_input_validation = np.zeros([num_validation_picture, rs], dtype=complex_precision)
    net_input_validation = np.zeros([num_validation_picture, 2, doi.shape[0], doi.shape[0]], dtype=complex_precision)
    delta_label_validation = np.zeros([num_validation_picture, 2, doi.shape[0], doi.shape[0]], dtype=complex_precision)
    for j in range(num_validation_picture):
        Es_input_validation[j:(j + 1), :] = Es_validation[j:(j + 1), :].reshape(1, rs)
        net_input_validation[j:(j + 1), 0:1, :, :] = xi_mask[j+15000, :].reshape(1, 1, doi.shape[0], doi.shape[1])
        net_input_validation[j:(j + 1), 1:2, :, :] = Ei.data[0, :].reshape(1, 1, doi.shape[0], doi.shape[1])
        delta_label_validation[j:(j + 1), 0:1, :, :] = xi[j+15000, :].reshape(1, 1, doi.shape[0], doi.shape[1])
        delta_label_validation[j:(j + 1), 1:2, :, :] = Et[j+15000, :].reshape(1, 1, doi.shape[0], doi.shape[1])
    validation_data = cnn_Dataset(net_input_validation, delta_label_validation, Es_input_validation)
    validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)
    print('Validation data shape:', validation_data.samples.shape)

    # 定义网络---------------------------------------------------------------------------------------------------------------
    model = unet_pair4(6, 64, 2, rs, doi).to(device)
    # ---initialize net------
    model.apply(weight_init)
    # 正式训练神经网络-----------------------------------------------------------------------------------------------------------
    # ------set weight decay------
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = torch.optim.Adam([
        {'params': weight_p, 'weight_decay': 0},
        {'params': bias_p, 'weight_decay': 0}], lr=0.0002)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5)

    loss_func = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    epochs = 100
    early_stopping_patience = 10
    early_stopping_min_delta = 0.0
    best_validation_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    epochs_ran = 0
    model_path = f'unet_rv_num{num_train_picture}_{rs}rx_pair4.pkl'
    loss_train = np.zeros([1, epochs])
    loss_valid = np.zeros([1, epochs])

    with trange(epochs) as t:
        for i in t:
            model.train()
            visual_dict = {'loss_submitted': 0.0
                           }  # 初始化用于可视化的字典
            for batch_branch, (X1, y, X2) in enumerate(train_dataloader):
                X_branch_1 = torch.zeros([branch_batch_size, 4, doi.shape[0], doi.shape[1]],
                                         dtype=torch_precision).to(
                    device)
                X_branch_1[:, 0:1, :, :] = torch.real(X1[:, 0:1, :, :])
                X_branch_1[:, 1:2, :, :] = torch.imag(X1[:, 0:1, :, :])
                X_branch_1[:, 2:3, :, :] = torch.real(X1[:, 1:2, :, :])
                X_branch_1[:, 3:4, :, :] = torch.imag(X1[:, 1:2, :, :])

                X_branch_2 = torch.zeros([branch_batch_size, 2*rs], dtype=torch_precision).to(device)
                X_branch_2[:, 0:rs] = torch.real(X2)
                X_branch_2[:, rs:2*rs] = torch.imag(X2)

                prediction1, prediction2 = model(x11=X_branch_1, x12=X_branch_2)

                deltaEt_true1 = torch.zeros([branch_batch_size, 2, doi.shape[0], doi.shape[1]],
                                            dtype=torch_precision).to(
                    device)
                deltaEt_true1[:, 0:1, :, :] = torch.real(y[:, 0:1, :, :])
                deltaEt_true1[:, 1:2, :, :] = torch.imag(y[:, 0:1, :, :])

                deltaEt_true2 = torch.zeros([branch_batch_size, 2, doi.shape[0], doi.shape[1]],
                                            dtype=torch_precision).to(
                    device)
                deltaEt_true2[:, 0:1, :, :] = torch.real(y[:, 1:2, :, :])
                deltaEt_true2[:, 1:2, :, :] = torch.imag(y[:, 1:2, :, :])

                loss_training = 1 / 2 * loss_func(prediction1, deltaEt_true1) \
                                + 1 / 2 * loss_func(prediction2, deltaEt_true2)

                #  更新权重
                optimizer.zero_grad()
                loss_training.backward()
                optimizer.step()
                visual_dict['loss_submitted'] += loss_training.item()  # 累加损失值

            scheduler.step() #  更新学习率
            train_loss = visual_dict['loss_submitted'] / len(train_dataloader)  # 计算平均损失
            loss_train[0, i] = train_loss
            t.set_description("Epoch %i" % i)
            t.set_description("Training loss_submitted：")
            t.set_postfix(loss=loss_train[0, i])
            time.sleep(.01)


            model.eval()
            with torch.no_grad():
                visual_dict_validation = {'loss_submitted': 0.0
                                         }  # 初始化用于可视化的字典
                for batch_branch, (X1, y, X2) in enumerate(validation_dataloader):
                    X_branch_1 = torch.zeros([1, 4, doi.shape[0], doi.shape[1]],
                                             dtype=torch_precision).to(
                        device)
                    X_branch_1[:, 0:1, :, :] = torch.real(X1[:, 0:1, :, :])
                    X_branch_1[:, 1:2, :, :] = torch.imag(X1[:, 0:1, :, :])
                    X_branch_1[:, 2:3, :, :] = torch.real(X1[:, 1:2, :, :])
                    X_branch_1[:, 3:4, :, :] = torch.imag(X1[:, 1:2, :, :])

                    X_branch_2 = torch.zeros([1, 2*rs], dtype=torch_precision).to(device)
                    X_branch_2[:, 0:rs] = torch.real(X2)
                    X_branch_2[:, rs:2*rs] = torch.imag(X2)

                    prediction1, prediction2 = model(x11=X_branch_1, x12=X_branch_2)

                    deltaEt_true1 = torch.zeros([1, 2, doi.shape[0], doi.shape[1]],
                                                dtype=torch_precision).to(
                        device)
                    deltaEt_true1[:, 0:1, :, :] = torch.real(y[:, 0:1, :, :])
                    deltaEt_true1[:, 1:2, :, :] = torch.imag(y[:, 0:1, :, :])

                    deltaEt_true2 = torch.zeros([1, 2, doi.shape[0], doi.shape[1]],
                                                dtype=torch_precision).to(
                        device)
                    deltaEt_true2[:, 0:1, :, :] = torch.real(y[:, 1:2, :, :])
                    deltaEt_true2[:, 1:2, :, :] = torch.imag(y[:, 1:2, :, :])

                    loss_validation = 1 / 2 * loss_func(prediction1, deltaEt_true1) \
                                    + 1 / 2 * loss_func(prediction2, deltaEt_true2)
                    visual_dict_validation['loss_submitted'] += loss_validation.item()  # 累加损失值


                validation_loss = visual_dict_validation['loss_submitted'] / len(validation_dataloader)  # 计算平均损失
                loss_valid[0, i] = validation_loss
                t.set_description("Epoch %i" % i)
                t.set_description("Training loss_submitted：")
                t.set_postfix(loss=loss_valid[0, i])
                time.sleep(.01)



            epochs_ran = i + 1
            if not np.isfinite(validation_loss):
                raise RuntimeError(f"Non-finite validation loss at epoch {epochs_ran}: {validation_loss}")

            if validation_loss < best_validation_loss - early_stopping_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epochs_ran
                epochs_without_improvement = 0
                torch.save(model, model_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epochs_ran}.")
                break

    print(f"Best validation loss: {best_validation_loss:.6e} at epoch {best_epoch}.")
    print(f"Best model saved to: {model_path}")
    np.save(f"loss_train_num{num_train_picture}_unet_rv_{rs}rx_pair4.npy", loss_train[:, :epochs_ran])
    np.save(f"loss_validation_num{num_train_picture}_unet_rv_{rs}rx_pair4.npy", loss_valid[:, :epochs_ran])
