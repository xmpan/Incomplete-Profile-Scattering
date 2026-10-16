import torch.nn as nn
from _config import *
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Upsample
from torch.utils.data import Dataset

class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.inc = nn.Sequential(
            Conv2d(in_ch, out_ch, kernel_size=3,
                   stride=1, padding=1, dilation=1, groups=1, bias=True),
            BatchNorm2d(out_ch, 5),
            ReLU(),
        )

    def forward(self, x):
        x = self.inc(x)
        return x


class Covn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Covn, self).__init__()
        self.covn = nn.Sequential(
            # Keep the spatial size for skip connections with an even-sized kernel.
            nn.ZeroPad2d((0, 1, 0, 1)),
            Conv2d(in_ch, out_ch, kernel_size=2,
                   stride=1, padding=0, dilation=1, groups=1, bias=True),
            BatchNorm2d(out_ch, 5),
            ReLU(),
        )

    def forward(self, x):
        x = self.covn(x)
        return x


class Covn_Squen(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Covn_Squen, self).__init__()
        self.con_sq = nn.Sequential(
            Conv2d(in_ch, out_ch, kernel_size=3,
                   stride=1, padding=1, dilation=1, groups=1, bias=True),
            BatchNorm2d(out_ch, 5),
            ReLU(),

            Conv2d(out_ch, out_ch, kernel_size=3,
                   stride=1, padding=1, dilation=1, groups=1, bias=True),
            BatchNorm2d(out_ch, 5),
            ReLU(),

            Conv2d(out_ch, out_ch, kernel_size=3,
                   stride=1, padding=1, dilation=1, groups=1, bias=True),
            BatchNorm2d(out_ch, 5),
            ReLU(),
        )

    def forward(self, x):
        x = self.con_sq(x)
        return x


class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            Upsample(scale_factor=2),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outc = nn.Sequential(
            Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
        )

    def forward(self, x):
        x = self.outc(x)
        return x

class rv_unet(nn.Module):
    def __init__(self, in_ch, basis_ch, out_ch):
        super(rv_unet, self).__init__()
        self.inc = Inconv(in_ch, basis_ch)
        self.con_sq1 = Covn_Squen(basis_ch, basis_ch)
        self.down1 = Down()
        self.con_sq2 = Covn_Squen(basis_ch, basis_ch * 2)
        self.down2 = Down()
        self.con_sq3 = Covn_Squen(basis_ch * 2, basis_ch * 4)
        self.down3 = Down()
        self.con_sq4 = Covn_Squen(basis_ch * 4, basis_ch * 8)

        self.up1 = Up()
        self.con1 = Covn(basis_ch * 8, basis_ch * 4)

        self.con_sq5 = Covn_Squen(basis_ch * 8, basis_ch * 4)
        self.up2 = Up()
        self.con2 = Covn(basis_ch * 4, basis_ch*2)

        self.con_sq6 = Covn_Squen(basis_ch * 4, basis_ch*2)

        self.up3 = Up()
        self.con3 = Covn(basis_ch * 2, basis_ch)

        self.con_sq7 = Covn_Squen(basis_ch * 2, basis_ch)

        self.outc = Outconv(basis_ch, out_ch)

    def forward(self, input):
        out = {}
        out['inc'] = input
        # print("input.size:",input.size())
        out['in'] = self.inc(out['inc'])
        # print("out[in]:",out['in'].size())
        out['seq1'] = self.con_sq1(out['in'])
        # print("out[seq1]:",out['seq1'].size())
        out['d1'] = self.down1(out['seq1'])
        # print("out[d1]:",out['d1'].size())
        out['seq2'] = self.con_sq2(out['d1'])
        # print("out[seq2]:",out['seq2'].size())
        out['d2'] = self.down2(out['seq2'])
        # print("out[d2]:",out['d2'].size())
        out['seq3'] = self.con_sq3(out['d2'])
        out['d3'] = self.down3(out['seq3'])
        # print("out[d2]:",out['d2'].size())
        out['seq4'] = self.con_sq4(out['d3'])

        # print("out[seq3]:",out['seq3'].size())
        out['u1'] = self.up1(out['seq4'])
        # print("out[u1]:",out['u1'].size())
        out['up_con1'] = self.con1(out['u1'])
        # print("out[up_con1]:",out['up_con1'].size())
        out['in4'] = torch.cat((out['up_con1'], out['seq3']), dim=1)
        # print("out[in4]:",out['in4'].size())
        out['seq5'] = self.con_sq5(out['in4'])
        # print("out[seq4]:",out['seq4'].size())

        out['u2'] = self.up2(out['seq5'])
        # print("out[u2]:",out['u2'].size())
        out['up_con2'] = self.con2(out['u2'])
        # print("out[up_con2]:",out['up_con2'].size())
        out['in5'] = torch.cat((out['up_con2'], out['seq2']), dim=1)
        # print("out[in5]:",out['in5'].size())
        out['seq6'] = self.con_sq6(out['in5'])

        out['u3'] = self.up3(out['seq6'])
        # print("out[u2]:",out['u2'].size())
        out['up_con3'] = self.con3(out['u3'])
        # print("out[up_con2]:",out['up_con2'].size())
        out['in6'] = torch.cat((out['up_con3'], out['seq1']), dim=1)
        # print("out[in5]:",out['in5'].size())
        out['seq7'] = self.con_sq7(out['in6'])

        # print("out[seq5]:",out['seq5'].size())
        out['pre_out'] = self.outc(out['seq7'])
        # print("out[pre_out]:",out['pre_out'].size())
        # out['out'] = out['inc']+out['pre_out']
        return out['pre_out']

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)


class cnn_Dataset(Dataset):
    def __init__(self, branch_input, target_input1, target_input2, transform=None, target_transform=None):
        self.samples = branch_input
        self.samples = torch.tensor(self.samples.reshape(
            [self.samples.shape[0], self.samples.shape[1], self.samples.shape[2], self.samples.shape[3]]),
            dtype=torch_complex_precision)#.to(device)
        self.targets1 = target_input1
        self.targets1 = torch.tensor(self.targets1.reshape([self.targets1.shape[0], self.targets1.shape[1],
                                                            self.targets1.shape[2], self.targets1.shape[3]]),
                                     dtype=torch_complex_precision)
        self.targets2 = torch.tensor(target_input2,  dtype=torch_complex_precision)
        self.transform = transform
        self.target_transform = target_transform
  #  必须定义这两个函数，可以用符号代替这两个方法的调用
    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        samples = self.samples[idx, :, :, :]
        targets1 = self.targets1[idx, :, :, :]
        targets2 = self.targets2[idx, :]
        return samples, targets1, targets2

class cnn_Dataset_noes(Dataset):
    def __init__(self, branch_input, target_input1, transform=None, target_transform=None):
        self.samples = branch_input
        self.samples = torch.tensor(self.samples.reshape(
            [self.samples.shape[0], self.samples.shape[1], self.samples.shape[2], self.samples.shape[3]]),
            dtype=torch_complex_precision)#.to(device)
        self.targets1 = target_input1
        self.targets1 = torch.tensor(self.targets1.reshape([self.targets1.shape[0], self.targets1.shape[1],
                                                            self.targets1.shape[2], self.targets1.shape[3]]),
                                     dtype=torch_complex_precision)
        self.transform = transform
        self.target_transform = target_transform
  #  必须定义这两个函数，可以用符号代替这两个方法的调用
    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        samples = self.samples[idx, :, :, :]
        targets1 = self.targets1[idx, :, :, :]
        return samples, targets1


class unet_pair3(nn.Module):
    def __init__(self, inch_forward, basis_forward, out_forward, rs, doi):
        super(unet_pair3, self).__init__()
        self.branch_forward_11 = rv_unet(in_ch=inch_forward, basis_ch=basis_forward, out_ch=out_forward)
        self.branch_forward_12 = nn.Linear(in_features=rs*2, out_features=doi.shape[0] * doi.shape[1] * 2)
        self.aa = nn.ReLU()
        self.branch_forward_2 = rv_unet(in_ch=inch_forward, basis_ch=basis_forward, out_ch=out_forward)
        self.doi = doi

    def forward(self, x11, x12):
        x12 = self.aa(self.branch_forward_12(x12))
        x12 = x12.reshape(x12.shape[0], 2, self.doi.shape[0], self.doi.shape[1])
        x1 = torch.cat((x11, x12), axis=1)
        x1 = self.branch_forward_11(x1)
        x1_out = x1 + x11[:, 0:2, :, :]
        x2 = torch.cat((x11, x1), axis=1)
        x2 = self.branch_forward_2(x2)
        x2_out = x2 + x11[:, 2:4, :, :]
        return x1_out, x2_out

class unet_pair1(nn.Module):
    def __init__(self, inch_forward, basis_forward, out_forward, rs, doi):
        super(unet_pair1, self).__init__()
        self.branch_forward_12 = nn.Linear(in_features=rs*2, out_features=doi.shape[0] * doi.shape[1] * 2)
        self.aa = nn.ReLU()
        self.branch_forward_11 = rv_unet(in_ch=inch_forward, basis_ch=basis_forward, out_ch=out_forward)
        self.branch_forward_2 = rv_unet(in_ch=inch_forward, basis_ch=basis_forward, out_ch=out_forward)
        self.doi = doi

    def forward(self, x11, x12):
        x12 = self.aa(self.branch_forward_12(x12))
        x12 = x12.reshape(x12.shape[0], 2, self.doi.shape[0], self.doi.shape[1])
        x1 = torch.cat((x11, x12), axis=1)
        x1 = self.branch_forward_11(x1)
        x1_out = x1
        x2 = torch.cat((x11, x1), axis=1)
        x2 = self.branch_forward_2(x2)
        x2_out = x2
        return x1_out, x2_out

class unet_pair2(nn.Module):
    def __init__(self, inch_forward, basis_forward, out_forward, doi):
        super(unet_pair2, self).__init__()
        self.branch_forward_11 = rv_unet(in_ch=inch_forward, basis_ch=basis_forward, out_ch=out_forward)
        self.branch_forward_12 = rv_unet(in_ch=inch_forward+2, basis_ch=basis_forward, out_ch=out_forward)
        self.doi = doi

    def forward(self, x11):
        x1 = self.branch_forward_11(x11)
        x1_out = x1 + x11[:, 0:2, :, :]
        x2 = torch.cat((x11, x1), axis=1)
        x2 = self.branch_forward_12(x2)
        x2_out = x2 + x11[:, 2:4, :, :]
        return x1_out, x2_out

class unet_pair4(nn.Module):
    def __init__(self, inch_forward, basis_forward, out_forward, rs, doi):
        super(unet_pair4, self).__init__()
        self.branch_forward_12 = nn.Linear(in_features=rs*2, out_features=doi.shape[0] * doi.shape[1] * 2)
        self.aa = nn.ReLU()
        self.branch_forward_11 = rv_unet(in_ch=inch_forward, basis_ch=basis_forward, out_ch=out_forward)
        self.branch_forward_2 = rv_unet(in_ch=inch_forward-2, basis_ch=basis_forward, out_ch=out_forward)
        self.doi = doi

    def forward(self, x11, x12):
        x12 = self.aa(self.branch_forward_12(x12))
        x12 = x12.reshape(x12.shape[0], 2, self.doi.shape[0], self.doi.shape[1])
        x1 = torch.cat((x11, x12), axis=1)
        x1 = self.branch_forward_11(x1)
        x1_out = x1 + x11[:, 0:2, :, :]
        x2 = torch.cat((x11[:, 2:4, :, :], x1_out), axis=1)
        x2 = self.branch_forward_2(x2)
        x2_out = x2
        return x1_out, x2_out
