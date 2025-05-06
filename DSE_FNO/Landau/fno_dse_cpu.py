

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



################################################################
# FNO_dse (VFT, SpectralConv2d_dse, FNO_dse same as Elasticity)
################################################################

# class for fully nonequispaced 2d points
class VFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        # because we don't mutliply by 2*pi/L below
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions))
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float()
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes-1), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float()


        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2-1)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2-1), 1)
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (X_mat+Y_mat))

        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)

        return forward_mat, inverse_mat

    def forward(self, data):
        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        data_inv = torch.bmm(self.V_inv, data)

        return data_inv


class SpectralConv2d_dse (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_dse, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        #self.weights2 = nn.Parameter(
        #    self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, transformer):
        batchsize = x.shape[0]
        num_pts = x.shape[-1]

        x = x.permute(0, 2, 1)

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft[:,0,:] = 0.0
        x_ft = x_ft.permute(0, 2, 1)
        # out_ft = self.compl_mul1d(x_ft, self.weights3)
        #x_ft = torch.reshape(x_ft, (batchsize, self.out_channels, 2*self.modes1, 2*self.modes1-1))
        x_ft = torch.reshape(x_ft, (batchsize, self.in_channels, 2*self.modes1, 2*self.modes1-1))

        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, self.modes1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes1] = -1j*self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes1], self.weights1.imag)
        ## Seems weight1 and weight2 are for positive and negative modes (in x or first dimension) but why different weights?
        #out_ft[:, :, -self.modes1:, :self.modes1] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes1], self.weights2)
        out_ft[:, 0, -self.modes1:, :self.modes1] = (-1j*self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes1], -self.weights1.imag))[:,0,:,:]
        out_ft[:, 1, -self.modes1:, :self.modes1] = (-1j*self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes1], self.weights1.imag))[:,1,:,:]

        # #Return to physical space
        ## Take advantage of real input data and the FFT has complex conjugate symmetry and hence the flip and conj
        x_ft = torch.reshape(out_ft, (batchsize, self.out_channels, 2*self.modes1**2))
        x_ft2 = x_ft[..., 2*self.modes1:].flip(-1, -2).conj()
        x_ft = torch.cat([x_ft, x_ft2], dim=-1)

        x_ft = x_ft.permute(0, 2, 1)
        x = transformer.inverse(x_ft) # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real


class FNO_dse (nn.Module):
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            1000,
        'num_test':             250,
        'batch_size':           20,
        'epochs':               51,
        'test_epochs':          10,

        'datapath':             "_Data/Landau/dt_002_T_2.5/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.001,
        'scheduler_step':       50,
        'scheduler_gamma':      0.5,
        #'learning_rate':        0.005,
        #'scheduler_step':       10,
        #'scheduler_gamma':      0.97,
        'weight_decay':         1e-4,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes1':               16,                     # Number of x-modes to use in the Fourier layer
        'modes2':               16,                     # Number of y-modes to use in the Fourier layer
        'width':                2,                     # Number of channels in the convolutional layers
    }
    def __init__(self, configs):
        super(FNO_dse, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.padding = 0 # pad the domain if input is non-periodic


        #self.fc0 = nn.Linear(1, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_dse(1, self.width, self.modes1, self.modes2)
        #self.conv1 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        #self.conv2 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        #self.conv3 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        #self.w0 = nn.Conv1d(self.width, self.width, 1)
        #self.w1 = nn.Conv1d(self.width, self.width, 1)
        #self.w2 = nn.Conv1d(self.width, self.width, 1)
        #self.w3 = nn.Conv1d(self.width, self.width, 1)

        #self.fc1 = nn.Linear(self.width, 128)
        #self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        transform = VFT(x[:,:,0], x[:,:,1], self.modes1)

        #x = self.fc0(x)
        x = x.permute(0, 2, 1)
        q = torch.ones([x.shape[0],1,x.shape[-1]],dtype=torch.float,device=torch.device('cpu'))
        #q = torch.mul(q.double(), (-1.0/100000))
        q = torch.mul(q.float(), -1.0)
        #q = q * (-1/100000)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        q = self.conv0(q, transform)
        q = q.permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        #x1 = self.conv0(x, transform)
        #x2 = self.w0(x)
        #x = x1 + x2
        #x = x1
        #x = F.gelu(x)

        #x1 = self.conv1(x, transform)
        #x2 = self.w1(x)
        #x = x1 + x2
        #x = F.gelu(x)

        #x1 = self.conv2(x, transform)
        #x2 = self.w2(x)
        #x = x1 + x2
        #x = F.gelu(x)

        #x1 = self.conv3(x, transform)
        #x2 = self.w3(x)
        #x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        #x = x.permute(0, 2, 1)
        #x = x.permute(0, 2, 3, 1)
        #x = self.fc1(x)
        #x = F.gelu(x)
        #x = self.fc2(x)

        return q

