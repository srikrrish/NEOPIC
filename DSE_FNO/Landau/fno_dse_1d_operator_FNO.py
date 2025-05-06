import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# class for 1-dimensional Fourier transforms on nonequispaced data, using the adjoint as an approximate inverse
class VandermondeTransform:
    def __init__(self, positions, modes):
        self.modes = modes
        #positions -= torch.min(positions)
        #self.positions = positions * 2 * np.pi / (torch.max(positions))
        #self.positions = positions / (8192) * 2 * np.pi
        self.positions = positions
        self.batch_size = self.positions.shape[0]

        self.Vt, self.Vc = self.make_matrix()

    def make_matrix(self):
        V = torch.zeros([self.batch_size,self.modes,self.positions.shape[1]], dtype=torch.cfloat).cuda()
        for row in range(self.modes):
            #V[:,row,:] = torch.exp(-1j * row * self.positions[:,:])
            V[:,row,:] = torch.exp(-1j * 0.5 * (row - int(self.modes/2)) * self.positions[:,:])

        V_inv = torch.conj(V.clone()).permute(0,2,1)
        #V_inv[0,:] = 0.5

        return V, V_inv

    def forward(self, data):
        #return torch.matmul(data, self.Vt)
        return torch.bmm(self.Vt,data)

    def inverse(self, data):
        #return torch.matmul(data, self.Vc)
        return torch.bmm(self.Vc,data)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d (nn.Module):
    def __init__(self, in_channels, out_channels, modes, transform=None):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.transform = transform

        self.scale = (1 / (in_channels*out_channels))
        #self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, int(self.modes/2)+1, dtype=torch.cfloat))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))
        #self.weights = nn.Linear(self.modes, self.modes, dtype=torch.cfloat)
        #kx = torch.arange(self.modes)
        #kx[0] = 1
        #for k in range(self.modes):
        #    self.weights[:,:,k] = (1 / (1j * kx[k])) * self.weights[:,:,k]


    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, transformer):
        # FNO dse
        x = x.permute(0, 2, 1)
        x_ft = transformer.forward(x.cfloat())
        x_ft = x_ft.permute(0, 2, 1)
        x_ft[:,:,int(self.modes/2)] = 0.0
        #x_ft[:,:,0] = 0.0
        out_ft = self.compl_mul1d(x_ft, self.weights)
        #out_ft = torch.zeros(x.shape[0], self.out_channels, self.modes, dtype=torch.cfloat, device=x.device)
        #out_ft[:,:,:int(self.modes/2)+1] = self.compl_mul1d(x_ft[:,:,:int(self.modes/2)+1], self.weights[:,:,:])
        #out_ft[:,:,int(self.modes/2)+1:self.modes] = self.compl_mul1d(x_ft[:,:,int(self.modes/2)+1:self.modes], torch.flipud(self.weights[:,:,1:int(self.modes/2)]))
        #x_ft = self.weights(x_ft)
        #x_ft = F.tanh(x_ft)
        #out_ft = x_ft
        kx = torch.arange(self.modes)
        kx[int(self.modes/2)] = 1
        #kx[0] = 1
        h = 4*np.pi / 32

        for k in range(self.modes):
            kvec = 0.5 * (kx[k] - int(self.modes/2))
            #kvec = kx[k]
            #out_ft[:,:,k] = (1 / (1j * kvec)) * (torch.sin(kvec*h/2)/(kvec*h/2))**2 * out_ft[:,:,k]
            out_ft[:,:,k] = (1 / (1j * kvec)) * out_ft[:,:,k]

        out_ft = out_ft.permute(0, 2, 1)
        x = transformer.inverse(out_ft)
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d_dd (nn.Module):
    def __init__(self, in_channels, out_channels, modes, transform=None):
        super(SpectralConv1d_dd, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.transform = transform

        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))


    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, transformer):
        # FNO dse
        x = x.permute(0, 2, 1)
        x_ft = transformer.forward(x.cfloat())
        x_ft = x_ft.permute(0, 2, 1)
        out_ft = self.compl_mul1d(x_ft, self.weights)
        out_ft = out_ft.permute(0, 2, 1)
        x = transformer.inverse(out_ft)
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real


class FNO_dse (nn.Module):
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            1000,
        'num_test':             250,
        'batch_size':           20,
        'epochs':               250,
        'test_epochs':          10,

        'datapath':             "_Data/Landau/1d_dt_002_T_2.5/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.001,
        'scheduler_step':       50,
        'scheduler_gamma':      0.5,
        #'learning_rate':        0.005,
        #'scheduler_step':       10,
        #'scheduler_gamma':      0.97,
        'weight_decay':         1e-5,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes':                16,                     # Number of x-modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
        #'width':                1,                     # Number of channels in the convolutional layers
    }
    def __init__(self, configs):
        super(FNO_dse, self).__init__()

        self.modes = configs['modes']
        self.width = configs['width']
        self.padding = 0 # pad the domain if input is non-periodic

        #self.point_data = configs['point_data']

        # Define Structured Matrix Method
        #transform = VandermondeTransform(self.point_data.squeeze(), self.modes)

        #self.fc0 = nn.Linear(1, 8)
        self.fc0 = nn.Linear(1, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        #self.fq0 = nn.Linear(1, self.width)
        self.conv0 = SpectralConv1d(1, 1, self.modes)
        self.conv1 = SpectralConv1d_dd(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d_dd(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d_dd(self.width, self.width, self.modes)
        self.conv4 = SpectralConv1d_dd(self.width, self.width, self.modes)
        ##self.w0 = nn.Conv1d(self.width, self.width, 1)
        #self.w1 = nn.Conv1d(1, self.width, 1)
        #self.w2 = nn.Conv1d(self.width, 1, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)

        #self.fq1 = nn.Linear(self.width,1)
        #self.fc1 = nn.Linear(32, 32)
        #self.fc1 = nn.Linear(8, 1)
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, 1)
        #self.fc1 = nn.Linear(1, 8)
        #self.fc2 = nn.Linear(8, 1)

    def forward(self, x):

        transform = VandermondeTransform(x[:,:], self.modes)
        #x = x[:,:,None]
        #x = x.permute(0, 2, 1)
        q = torch.ones([x.shape[0],1,x.shape[-1]],dtype=torch.float,device=torch.device('cuda'))
        ##q = torch.mul(q.double(), (-1.0/100000))
        q = torch.mul(q.float(), -1.0)
        #q = q.permute(0,2,1)
        #q = self.fq0(q)
        #q = q.permute(0,2,1)


        q = self.conv0(q, transform)

        ##DD part

        #x = q
        x = x[:,:,None]
        #x = x.permute(0, 2, 1)
        #x = self.conv0(x, transform)
        #x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv1(x, transform)
        x2 = self.w1(x)
        x = x1 + x2
        #x = x2
        x = F.gelu(x)

        x1 = self.conv2(x, transform)
        x2 = self.w2(x)
        x = x1 + x2
        #x = x2
        x = F.gelu(x)

        x1 = self.conv3(x, transform)
        x2 = self.w3(x)
        x = x1 + x2
        #x = x2
        x = F.gelu(x)

        x1 = self.conv4(x, transform)
        x2 = self.w4(x)
        x = x1 + x2
        ##x = x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        #q2 = self.w0(q)
        #q = q1 + q2
        #q = q1
        #q = F.gelu(q)

        #q1 = self.conv1(q, transform)
        #q2 = self.w1(q)
        #q = q1 + q2
        #q = F.gelu(q)

        #q1 = self.conv2(q, transform)
        #q2 = self.w2(q)
        #q = q1 + q2
        #q = F.gelu(q)

        #q1 = self.conv3(q, transform)
        #q2 = self.w3(q)
        #q = q1 + q2
        #q = F.tanh(q)

        #x1 = self.conv0(x, transform)
        #x2 = self.w0(x)
        #x = x1 + x2
        #x = F.gelu(x)

        #x1 = self.conv1(x, transform)
        #x2 = self.w1(x)
        #x = x1 + x2
        #x = F.gelu(x)

        #x1 = self.conv2(x, transform)
        #x2 = self.w2(x)
        #x = x1 + x2
        #x = F.tanh(x)

        #x1 = self.conv3(x, transform)
        #x2 = self.w3(x)
        #x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        #x = x.permute(0, 2, 1)
        #q = q.permute(0, 2, 1)
        #x = self.fc0(x[:,:,None])
        #x = F.gelu(x)
        #x = self.fc1(x[:,:,None])
        qf = x.squeeze()+q.squeeze()
        #qf = x.squeeze()
        #q = self.fc1(q)
        #q = F.gelu(q)
        #q = self.fc2(q)
        #qf = q.squeeze()

        return qf

