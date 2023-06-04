import torch
from torch import nn 
def periodic_padding(input:torch.Tensor,padding:int):
    if len(input.size()) !=4:
        print("The tenor does not fit the size!")
        return 
    else:
        M1 = torch.cat([input[:,:,:, -padding: ],input,input[:,:,:, 0:padding ]],dim=-1)
        M1 = torch.cat([M1[:,:, -padding: ,:],M1,M1[:,:, 0:padding ,:]],dim=-2)
        return M1


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size,2)
        stack = [t_t.contiguous().view(batch_size, d_height,d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


def PSD(data):
    import numpy as np
    

    # Computational box and dimensions of DNS daa
    Nx = 256
    Nz  = 256
    Lx  = 12
    Lz  = 6

    # Wavenumber spacing
    dkx = 2*np.pi/Lx
    dkz = 2*np.pi/Lz

    # Creating the wavenumber grid. The fftfreq returns a one dimensional array containing the wave vectors
    # for the fftn in the correct order. Since this is a fraction of 1 returned, we multiply by N to get 
    # a pixel frequency and also dk to get a physical meaning and into wave domain?
    # kx = dkx * np.fft.fftfreq(Nx) * Nx
    # kz = dkz * np.fft.fftfreq(Nz) * Nz
    # [kkx,kkz]=np.meshgrid(kx,kz)
    x_range=np.linspace(1,Nx,Nx)
    z_range=np.linspace(1,Nz,Nz)
    kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
    kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])
    [kkx,kkz]=np.meshgrid(kx,kz)
    kkx_norm= np.sqrt(kkx**2)
    kkz_norm = np.sqrt(kkz**2)

    # We convert to wavelength, however since the DC components creates a division by zero, we ignore the 
    # error and set to zero if the division was zero.
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Lambda_x = (2*np.pi/kkx_norm)*u_tau/nu
    Lambda_z = (2*np.pi/kkz_norm)*u_tau/nu
    # calculating wavelength in plus units


    # It doesn't matter if the mean is subtracted as far as I can tell
    Theta_fluc_targ=data-np.mean(data)
  

    # We compute the 2 dimensional discrete Fourier Transform
    fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
    # fourier_image_pred = np.fft.fftn(Theta_fluc_pred)

    # The now contains complex valued amplitudes of all Fourier components. We are only interested in
    # the size of the amplitudes. We will further assume that the average amplitude is zero, so 
    # that we only require the square of the amplitudes to compute the variances.
    # We also compute the pre-multiplication with the wavenumber vectors
    fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0)*kkx*kkz
    # fourier_amplitudes_pred = np.mean(np.abs(fourier_image_pred)**2,axis=0)*kkx*kkz

    # We remove the DC component (It fucks up the plots), and we remove the negative symmetric part since
    # for real signals signals, the coefficients of positive and negative frequencies become complex conjugates.
    # That means, we do not need both sides of spectrum to represent the signal, a single side will do. This is
    #  known as Single Side Band (SSB) spectrum (We might need to multiply by 2, https://medium.com/analytics-vidhya/breaking-down-confusions-over-fast-fourier-transform-fft-1561a029b1ab)
    # however since the do not use the magnitude of the power
    # fourier_amplitudes_targ = fourier_amplitudes_targ
    # fourier_amplitudes_pred = fourier_amplitudes_pred
    Lambda_x = Lambda_x
    Lambda_z = Lambda_z

    return fourier_amplitudes_targ, Lambda_x, Lambda_z