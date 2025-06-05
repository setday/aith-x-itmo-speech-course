import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import audio_config

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def get_stft(self, x, fft_size, hop_size, win_length):
        return torch.stft(
            x.squeeze(1), 
            n_fft=fft_size, 
            hop_length=hop_size, 
            win_length=win_length, 
            window=torch.hann_window(win_length).to(x.device),
            normalized=audio_config.signal_norm,
            return_complex=True
        )
        
    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_mag = self.get_stft(x, fft_size, hop_size, win_length).abs()
            y_mag = self.get_stft(y, fft_size, hop_size, win_length).abs()
            
            sc_loss += torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")        
            mag_loss += F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
        
        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)
        
        return sc_loss + mag_loss
