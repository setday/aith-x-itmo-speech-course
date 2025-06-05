import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchaudio

import numpy as np

from configs import audio_config


class LJSpeechDataset(Dataset):
    def __init__(self, root_dir, segment_size):
        self.ljspeech = torchaudio.datasets.LJSPEECH(root=root_dir, download=True)
        self.segment_size = segment_size
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_config.sample_rate,
            n_fft=audio_config.fft_size,
            win_length=audio_config.win_length,
            hop_length=audio_config.hop_length,
            n_mels=audio_config.num_mels,
            f_min=audio_config.mel_fmin,
            f_max=audio_config.mel_fmax,
            power=audio_config.power,
            normalized=audio_config.signal_norm
        )
        
    def __len__(self):
        return len(self.ljspeech)
    
    def __getitem__(self, idx):
        waveform, sample_rate, _, _ = self.ljspeech[idx]
        
        if sample_rate != audio_config.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, audio_config.sample_rate)(waveform)

        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if waveform.size(1) > self.segment_size:
            start = np.random.randint(0, waveform.size(1) - self.segment_size)
            waveform = waveform[:, start:start + self.segment_size]
        else:
            padding = self.segment_size - waveform.size(1)
            waveform = F.pad(waveform, (0, padding), "constant", 0)
        
        waveform = waveform / torch.max(torch.abs(waveform))
        
        mel_spec = self.mel_transform.forward(waveform)[0].clamp(min=1e-5).log()
        
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
        
        return {
            "waveform": waveform,
            "mel_spec": mel_spec
        }
