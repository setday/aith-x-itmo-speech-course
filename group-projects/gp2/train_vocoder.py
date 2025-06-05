import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchaudio

from tqdm import tqdm

from parallel_wavegan import PWGGenerator, PWGDiscriminator
from datasets import LJSpeechDataset
from mr_stft_loss import MultiResolutionSTFTLoss
from configs import audio_config, TrainConfig as Config


def train(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = PWGGenerator(
        in_channels=audio_config.num_mels, 
        out_channels=1, 
        upsample_scales=config.upsample_scales
    ).to(device)
    
    discriminator = PWGDiscriminator().to(device)
    
    optim_g = optim.Adam(generator.parameters(), lr=config.learning_rate)
    optim_d = optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    
    dataset = LJSpeechDataset(config.data_path, config.train_seq_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_lengths=[512, 1024, 2048]
    ).to(device)
    
    generator.train()
    discriminator.train()
    
    step = 0
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in tqdm(dataloader):
            waveform = batch["waveform"].to(device)
            mel_spec = batch["mel_spec"].to(device)
            
            optim_d.zero_grad()
            
            d_real = discriminator(waveform)
            fake_audio = generator(mel_spec)[:, :, :config.train_seq_length]
            d_fake = discriminator(fake_audio.detach())
            
            loss_d_real = torch.mean((d_real - 1.0) ** 2) # MSE loss for real
            loss_d_fake = torch.mean(d_fake ** 2) # MSE loss for fake
            loss_d = loss_d_real + loss_d_fake
            
            loss_d.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), config.gradient_clip)
            optim_d.step()
            
            optim_g.zero_grad()
            
            d_fake = discriminator(fake_audio)
            adv_loss = torch.mean((d_fake - 1.0) ** 2) # MSE loss for adversarial
            
            recon_loss = stft_loss(fake_audio, waveform)
            
            loss_g = recon_loss + config.lambda_adv * adv_loss
            
            loss_g.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), config.gradient_clip)
            optim_g.step()
            
            if step % config.log_interval == 0:
                print(f"Step {step}, Gen Loss: {loss_g.item():.4f}, Disc Loss: {loss_d.item():.4f}")
                
                if step % (config.log_interval * 10) == 0:
                    with torch.no_grad():
                        sample_mel = mel_spec[0:1]
                        sample_wav = generator(sample_mel).squeeze(0)
                        
                        torchaudio.save(
                            f"{config.checkpoint_dir}/sample_{step}.wav",
                            sample_wav.cpu(),
                            audio_config.sample_rate
                        )
            
            if step % config.checkpoint_interval == 0:
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                }, f"{config.checkpoint_dir}/checkpoint_{step}.pt")
            
            step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ParallelWaveGAN vocoder")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--data_path", type=str, help="Path to LJSpeech dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    config = Config()
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.epochs = args.epochs
    if args.data_path: config.data_path = args.data_path
    if args.checkpoint_dir: config.checkpoint_dir = args.checkpoint_dir
    
    train(config)