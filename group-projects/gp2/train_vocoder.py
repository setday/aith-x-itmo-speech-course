import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from parallel_wavegan import PWGGenerator, PWGDiscriminator
from datasets import LJSpeechDataset
from mr_stft_loss import MultiResolutionSTFTLoss
from configs import audio_config, TrainConfig as Config


torch.set_float32_matmul_precision('medium')

class ParallelWaveGANModel(pl.LightningModule):
    def __init__(self, config, audio_config):
        super().__init__()

        self.save_hyperparameters()
        self.config = config
        self.audio_config = audio_config
        
        self.generator = PWGGenerator(
            in_channels=audio_config.num_mels, 
            out_channels=1, 
            upsample_scales=config.upsample_scales
        )
        
        self.discriminator = PWGDiscriminator()
        
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048]
        )
        
        self.automatic_optimization = False
    
    def forward(self, mel_spec):
        return self.generator(mel_spec)

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        mel_spec = batch["mel_spec"]
        
        opt_d, opt_g = self.optimizers()
        
        opt_d.zero_grad()
        
        d_real = self.discriminator(waveform)
        fake_audio = self.generator(mel_spec)[:, :, :self.config.train_seq_length]
        d_fake = self.discriminator(fake_audio.detach())

        loss_d_real = F.mse_loss(d_real, torch.ones_like(d_real))
        loss_d_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake))
        loss_d = loss_d_real + loss_d_fake
        
        self.manual_backward(loss_d)
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.gradient_clip)
        opt_d.step()
        
        opt_g.zero_grad()

        d_fake = self.discriminator(fake_audio)
        adv_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        recon_loss = self.stft_loss(fake_audio, waveform)
        
        loss_g = recon_loss + self.config.lambda_adv * adv_loss
        
        self.manual_backward(loss_g)
        nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.gradient_clip)
        opt_g.step()
        
        self.log_dict({
            "loss_d": loss_d,
            "loss_g": loss_g,
            "adv_loss": adv_loss,
            "recon_loss": recon_loss
        }, prog_bar=True, on_step=True, logger=True, on_epoch=True)
        
        if self.global_step % self.config.log_interval == 0:
            self._log_audio(mel_spec, fake_audio)
            
        return {"loss": loss_g}

    def _log_audio(self, mel_spec, fake_audio):
        with torch.no_grad():
            sample_mel = mel_spec[0:1]
            sample_wav = self.generator(sample_mel).squeeze(0)
            
            # os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            # filename = f"{self.config.checkpoint_dir}/sample_{self.global_step}.wav"
            # torchaudio.save(
            #     filename,
            #     sample_wav.cpu(),
            #     self.audio_config.sample_rate
            # )
            
            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.add_audio(
                    f"generated_audio_{self.global_step}", 
                    sample_wav.unsqueeze(0), 
                    self.global_step, 
                    self.audio_config.sample_rate
                )
    
    def configure_optimizers(self):
        optim_d = optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)
        optim_g = optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        
        return [optim_d, optim_g], []


def train(config):
    dataset = LJSpeechDataset(config.data_path, config.train_seq_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    model = ParallelWaveGANModel(config, audio_config)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename='wavegan-{step}',
        save_top_k=5,
        monitor='loss_g',
        every_n_train_steps=config.checkpoint_interval
    )
    
    pl.Trainer(
        max_epochs=config.epochs,
        accelerator='auto',
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger("lightning_logs", name="wavegan"),
        log_every_n_steps=5,
    ).fit(model, dataloader)


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