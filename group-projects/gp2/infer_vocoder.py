import os
import argparse

import torch

import torchaudio

from tqdm import tqdm

from parallel_wavegan import PWGGenerator
from train_vocoder import audio_config, Config
from t2spec_converter import TextToSpecConverter
    

def inference(checkpoint_path, text_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generator = PWGGenerator(
        in_channels=audio_config.num_mels, 
        out_channels=1, 
        upsample_scales=Config().upsample_scales,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    t2s = TextToSpecConverter()
    
    with open(text_path, 'r') as f:
        for idx, line in tqdm(enumerate(f), desc="Generating audio"):
            mel_spec = torch.tensor(t2s.text2spec(line.strip()), device=device).T

            with torch.no_grad():
                audio = generator(mel_spec)
            torchaudio.save(os.path.join(output_dir, f"output_{idx}.wav"), audio.cpu(), audio_config.sample_rate)

    print(f"Generated audio saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ParallelWaveGAN vocoder")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint for inference")
    parser.add_argument("--text_path", type=str, help="Path to text file for inference")
    parser.add_argument("--output_dir", type=str, default="./output", help="Path to save generated audio")
    
    args = parser.parse_args()

    if not args.checkpoint_path or not args.text_path:
        print("For inference mode, checkpoint_path and text_path must be provided")
    else:
        inference(args.checkpoint_path, args.text_path, args.output_dir)