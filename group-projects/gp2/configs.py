from dataclasses import dataclass


@dataclass
class BaseAudioConfig:
    fft_size=1024
    win_length=1024
    hop_length=256
    sample_rate=22050
    power=1.5
    num_mels=80
    mel_fmin=0.0
    mel_fmax=8000.0
    spec_gain=1
    signal_norm=False

@dataclass
class TrainConfig:
    data_path="e:/Projects/AITH/aith-x-itmo-speech-course/tts_models"
    batch_size=8
    num_workers=4
        
    upsample_scales=[4, 4, 4, 4]
        
    epochs=100
    learning_rate=0.0001
    gradient_clip=5.0
        
    checkpoint_dir="checkpoints"
    log_interval=100
    checkpoint_interval=1000
        
    lambda_adv=4.0

    train_seq_length=16000


audio_config = BaseAudioConfig()
