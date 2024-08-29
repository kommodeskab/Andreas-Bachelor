from torch.utils.data import Dataset
import torchaudio
import os
import torch

class Crawler(Dataset):
    def __init__(self, path : str):
        self.path = path
        self.files = os.listdir(path)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        waveform, sample_rate = torchaudio.load(os.path.join(self.path, filename))
        return waveform, sample_rate
    
class FormattedAudioCrawler(Crawler):
    def __init__(
        self, path : str, 
        audio_length : int = 4, 
        random_crop : bool = False, 
        sample_rate : int = 16000, 
        ):
        super().__init__(path = path)
        self.audio_length = audio_length
        self.random_crop = random_crop
        self.sample_rate = sample_rate
        
    def __getitem__(self, idx : int):
        waveform, old_sample_rate = super().__getitem__(idx)
        waveform = torchaudio.transforms.Resample(old_sample_rate, self.sample_rate)(waveform)
        
        seq_length = waveform.shape[1]
        new_seq_length = self.audio_length * self.sample_rate
        if seq_length > new_seq_length:
            # random crop
            start = torch.randint(0, seq_length - new_seq_length, (1,)).item() if self.random_crop else 0
            waveform = waveform[:, start:start + new_seq_length]
        else:
            padding_size = new_seq_length - seq_length
            waveform = torch.nn.functional.pad(waveform, (0, padding_size))
        
        return waveform