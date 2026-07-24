import random
import torch
import torch.utils.data
import librosa

def load_wav(full_path, sample_rate):
    data, _ = librosa.load(full_path, sr=sample_rate, mono=True)
    return data

def get_dataset_filelist(input_file_list):
    with open(input_file_list, 'r') as fi:
        files = [x for x in fi.read().split('\n') if len(x) > 0]

    return files

class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, sampling_rate, hop_size, ratio, train=True, shuffle=True, n_cache_reuse=0, device=None, rank=0):
        self.audio_files = training_files
        random.seed(1234 + rank)  # Use rank-specific seed for shuffling
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.ratio = ratio
        self.train = train
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.rank = rank
        

    def __getitem__(self, index):
        filename = self.audio_files[index]

        if self._cache_ref_count == 0:
            audio = load_wav(filename, self.sampling_rate)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)  # [T]
        audio = audio.unsqueeze(0)  # [1, T]

        if audio.size(1) <= self.segment_size:
            pad_length = self.segment_size - audio.size(1)
            padding_tensor = audio.repeat(1, 1 + pad_length // audio.size(1))
            audio = torch.cat((audio, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start: audio_start + self.segment_size] #[1,T]
        else:
            audio = audio[:, : self.segment_size]

        return audio.squeeze() #[batch_size, T]

    def __len__(self):
        return len(self.audio_files)
