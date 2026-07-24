import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr

from whisper_normalizer.english import EnglishTextNormalizer

class Parakeet(nn.Module):
    def __init__(self, model_path: str = "your_asr_model_path", device: str = 'cuda'):
        super().__init__()

        self.model = nemo_asr.models.ASRModel.restore_from(model_path)
        self.model.to(device)
        self.model.eval()
        self.normalizer = EnglishTextNormalizer()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        if hasattr(self.model, "disable_cuda_graphs"):
            self.model.disable_cuda_graphs()
            
    def get_embeddings(self, audio_signal):
        """
        :param audio_signal: [B, T_raw]
        """
        length = torch.tensor([audio_signal.shape[-1]] * audio_signal.shape[0]).to(self.model.device)
        encoded, encoded_len = self.model.forward(
            input_signal=audio_signal, input_signal_length=length
        )
        logits = self.model.ctc_decoder.decoder_layers(encoded).transpose(1, 2)
        
        return {
            "logits": logits, 
            "embeddings": None, 
            "output_lengths": None
        }

    def transcribe(self, audio_signal):
        """
        :param audio_signal: [B, T_raw]
        """
        audio_list = [t for t in audio_signal]
        hyp = self.model.transcribe(audio=audio_list, batch_size=len(audio_list), verbose=False)
        text = [self.normalizer(t.text) for t in hyp]
        
        return text
