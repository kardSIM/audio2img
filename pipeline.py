import numpy as np
import torch
import torchaudio
import librosa


class AudioToImagePipeline:
    def __init__(self, audio_encoder, feature_extractor, diffusion_pipeline, device):
        self.feature_extractor = feature_extractor
        self.audio_encoder = audio_encoder
        self.device = device
        self.diffusion_pipeline = diffusion_pipeline

    def __call__(self, audio_file):
        with torch.no_grad():
            waveform, sample_rate = self.load_audio(audio_file)
            white_noise = self.get_white_noise()
            bos_emb = self.get_bos().to(self.device)
            prompt_embeds = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", truncation=True, padding='max_length', max_length=375)
            prompt_embeds = self.audio_encoder(prompt_embeds.input_features).last_hidden_state.to(self.device)
            negative_prompt_embeds = self.feature_extractor(white_noise, sampling_rate=16000, return_tensors="pt", truncation=True, padding='max_length', max_length=375)
            negative_prompt_embeds = self.audio_encoder(negative_prompt_embeds.input_features).last_hidden_state.to(self.device)
            prompt_embeds = torch.cat((bos_emb[:,0,:].unsqueeze(1), prompt_embeds), dim=1)
            negative_prompt_embeds = torch.cat((bos_emb[:,0,:].unsqueeze(1), negative_prompt_embeds), dim=1)         
            image = self.diffusion_pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds).images[0]
        return image

    def load_audio(self, audio_file):
        if audio_file.lower().endswith('.mp3'):
            waveform, sample_rate = librosa.load(audio_file, sr=16000, mono=True)
            waveform = torch.from_numpy(waveform).unsqueeze(0) 
        elif audio_file.lower().endswith('.wav'):       
            waveform, sample_rate = torchaudio.load(audio_file)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)          
        else:
            raise ValueError("Unsupported audio format.")
        return waveform, sample_rate

    def get_bos(self):
        text_encoder = self.diffusion_pipeline.text_encoder
        tokenizer = self.diffusion_pipeline.tokenizer
        bos_token = tokenizer("", return_tensors="pt", truncation=True, padding='max_length', max_length=48)
        bos_emb = text_encoder(**bos_token).last_hidden_state
        return bos_emb

    def get_white_noise(self):
        duration, samplerate, amplitude = 4.0, 16000, 0.01      
        white_noise = amplitude * np.random.randn(int(duration * samplerate))    
        zero_noise = np.zeros(int(duration * samplerate))
        return white_noise
