# model_utils.py
import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from whisperspeech.vq_stoks import RQBottleneckTransformer

def setup_vq_model(device="cpu"):
    model_path = "whisper-vq-stoks-medium-en+pl-fixed.model"
    if not os.path.exists(model_path):
        hf_hub_download(
            repo_id="jan-hq/WhisperVQ",
            filename=model_path,
            local_dir=".",
        )

    model = RQBottleneckTransformer.load_model(model_path).to(device)
    model.ensure_whisper(device)
    return model

def audio_to_tokens(audio_path, model, target_bandwidth=1.5):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    with torch.no_grad():
        codes = model.encode_audio(wav.to(model.device))
        codes = codes[0].cpu().tolist()
    
    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|sound_start|>{result}<|sound_end|>'
