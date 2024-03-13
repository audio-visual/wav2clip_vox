import numpy as np
import torch

from .model.encoder import ResNetExtractor


MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"


def get_model(device="cpu", pretrained=True, url=None, frame_length=None, hop_length=None):
    if pretrained:
        if url is None:
            checkpoint = torch.hub.load_state_dict_from_url(
                MODEL_URL, map_location=device, progress=True
            ) 
        else:
            checkpoint = url
        model = ResNetExtractor(
            checkpoint=checkpoint,
            scenario="frozen",
            transform=True,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    else:
        model = ResNetExtractor(
            scenario="supervise", frame_length=frame_length, hop_length=hop_length
        )
    model.to(device)
    return model


def embed_audio(audio, model):
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=0)
    return (
        model(torch.from_numpy(audio).to(next(model.parameters()).device))
        .detach()
        .cpu()
        .numpy()
    )
