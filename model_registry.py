import os
from pathlib import Path

BASE_MODELS_DIR = Path("/mnt/data/multilingual-audio-models/models")

def _get_model_path(model_name: str) -> str:
    model_dir = BASE_MODELS_DIR / model_name
    if not model_dir.exists():
        return str(model_dir)

    subdirs = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    has_files_in_root = any((model_dir / f).exists() for f in ["config.json", "model.safetensors", "pytorch_model.bin"])
    
    if not has_files_in_root and len(subdirs) > 0:
        for subdir in subdirs:
            if any((subdir / f).exists() for f in ["config.json", "model.safetensors", "pytorch_model.bin"]):
                return str(subdir)
        if len(subdirs) == 1:
            return str(subdirs[0])
            
    return str(model_dir)

def _determine_family_and_runner(name: str):
    name_lower = name.lower()
    if "qwen" in name_lower:
        return "Qwen", "QwenRunner"
    elif "whisper" in name_lower:
        return "Whisper", "WhisperRunner"
    elif "moss" in name_lower:
        return "MOSS", "MOSSRunner"
    elif "gemma" in name_lower:
        return "Gemma", "GemmaRunner"
    elif "moonshine" in name_lower:
        return "Moonshine", "MoonshineRunner"
    elif "aero" in name_lower:
        return "Aero", "AeroRunner"
    elif "flamingo" in name_lower:
        return "Flamingo", "FlamingoRunner"
    elif "phi" in name_lower:
        return "Phi", "PhiRunner"
    elif "seamless" in name_lower:
        return "Seamless", "SeamlessRunner"
    elif "ultravox" in name_lower:
        return "Ultravox", "UltravoxRunner"
    elif "voxtral" in name_lower:
        return "Voxtral", "VoxtralRunner"
    elif "canary" in name_lower or "parakeet" in name_lower:
        return "Nvidia", "NvidiaRunner"
    elif "granite" in name_lower:
        return "Granite", "GraniteRunner"
    elif "wave2vec" in name_lower:
        return "Wave2Vec", "Wave2VecRunner"
    elif "kimi" in name_lower:
        return "Kimi", "KimiRunner"
    elif "gpt" in name_lower:
        return "GPT", "GPTRunner"
    else:
        return "Unknown", "DefaultRunner"

models_list = [
    "Aero-1-Audio",
    "Audio-Flamingo-Next",
    "GPT-Audio-mini",
    "MOSS-Audio",
    "MOSS-Audio-4B-Instruct",
    "MOSS-Audio-4B-Thinking",
    "MOSS-Audio-8B-Instruct",
    "MOSS-Audio-8B-Thinking",
    "Phi-4-Multimodal-Instruct",
    "Qwen-2.5-Omni-3B",
    "Qwen-2.5-Omni-7B",
    "Qwen2-Audio-7B",
    "Whisper + Gemma 2 9B",
    "Whisper + Llama 3.1 8B",
    "Whisper + Mistral 7B",
    "Whisper+Qwen-2.5-7B",
    "gemma-3n-E2B",
    "gemma-3n-E4B",
    "gemma-4-12b-unified",
    "ibm-granite-speech-8b",
    "kimi-audio-7b",
    "moonshine-base",
    "moonshine-tiny",
    "nvidia-canary-qwen-2.5b",
    "nvidia-parakeet-tdt-1.1b",
    "qwen3-asr-1.7b",
    "qwen3-omni-30b-a3b-instruct",
    "qwen3-omni-30b-a3b-thinking",
    "seamless-m4t-v2-large",
    "ultravox-v0.6-32b",
    "voxtral-mini-3b-2507",
    "voxtral-small-24b",
    "wave2vec-2.0",
    "whisper-large-v3",
    "whisper-v3-large-turbo"
]

MODEL_CATALOGUE = {}
for m in models_list:
    family, runner = _determine_family_and_runner(m)
    MODEL_CATALOGUE[m] = {
        "name": m,
        "family": family,
        "runner": runner,
        "path": _get_model_path(m)
    }

def get_model(name: str) -> dict:
    """
    Retrieve the model details from the catalogue.
    """
    if name not in MODEL_CATALOGUE:
        raise ValueError(f"Model '{name}' not found in registry.")
    return MODEL_CATALOGUE[name]
