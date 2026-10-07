import os
import json
import io
import soundfile as sf
from datasets import load_dataset, Audio
from pathlib import Path

# Fix related to soundfile/libsndfile missing issues on some systems, 
# although we just try to use soundfile and if not installed we will have installed it.

DATASET_DIR = Path("/mnt/data/multilingual-audio-models/data/fleurs_hi_test")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading google/fleurs (hi_in) - 10 samples (Streaming mode)...")
ds = load_dataset("google/fleurs", "hi_in", split="test", streaming=True)
# Bypass the HuggingFace torchcodec requirement by turning off automatic decoding
ds = ds.cast_column("audio", Audio(decode=False))

metadata = []
for i, item in enumerate(ds):
    if i >= 10:
        break
    
    # We get raw bytes now, so we manually decode it using soundfile
    raw_audio_bytes = item['audio']['bytes']
    audio_data, sr = sf.read(io.BytesIO(raw_audio_bytes))
    
    transcription = item['raw_transcription']
    
    audio_path = DATASET_DIR / f"sample_{i:03d}.wav"
    sf.write(str(audio_path), audio_data, sr)
    
    metadata.append({
        "id": i,
        "audio_path": str(audio_path),
        "reference_text": transcription
    })

metadata_path = DATASET_DIR / "metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print(f"Successfully saved 10 audio samples and metadata.json to {DATASET_DIR}")
