import torch
import librosa
from transformers import AutoProcessor

import sys
from pathlib import Path
# Add parent directory to sys.path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config import DEVICE, MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, DTYPE, LOAD_IN_4BIT, MAX_AUDIO_SECONDS

class QwenOmniRunner:
    """
    Runner for Qwen Omni models:
    - Qwen-2.5-Omni-3B
    - Qwen-2.5-Omni-7B
    - Qwen3-Omni-30B-A3B-Thinking
    - Qwen3-Omni-30B-A3B-Instruct
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.processor = None
        self.model = None

    def load(self):
        """
        Loads the Qwen Omni processor and model into memory.
        """
        print(f"[QwenOmniRunner] Loading processor from {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        print(f"[QwenOmniRunner] Loading model from {self.model_path} to {DEVICE}...")
        
        # Determine dtype
        torch_dtype = getattr(torch, DTYPE, torch.float16)
        
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch_dtype
        }
        
        if LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        if "Qwen3" in self.model_path:
            from transformers import Qwen3OmniMoeForConditionalGeneration as OmniModel
        else:
            from transformers import Qwen2_5OmniForConditionalGeneration as OmniModel
            
        self.model = OmniModel.from_pretrained(
            self.model_path, 
            **kwargs
        )
        self.model.eval()

    def run(self, audio_path: str, prompt: str = "Transcribe the following audio:") -> str:
        """
        Runs inference on the provided audio file.
        
        Args:
            audio_path (str): The absolute path to the audio file.
            prompt (str): Optional text prompt guiding the model.
            
        Returns:
            str: The generated transcription or response.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load() before run().")
            
        # Determine target sampling rate from the processor, fallback to 16000
        target_sr = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        
        # Load audio using librosa
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # Truncate if audio is too long
        max_samples = int(MAX_AUDIO_SECONDS * sr)
        if len(audio) > max_samples:
            print(f"[QwenOmniRunner] Truncating audio to {MAX_AUDIO_SECONDS}s")
            audio = audio[:max_samples]
        
        # Prepare inputs (Qwen Omni series handles audio multimodal inputs)
        # Format depends slightly on the exact Qwen version, but typically AutoProcessor handles it via the `audios` kwarg
        try:
            # Standard Qwen-Audio processor usage
            inputs = self.processor(
                text=prompt, 
                audios=audio, 
                return_tensors="pt"
            )
        except TypeError:
            # Fallback if processor expects a different format (e.g. some Qwen-Omni models)
            inputs = self.processor(
                text=prompt, 
                audio=audio, 
                sampling_rate=sr,
                return_tensors="pt"
            )
            
        # Move tensors to appropriate device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE
            )
            
        # Decode the output
        # Depending on the model, we might need to strip the prompt length
        input_length = inputs.get("input_ids", [[]])[0].shape[0] if "input_ids" in inputs else 0
        generated_ids = outputs[0][input_length:] if input_length > 0 and outputs[0].shape[0] > input_length else outputs[0]
        
        transcription = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        return transcription
