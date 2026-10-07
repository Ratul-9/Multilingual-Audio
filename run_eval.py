import argparse
import json
import time
from pathlib import Path

from config import RESULTS_DIR
from model_registry import get_model

def get_runner(runner_name: str, model_path: str):
    """
    Factory function to initialize and return the appropriate runner.
    (Runners will be fully implemented later in the runners/ directory).
    """
    if runner_name == "QwenRunner":
        from runners.qwen_omni import QwenOmniRunner
        return QwenOmniRunner(model_path)
    
    # TODO: Add other runners here as they are implemented
    
    class DummyRunner:
        """A placeholder runner for structural testing."""
        def __init__(self, name, path):
            self.name = name
            self.path = path
            
        def load(self):
            print(f"[Runner] Loading model from '{self.path}' using '{self.name}'...")
            
        def run(self, audio_path: str, prompt: str = ""):
            # Placeholder for inference logic
            return "Transcription placeholder text"
            
    return DummyRunner(runner_name, model_path)

def evaluate(model_name: str, dataset_path: str):
    print(f"--- Starting Evaluation ---")
    print(f"Model:   {model_name}")
    print(f"Dataset: {dataset_path}")
    
    # 1. Look up model in registry
    try:
        model_info = get_model(model_name)
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    print(f"Registry Info: {model_info}")
    
    # 2. Select appropriate runner and load model
    runner = get_runner(model_info["runner"], model_info["path"])
    runner.load()
    
    # 3. Run evaluation 
    print("\n[Evaluation] Running evaluation on dataset...")
    
    dataset_dir = Path(dataset_path)
    metadata_file = dataset_dir / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Error: Could not find {metadata_file}. Ensure the dataset has a metadata.json")
        return
        
    with open(metadata_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    predictions = []
    
    for item in dataset:
        audio_file = item.get("audio_path")
        if not audio_file or not Path(audio_file).exists():
            print(f"  Skipping {audio_file}: File not found.")
            continue
            
        print(f"  Processing {Path(audio_file).name}...")
        try:
            pred = runner.run(audio_file)
            print(f"    -> Output: {pred}")
            
            predictions.append({
                "id": item.get("id"),
                "audio_path": audio_file,
                "reference": item.get("reference_text", ""),
                "prediction": pred
            })
        except Exception as e:
            print(f"    -> Error: {e}")
    
    # Creating results dictionary
    results = {
        "model": model_name,
        "dataset": str(dataset_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "wer": 0.0  # Placeholder for Word Error Rate calculation later
        },
        "predictions": predictions
    }
    
    # 4. Save the results
    safe_model_name = model_name.replace("/", "_")
    safe_dataset_name = Path(dataset_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    result_filename = f"{safe_model_name}_{safe_dataset_name}_{timestamp}.json"
    result_filepath = RESULTS_DIR / result_filename
    
    print(f"\n[Saving] Saving evaluation results to '{result_filepath}'...")
    with open(result_filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print("--- Evaluation Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Run evaluations on multilingual audio models.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Name of the model as registered in model_registry.py"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Path or identifier for the dataset to evaluate on"
    )
    
    args = parser.parse_args()
    
    evaluate(args.model, args.dataset)

if __name__ == "__main__":
    main()
