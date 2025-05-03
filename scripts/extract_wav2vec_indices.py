import os
import numpy as np
import torch
import librosa
import argparse
from tqdm import tqdm
import fairseq

def load_wav(wav_path, target_sr=16000):
    """Load audio and resample to 16kHz"""
    audio, sr = librosa.load(wav_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def extract_wav2vec2_features_fairseq(model, audio, device, feature_type='indices_2d'):
    """Extract quantization features using fairseq model
    
    Args:
        model: The wav2vec2 model
        audio: Input audio
        device: Computing device
        feature_type: Type of features to extract:
                     'continuous' - continuous representations [T, C]
                     'indices_2d' - discrete indices [T, G] 
                     'indices_1d' - flattened indices [T]
    
    Returns:
        Features with time as first dimension
    """
    model = model.to(device)
    model.eval()
    
    # Convert to tensor
    source = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Use model's quantize method which properly handles feature extraction, 
        # transposition, and layer normalization
        result = model.quantize(source)
        
        # quantize calls forward_idx, which returns (continuous_features, indices)
        if isinstance(result, tuple):
            continuous_features = result[0]  # First element is quantized representations
            indices = result[1]  # Second element is indices
            
            if feature_type == 'continuous':
                # Return continuous features with shape [T, C]
                features = continuous_features.squeeze(0).cpu().numpy()
            elif feature_type == 'indices_1d':
                # Return flattened indices with shape [T]
                indices = indices.squeeze(0).cpu().numpy()
                # Convert multiple codebook indices to a single index per time step
                if indices.ndim > 1:
                    num_vars = model.quantizer.num_vars
                    # Convert indices to a single value using num_vars as the base
                    flattened_features = np.zeros(indices.shape[0], dtype=np.int64)
                    for g in range(indices.shape[1]):
                        flattened_features += indices[:, g] * (num_vars ** g)
                    features = flattened_features
                else:
                    features = indices  # Already 1D
            else:  # indices_2d
                # Return indices with shape [T, G]
                features = indices.squeeze(0).cpu().numpy()
        else:
            # In case forward_idx returns only one value
            features = result.squeeze(0).cpu().numpy()
        
    return features

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load fairseq model
    print(f"Loading model: {args.model_path}")
    
    try:
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [args.model_path]
        )
        model = models[0]
        # Ensure model has a quantizer
        if not hasattr(model, 'quantizer') or model.quantizer is None:
            raise ValueError("Model doesn't have a quantizer, cannot extract quantization features")
            
        # Ensure model is on the correct device
        model = model.to(device)
    except Exception as e:
        print(f"Failed to load fairseq model: {e}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read file list
    with open(args.flist, 'r') as f:
        wav_files = [line.strip() for line in f.readlines()]
    
    # Process each file
    for wav_idx, wav_path in enumerate(tqdm(wav_files, desc="Processing audio files")):
        try:
            # Get base filename
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            
            # Load and resample audio
            audio, sr = load_wav(wav_path)
            
            # Extract features
            features = extract_wav2vec2_features_fairseq(
                model, 
                audio, 
                device, 
                feature_type=args.feature_type
            )
            
            # Save as npy file
            np.save(output_path, features)
            
            # If this is the first successful file, print some information
            if wav_idx == 0:
                print(f"Feature type: {args.feature_type}")
                print(f"Feature shape: {features.shape}")
                print(f"Feature dtype: {features.dtype}")
                if args.feature_type != 'continuous':
                    print(f"Indices range: {features.min()} to {features.max()}")
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            import traceback
            print(traceback.format_exc())
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from wav2vec2 model and save as npy files")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to fairseq wav2vec2 model")
    parser.add_argument("--flist", type=str, required=True, 
                       help="File containing list of wav file paths")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Directory to output npy files")
    parser.add_argument("--feature_type", type=str, choices=['continuous', 'indices_1d', 'indices_2d'], 
                       default='indices_2d', 
                       help="Type of features to extract: continuous=vector representations, indices_1d=flattened indices, indices_2d=original multi-group indices")
    args = parser.parse_args()
    
    main(args)