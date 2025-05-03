import os
import numpy as np
import torch
import librosa
import argparse
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def load_wav(wav_path, target_sr=24000):
    """Load audio and resample to 24kHz (Moshi's sample rate)"""
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def extract_semantic_indices(model, audio, device):
    """Extract the first codec indices (semantic codec) from model.encode"""
    model = model.to(device)
    model.eval()
    
    # Convert to tensor and add batch and channel dimensions [B, C, T]
    source = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get all codebook indices [B, K, T]
        codes = model.encode(source)
        # Only keep the first layer codec [B, T] and squeeze to remove batch dimension
        features = codes[:, 0, :].squeeze(0)
        
        # Convert tensor to numpy
        features_np = features.cpu().numpy()
        
    return features_np

def extract_quantized_latent(model, audio, device):
    """Extract quantized latent features from model.encode_to_latent"""
    model = model.to(device)
    model.eval()
    
    # Convert to tensor and add batch and channel dimensions [B, C, T]
    source = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Extract quantized latent features
        features = model.encode_to_latent(source, quantize=True)
        # Squeeze to remove batch dimension
        features = features.squeeze(0)
        
        # Convert tensor to numpy
        features_np = features.cpu().numpy()
        
        # Transpose to have time dimension first
        # Assuming features_np has shape [C, T], we want [T, C]
        features_np = features_np.transpose()
        
    return features_np

def extract_semantic_representation(model, audio, device):
    """Extract the first codec representation (embeddings not indices) from model"""
    model = model.to(device)
    model.eval()
    
    # Convert to tensor and add batch and channel dimensions [B, C, T]
    source = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # First get all codebook indices
        codes = model.encode(source)
        
        # Keep only the first layer indices (semantic codebook)
        semantic_codes = codes[:, 0:1, :]
        
        # Create a codes tensor with zeros for all other codebooks
        full_codes = torch.zeros_like(codes)
        full_codes[:, 0:1, :] = semantic_codes
        
        # Use the model's decode_latent function to get the representation
        # This should work with any quantizer type
        semantic_repr = model.decode_latent(full_codes)
        
        # Squeeze to remove batch dimension and transpose 
        semantic_repr = semantic_repr.squeeze(0).transpose(0, 1)
        
        # Convert to numpy
        features_np = semantic_repr.cpu().numpy()
        
    return features_np

def load_model(model_path, device, hf_repo="kyutai/moshiko-pytorch-bf16"):
    """Load Moshi model"""
    try:
        # Try to load from moshi.models
        from moshi.models import loaders
        
        # If local model path not provided, download from HuggingFace
        if not os.path.exists(model_path):
            print(f"Downloading model from HuggingFace: {hf_repo}")
            model_path = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        
        print(f"Loading model: {model_path}")
        model = loaders.get_mimi(model_path, device)
        return model
    except ImportError:
        try:
            # Try loading with rustymimi
            import rustymimi
            print(f"Loading model with rustymimi: {model_path}")
            model = rustymimi.Tokenizer(str(model_path))
            return model
        except ImportError:
            raise ImportError("Cannot import moshi or rustymimi, please make sure you have installed the required libraries")

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device, args.hf_repo)
    
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
            
            # Load audio
            audio, sr = load_wav(wav_path, target_sr=model.sample_rate)
            
            # Extract features based on the feature_type argument
            if args.feature_type == "semantic_indices":
                features = extract_semantic_indices(model, audio, device)
                output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            elif args.feature_type == "latent":
                features = extract_quantized_latent(model, audio, device)
                output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            elif args.feature_type == "semantic_repr":
                features = extract_semantic_representation(model, audio, device)
                output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            
            # Save as npy file
            np.save(output_path, features)
            
            # If this is the first successfully processed file, print some information
            if wav_idx == 0:
                print(f"Feature shape: {features.shape}")
                print(f"Feature type: {features.dtype}")
                print(f"Feature value range: {features.min()} to {features.max()}")
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            import traceback
            print(traceback.format_exc())
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from Moshi mini codec model and save as npy files")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to Moshi model")
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16", 
                        help="If local model doesn't exist, download from this HuggingFace repository (using bf16 for best quality)")
    parser.add_argument("--flist", type=str, required=True, 
                        help="File containing list of wav file paths")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory for output npy files")
    parser.add_argument("--feature_type", type=str, 
                       choices=["semantic_indices", "latent", "semantic_repr"], 
                       default="semantic_indices",
                       help="Type of feature to extract: 'semantic_indices' for first codec indices, 'latent' for quantized latent features, or 'semantic_repr' for first codec representations")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    main(args)