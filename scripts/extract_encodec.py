import os
import numpy as np
import torch
import librosa
import argparse
from tqdm import tqdm

from encodec import EncodecModel
from encodec.utils import convert_audio


def load_wav(wav_path, target_sr=24000, channels=1):
    """Load audio and resample to target sample rate"""
    audio, sr = librosa.load(wav_path, sr=None, mono=channels==1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Ensure audio has correct number of channels
    if channels == 1 and audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    elif channels == 2 and audio.ndim == 1:
        audio = np.stack([audio, audio])
    
    return audio, target_sr


def extract_first_codebook_indices(model, audio, device):
    """Extract the first codec indices (semantic codec) from model.encode"""
    model = model.to(device)
    model.eval()
    
    # Convert to tensor and add batch dimension
    if audio.ndim == 1:
        # Mono audio
        source = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        # Stereo audio
        source = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get encoded frames
        encoded_frames = model.encode(source)
        
        # Extract first codebook indices
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        
        # Only keep the first layer indices [B, T] and squeeze to remove batch dimension
        features = codes[:, 0, :].squeeze(0)
        
        # Convert tensor to numpy
        features_np = features.cpu().numpy()
        
    return features_np


def extract_first_codebook_repr(model, audio, device):
    """Extract representation of only the first codebook"""
    model = model.to(device)
    model.eval()
    
    # Convert to tensor and add batch dimension
    if audio.ndim == 1:
        # Mono audio
        source = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        # Stereo audio
        source = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get encoded frames
        encoded_frames = model.encode(source)
        
        # Extract first codebook indices
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        
        # Keep only the first layer indices and create a codes tensor with zeros for all other codebooks
        first_codes = codes.clone()
        first_codes[:, 1:, :] = 0
        
        # Use model's quantizer to decode only the first codebook representation
        # Get the codebook weights from the first quantizer
        codebook_weights = model.quantizer.vq.layers[0].codebook
        
        # Use the indices to lookup embeddings from codebook
        indices = first_codes[:, 0, :]
        embeddings = torch.nn.functional.embedding(indices, codebook_weights)  # [B, T, D]
        
        # Prepare the shape for decoder (need to transpose)
        embeddings = embeddings.transpose(1, 2)  # [B, D, T]
        
        # Decode to get the first codebook representation
        semantic_repr = model.decoder(embeddings)
        
        # Squeeze to remove batch dimension
        semantic_repr = semantic_repr.squeeze(0)
        
        # Convert to numpy
        features_np = semantic_repr.cpu().numpy()
        
    return features_np


def extract_all_codebook_repr(model, audio, device):
    """Extract representation using all codebooks"""
    model = model.to(device)
    model.eval()
    
    # Convert to tensor and add batch dimension
    if audio.ndim == 1:
        # Mono audio
        source = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        # Stereo audio
        source = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        # First get encoder output (latent features before quantization)
        emb = model.encoder(source)
        
        # Decode the latent representation (without quantization)
        representation = model.decoder(emb)
        
        # Squeeze to remove batch dimension
        representation = representation.squeeze(0)
        
        # Convert to numpy
        features_np = representation.cpu().numpy()
        
    return features_np


def load_model(model_type="encodec_24khz", device=None):
    """Load EnCodec model"""
    try:
        if model_type == "encodec_24khz":
            model = EncodecModel.encodec_model_24khz(pretrained=True)
        elif model_type == "encodec_48khz":
            model = EncodecModel.encodec_model_48khz(pretrained=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if device:
            model = model.to(device)
        return model
    except ImportError:
        raise ImportError("Cannot import encodec, please make sure you have installed the required libraries")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_type, device)
    
    # Set bandwidth (determines number of codebooks used)
    bandwidth = args.bandwidth
    model.set_target_bandwidth(bandwidth)
    print(f"Using bandwidth: {bandwidth} kbps")
    
    # Get sample rate and channels based on model type
    if args.model_type == "encodec_24khz":
        sample_rate = 24000
        channels = 1
    else:  # encodec_48khz
        sample_rate = 48000
        channels = 2
    
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
            audio, sr = load_wav(wav_path, target_sr=sample_rate, channels=channels)
            
            # Extract features based on the feature_type argument
            if args.feature_type == "first_codebook_indices":
                features = extract_first_codebook_indices(model, audio, device)
                output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            elif args.feature_type == "first_codebook_repr":
                features = extract_first_codebook_repr(model, audio, device)
                output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            elif args.feature_type == "all_codebook_repr":
                features = extract_all_codebook_repr(model, audio, device)
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
    parser = argparse.ArgumentParser(description="Extract features from EnCodec model and save as npy files")
    parser.add_argument("--model_type", type=str, choices=["encodec_24khz", "encodec_48khz"], default="encodec_24khz", 
                        help="Type of EnCodec model to use")
    parser.add_argument("--bandwidth", type=float, choices=[1.5, 3.0, 6.0, 12.0, 24.0], default=6.0, 
                        help="Target bandwidth (kbps) - determines number of codebooks used")
    parser.add_argument("--flist", type=str, required=True, 
                        help="File containing list of wav file paths")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory for output npy files")
    parser.add_argument("--feature_type", type=str, 
                       choices=["first_codebook_indices", "first_codebook_repr", "all_codebook_repr"], 
                       default="first_codebook_indices",
                       help="Type of feature to extract: 'first_codebook_indices' for first codec indices, 'first_codebook_repr' for first codec representations, or 'all_codebook_repr' for all codebook representations")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    main(args)