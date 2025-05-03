import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import s3tokenizer

def load_audio_and_process(wav_path):
    """Load audio and compute mel spectrogram using s3tokenizer's functions"""
    audio = s3tokenizer.load_audio(wav_path)
    mel = s3tokenizer.log_mel_spectrogram(audio)
    return mel

def extract_indices(model, mel, device):
    """Extract the token indices from s3tokenizer model"""
    model = model.to(device)
    model.eval()
    
    # Add batch dimension and move to device
    mel = mel.unsqueeze(0).to(device)
    mel_len = torch.tensor([mel.size(2)], device=device)
    
    with torch.no_grad():
        # Get indices using the quantize method
        codes, codes_len = model.quantize(mel, mel_len)
        
        # Get only the valid indices and convert to numpy
        valid_length = codes_len[0].item()
        indices = codes[0, :valid_length].cpu().numpy()
        
    return indices

def load_model(model_name, device):
    """Load S3Tokenizer model"""
    print(f"Loading model: {model_name}")
    model = s3tokenizer.load_model(model_name)
    return model

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_name, device)
    
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
            
            # Load audio and compute mel spectrogram
            mel = load_audio_and_process(wav_path)
            
            # Extract indices
            indices = extract_indices(model, mel, device)
            
            # Save as npy file
            output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            np.save(output_path, indices)
            
            # If this is the first successfully processed file, print some information
            if wav_idx == 0:
                print(f"Feature shape: {indices.shape} (time dimension)")
                print(f"Feature type: {indices.dtype}")
                print(f"Feature value range: {indices.min()} to {indices.max()}")
                print(f"Saved as 1D array with shape [T] where T={len(indices)}")
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            import traceback
            print(traceback.format_exc())
    
    print("Processing complete!")

def process_batch(args):
    """Process multiple audio files in a single batch for better efficiency"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_name, device)
    model = model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read file list
    with open(args.flist, 'r') as f:
        wav_files = [line.strip() for line in f.readlines()]
    
    # Process files in batches
    batch_size = args.batch_size
    for i in range(0, len(wav_files), batch_size):
        batch_files = wav_files[i:i+batch_size]
        batch_mels = []
        
        # Load audio and compute mel spectrograms
        for wav_path in tqdm(batch_files, desc=f"Loading batch {i//batch_size+1}/{(len(wav_files)-1)//batch_size+1}"):
            try:
                mel = load_audio_and_process(wav_path)
                batch_mels.append(mel)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                batch_mels.append(None)
        
        # Skip empty batches
        valid_mels = [mel for mel in batch_mels if mel is not None]
        if not valid_mels:
            continue
        
        # Pad and create batch
        mels, mels_lens = s3tokenizer.padding(valid_mels)
        mels = mels.to(device)
        mels_lens = mels_lens.to(device)
        
        with torch.no_grad():
            # Get indices using the quantize method
            codes, codes_lens = model.quantize(mels, mels_lens)
            
            # Save results
            valid_idx = 0
            for j, mel in enumerate(batch_mels):
                if mel is None:
                    continue
                
                wav_path = batch_files[j]
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                
                # Get only the valid indices and convert to numpy
                valid_length = codes_lens[valid_idx].item()
                indices = codes[valid_idx, :valid_length].cpu().numpy()
                
                # Ensure shape is [T]
                if len(indices.shape) > 1:
                    indices = indices.squeeze()
                
                # Save as npy file
                output_path = os.path.join(args.output_dir, f"{base_name}.npy")
                np.save(output_path, indices)
                
                # Print info for the first file
                if i == 0 and j == 0:
                    print(f"Feature shape: {indices.shape} (time dimension)")
                    print(f"Feature type: {indices.dtype}")
                    print(f"Feature value range: {indices.min()} to {indices.max()}")
                    print(f"Saved as 1D array with shape [T] where T={len(indices)}")
                
                valid_idx += 1
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract token indices from S3Tokenizer model and save as npy files")
    parser.add_argument("--model_name", type=str, default="speech_tokenizer_v2_25hz", 
                        help="S3Tokenizer model name to use")
    parser.add_argument("--flist", type=str, required=True, 
                        help="File containing list of wav file paths")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory for output npy files")
    parser.add_argument("--batch", action="store_true", 
                        help="Process audio files in batches for better efficiency")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    if args.batch:
        process_batch(args)
    else:
        main(args)