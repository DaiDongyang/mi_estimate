#!/usr/bin/env python3
"""
Extract Mel Spectrogram Features

Reads audio file paths from filelist, extracts mel spectrogram features and saves them as npy files.
"""
import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf


def extract_mel_spectrogram(audio_path, target_sr=16000, n_fft=2048, hop_length=None, n_mels=128):
    """
    Extract mel spectrogram from audio file
    
    Parameters:
        audio_path: Path to audio file
        target_sr: Target sampling rate
        n_fft: FFT window size
        hop_length: Frame shift, if None it's automatically set to 10ms based on target_sr
        n_mels: Number of mel filters
        
    Returns:
        Mel spectrogram features
    """
    # Load audio and resample to target sampling rate
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)
    except:
        # Try using soundfile for loading (handles some special formats)
        audio, sr = sf.read(audio_path)
        # If multi-channel, take the first channel
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        # Resample to target sampling rate
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # If hop_length is not specified, set it to 10ms
    if hop_length is None:
        hop_length = int(target_sr * 0.01)  # 10ms
    
    # Calculate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=target_sr, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to decibel scale (better for human perception)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Use each time frame as a feature vector
    feature_vectors = []
    for i in range(mel_spec_db.shape[1]):
        feature_vectors.append(mel_spec_db[:, i])
    
    # Return matrix composed of all time frames
    if len(feature_vectors) > 0:
        return np.array(feature_vectors)
    else:
        # Handle extremely short audio
        return np.zeros((1, n_mels))


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read file list
    with open(args.flist, 'r') as f:
        wav_files = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(wav_files)} audio files")
    print(f"Target sampling rate: {args.target_sr}Hz")
    
    # If hop_length is not specified, calculate 10ms frame shift based on sampling rate
    if args.hop_length is None:
        hop_length = int(args.target_sr * 0.01)  # 10ms
    else:
        hop_length = args.hop_length
    
    print(f"Using parameters: target_sr={args.target_sr}Hz, hop_length={hop_length} samples ({hop_length/args.target_sr*1000:.1f}ms)")
    
    # Process each file
    for wav_idx, wav_path in enumerate(tqdm(wav_files, desc="Extracting mel spectrogram")):
        try:
            # Get base filename
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            # Build output path, including sampling rate info
            output_path = os.path.join(args.output_dir, f"{base_name}_mel_{args.target_sr//1000}k.npy")
            
            # Extract features
            features = extract_mel_spectrogram(
                wav_path, 
                target_sr=args.target_sr, 
                n_fft=args.n_fft, 
                hop_length=hop_length, 
                n_mels=args.n_mels
            )
            
            # Save features
            np.save(output_path, features)
            
            # If this is the first successful file, print some information
            if wav_idx == 0:
                print(f"Mel spectrogram feature shape: {features.shape}")
                print(f"Mel spectrogram feature type: {features.dtype}")
                print(f"Number of mel filters: {args.n_mels}")
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            import traceback
            print(traceback.format_exc())
    
    print("Mel spectrogram feature extraction completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read audio files from filelist and extract mel spectrogram features')
    parser.add_argument("--flist", type=str, required=True, 
                        help="List file containing wav file paths")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory for output npy files")
    parser.add_argument('--target_sr', type=int, default=16000, 
                        choices=[16000, 24000], help='Target sampling rate, supports 16000 or 24000Hz')
    parser.add_argument('--n_fft', type=int, default=2048, 
                        help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=None, 
                        help='Frame shift, default is None which will automatically set to 10ms (calculated based on sampling rate)')
    parser.add_argument('--n_mels', type=int, default=128, 
                        help='Number of mel filters')
    args = parser.parse_args()
    
    main(args)