#!/usr/bin/env python3
"""
Extract Complex Spectrogram Features

Reads audio file paths from filelist, extracts complex spectrogram features and saves them as npy files.
The real and imaginary parts of the complex spectrogram are concatenated into a real-valued vector.
"""
import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf


def extract_complex_spectrogram(audio_path, n_fft=2048, hop_length=None, win_length=None, target_sr=16000):
    """
    Extract complex spectrogram from audio file
    
    Parameters:
        audio_path: Path to audio file
        n_fft: FFT window size
        hop_length: Frame shift, if None it's automatically set to 10ms based on target_sr
        win_length: Window length, defaults to n_fft
        target_sr: Target sampling rate, default 16000Hz
        
    Returns:
        Vector with concatenated real and imaginary parts of the complex spectrogram
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
    
    # Calculate STFT
    complex_spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Extract real and imaginary parts
    real_part = complex_spec.real
    imag_part = complex_spec.imag
    
    # Concatenate real and imaginary parts into a vector
    # First process each time frame into a feature vector
    feature_vectors = []
    for i in range(complex_spec.shape[1]):
        # Extract real and imaginary parts for the current time frame
        frame_real = real_part[:, i]
        frame_imag = imag_part[:, i]
        # Concatenate real and imaginary parts
        frame_features = np.concatenate([frame_real, frame_imag])
        feature_vectors.append(frame_features)
    
    # Return feature matrix for all time frames
    if len(feature_vectors) > 0:
        return np.array(feature_vectors)
    else:
        # Handle extremely short audio
        return np.zeros((1, n_fft * 2))


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
    for wav_idx, wav_path in enumerate(tqdm(wav_files, desc="Extracting complex spectrogram")):
        try:
            # Get base filename
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            # Build output path, including sampling rate info
            # output_path = os.path.join(args.output_dir, f"{base_name}_complex_{args.target_sr//1000}k.npy")
            output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            
            # Extract features
            features = extract_complex_spectrogram(wav_path, args.n_fft, hop_length, target_sr=args.target_sr)
            
            # Save features
            np.save(output_path, features)
            
            # If this is the first successful file, print some information
            if wav_idx == 0:
                print(f"Complex spectrogram feature shape: {features.shape}")
                print(f"Complex spectrogram feature type: {features.dtype}")
                print(f"Feature dimension: {features.shape[1]} (real + imaginary)")
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            import traceback
            print(traceback.format_exc())
    
    print("Complex spectrogram feature extraction completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read audio files from filelist and extract complex spectrogram features')
    parser.add_argument("--flist", type=str, required=True, 
                        help="List file containing wav file paths")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory for output npy files")
    parser.add_argument('--n_fft', type=int, default=2048, 
                        help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=None, 
                        help='Frame shift, default is None which will automatically set to 10ms (calculated based on sampling rate)')
    parser.add_argument('--target_sr', type=int, default=16000, 
                        choices=[16000, 24000], help='Target sampling rate, supports 16000 or 24000Hz')
    args = parser.parse_args()
    
    main(args)