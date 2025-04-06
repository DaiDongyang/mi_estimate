#!/usr/bin/env python3
"""
提取复数谱特征

从音频目录提取复数谱特征并保存为npy文件。
复数谱的实部和虚部会拼接成一个实数向量。
"""
import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf


def extract_complex_spectrogram(audio_path, n_fft=2048, hop_length=512, win_length=None):
    """
    从音频文件提取复数谱
    
    参数:
        audio_path: 音频文件路径
        n_fft: FFT窗口大小
        hop_length: 帧移
        win_length: 窗口长度，默认等于n_fft
        
    返回:
        复数谱的实部和虚部拼接后的向量
    """
    # 加载音频
    try:
        audio, sr = librosa.load(audio_path, sr=None)
    except:
        # 尝试使用soundfile加载（处理某些特殊格式）
        audio, sr = sf.read(audio_path)
        # 如果是多通道，取第一个通道
        if len(audio.shape) > 1:
            audio = audio[:, 0]
    
    # 计算STFT
    complex_spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # 提取实部和虚部
    real_part = complex_spec.real
    imag_part = complex_spec.imag
    
    # 将实部和虚部拼接成一个向量
    # 先将每个时间帧处理成一个特征向量
    feature_vectors = []
    for i in range(complex_spec.shape[1]):
        # 提取当前时间帧的实部和虚部
        frame_real = real_part[:, i]
        frame_imag = imag_part[:, i]
        # 拼接实部和虚部
        frame_features = np.concatenate([frame_real, frame_imag])
        feature_vectors.append(frame_features)
    
    # 将所有时间帧的特征向量平均，得到一个表示整个音频的特征向量
    # 根据需要，也可以返回所有时间帧的特征矩阵
    if len(feature_vectors) > 0:
        return np.array(feature_vectors)
    else:
        # 处理极短的音频
        return np.zeros((1, n_fft * 2))


def process_audio_directory(input_dir, output_dir, n_fft=2048, hop_length=512):
    """
    处理整个音频目录，提取复数谱特征
    
    参数:
        input_dir: 输入音频目录
        output_dir: 输出特征目录
        n_fft: FFT窗口大小
        hop_length: 帧移
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 处理每个音频文件
    for audio_path in tqdm(audio_files, desc="提取复数谱"):
        try:
            # 获取相对路径作为特征文件的基础名
            rel_path = os.path.relpath(audio_path, input_dir)
            # 替换扩展名为.npy
            base_name = os.path.splitext(rel_path)[0]
            # 替换路径分隔符为下划线
            base_name = base_name.replace(os.path.sep, '_')
            # 构建输出路径
            output_path = os.path.join(output_dir, f"{base_name}_complex.npy")
            
            # 提取特征
            features = extract_complex_spectrogram(audio_path, n_fft, hop_length)
            
            # 保存特征
            np.save(output_path, features)
        except Exception as e:
            print(f"处理文件 {audio_path} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='从音频目录提取复数谱特征')
    parser.add_argument('--input_dir', type=str, required=True, help='输入音频目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出特征目录')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT窗口大小')
    parser.add_argument('--hop_length', type=int, default=512, help='帧移')
    args = parser.parse_args()
    
    process_audio_directory(args.input_dir, args.output_dir, args.n_fft, args.hop_length)
    print(f"复数谱特征提取完成，已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()