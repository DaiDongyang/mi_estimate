#!/usr/bin/env python3
"""
提取梅尔谱特征

从音频目录提取梅尔谱特征并保存为npy文件。
"""
import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf


def extract_mel_spectrogram(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    从音频文件提取梅尔谱
    
    参数:
        audio_path: 音频文件路径
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: 梅尔滤波器数量
        
    返回:
        梅尔谱特征
    """
    # 加载音频
    try:
        audio, sr = librosa.load(audio_path, sr=sr)
    except:
        # 尝试使用soundfile加载（处理某些特殊格式）
        audio, sr_orig = sf.read(audio_path)
        # 如果是多通道，取第一个通道
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        # 重采样到目标采样率
        if sr_orig != sr:
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr)
    
    # 计算梅尔谱
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # 转换为分贝刻度 (更符合人类听觉)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 将每个时间帧作为一个特征向量
    feature_vectors = []
    for i in range(mel_spec_db.shape[1]):
        feature_vectors.append(mel_spec_db[:, i])
    
    # 将所有时间帧组成的矩阵返回
    if len(feature_vectors) > 0:
        return np.array(feature_vectors)
    else:
        # 处理极短的音频
        return np.zeros((1, n_mels))


def process_audio_directory(input_dir, output_dir, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    处理整个音频目录，提取梅尔谱特征
    
    参数:
        input_dir: 输入音频目录
        output_dir: 输出特征目录
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: 梅尔滤波器数量
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
    for audio_path in tqdm(audio_files, desc="提取梅尔谱"):
        try:
            # 获取相对路径作为特征文件的基础名
            rel_path = os.path.relpath(audio_path, input_dir)
            # 替换扩展名为.npy
            base_name = os.path.splitext(rel_path)[0]
            # 替换路径分隔符为下划线
            base_name = base_name.replace(os.path.sep, '_')
            # 构建输出路径
            output_path = os.path.join(output_dir, f"{base_name}_mel.npy")
            
            # 提取特征
            features = extract_mel_spectrogram(audio_path, sr, n_fft, hop_length, n_mels)
            
            # 保存特征
            np.save(output_path, features)
        except Exception as e:
            print(f"处理文件 {audio_path} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='从音频目录提取梅尔谱特征')
    parser.add_argument('--input_dir', type=str, required=True, help='输入音频目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出特征目录')
    parser.add_argument('--sr', type=int, default=22050, help='采样率')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT窗口大小')
    parser.add_argument('--hop_length', type=int, default=512, help='帧移')
    parser.add_argument('--n_mels', type=int, default=128, help='梅尔滤波器数量')
    args = parser.parse_args()
    
    process_audio_directory(
        args.input_dir, 
        args.output_dir, 
        args.sr, 
        args.n_fft, 
        args.hop_length, 
        args.n_mels
    )
    print(f"梅尔谱特征提取完成，已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()