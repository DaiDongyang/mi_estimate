#!/usr/bin/env python3
"""
生成数据集CSV文件

基于梅尔谱和复数谱特征目录生成train.csv、valid.csv和test.csv文件。
"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def get_matching_features(mel_dir, complex_dir):
    """
    获取两个目录中匹配的特征文件
    
    参数:
        mel_dir: 梅尔谱特征目录
        complex_dir: 复数谱特征目录
        
    返回:
        匹配的特征文件列表，每项包含(id, mel_file, complex_file)
    """
    # 获取所有梅尔谱特征文件
    mel_files = {}
    for file in os.listdir(mel_dir):
        if file.endswith('.npy'):
            # 提取基础名（不含_mel后缀和.npy扩展名）
            base_name = file.replace('_mel.npy', '')
            mel_files[base_name] = os.path.join(mel_dir, file)
    
    # 获取所有复数谱特征文件
    complex_files = {}
    for file in os.listdir(complex_dir):
        if file.endswith('.npy'):
            # 提取基础名（不含_complex后缀和.npy扩展名）
            base_name = file.replace('_complex.npy', '')
            complex_files[base_name] = os.path.join(complex_dir, file)
    
    # 找到两个目录中匹配的文件
    matching_features = []
    for base_name in mel_files:
        if base_name in complex_files:
            matching_features.append((
                base_name,                # 使用基础名作为ID
                mel_files[base_name],     # 梅尔谱文件路径
                complex_files[base_name]  # 复数谱文件路径
            ))
    
    print(f"找到 {len(matching_features)} 对匹配的特征文件")
    return matching_features


def check_feature_validity(feature_pairs, max_check=10):
    """
    检查特征文件的有效性和尺寸
    
    参数:
        feature_pairs: 特征文件对列表
        max_check: 最大检查数量
    """
    num_to_check = min(max_check, len(feature_pairs))
    print(f"随机检查 {num_to_check} 对特征文件的有效性...")
    
    indices = random.sample(range(len(feature_pairs)), num_to_check)
    
    for idx in indices:
        id_name, mel_path, complex_path = feature_pairs[idx]
        try:
            # 加载特征文件
            mel_feature = np.load(mel_path)
            complex_feature = np.load(complex_path)
            
            # 打印形状信息
            print(f"ID: {id_name}")
            print(f"  梅尔谱形状: {mel_feature.shape}")
            print(f"  复数谱形状: {complex_feature.shape}")
        except Exception as e:
            print(f"加载特征文件失败 (ID={id_name}): {e}")


def generate_csv_files(feature_pairs, output_dir, train_ratio=0.8, valid_ratio=0.1):
    """
    生成训练、验证和测试CSV文件
    
    参数:
        feature_pairs: 特征文件对列表
        output_dir: 输出CSV目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机打乱特征对
    random.shuffle(feature_pairs)
    
    # 计算分割点
    n_samples = len(feature_pairs)
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)
    
    # 分割数据集
    train_pairs = feature_pairs[:n_train]
    valid_pairs = feature_pairs[n_train:n_train+n_valid]
    test_pairs = feature_pairs[n_train+n_valid:]
    
    print(f"数据集划分: 训练={len(train_pairs)}, 验证={len(valid_pairs)}, 测试={len(test_pairs)}")
    
    # 创建训练集CSV
    train_df = pd.DataFrame(
        train_pairs, 
        columns=['id', 'source_feature', 'target_feature']
    )
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    
    # 创建验证集CSV
    valid_df = pd.DataFrame(
        valid_pairs, 
        columns=['id', 'source_feature', 'target_feature']
    )
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    
    # 创建测试集CSV
    test_df = pd.DataFrame(
        test_pairs, 
        columns=['id', 'source_feature', 'target_feature']
    )
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"CSV文件已生成到目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='生成数据集CSV文件')
    parser.add_argument('--mel_dir', type=str, required=True, help='梅尔谱特征目录')
    parser.add_argument('--complex_dir', type=str, required=True, help='复数谱特征目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出CSV目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--reverse', action='store_true', help='反转源和目标（使复数谱为源，梅尔谱为目标）')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取匹配的特征文件
    feature_pairs = get_matching_features(args.mel_dir, args.complex_dir)
    
    # 检查特征文件有效性
    check_feature_validity(feature_pairs)
    
    # 如果需要反转源和目标
    if args.reverse:
        print("反转源和目标特征...")
        feature_pairs = [(id_name, complex_path, mel_path) 
                         for id_name, mel_path, complex_path in feature_pairs]
    
    # 生成CSV文件
    generate_csv_files(
        feature_pairs, 
        args.output_dir, 
        args.train_ratio, 
        args.valid_ratio
    )


if __name__ == "__main__":
    main()