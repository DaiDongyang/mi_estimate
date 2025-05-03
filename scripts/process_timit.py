#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
from pathlib import Path

def create_directory(directory):
    """创建目录（如果不存在）"""
    os.makedirs(directory, exist_ok=True)

def copy_and_rename_file(source_file, target_file):
    """复制并重命名文件"""
    shutil.copy2(source_file, target_file)

def get_clean_base_name(file_path):
    """获取干净的基本文件名（去除多余的扩展名）"""
    base_name = file_path.stem
    if base_name.endswith(".WAV"):
        base_name = base_name[:-4]
    return base_name

def prepare_file_lists(target_dir):
    """准备训练、验证和测试集的文件列表"""
    train_flist = open(target_dir / "train.flist", "w")
    valid_flist = open(target_dir / "valid.flist", "w")
    test_flist = open(target_dir / "test.flist", "w")
    return train_flist, valid_flist, test_flist

def close_file_lists(train_flist, valid_flist, test_flist):
    """关闭所有文件列表"""
    train_flist.close()
    valid_flist.close()
    test_flist.close()

def process_speaker_directory(speaker_dir, target_speaker_dir, file_list, target_dir, processed_files):
    """处理单个说话人目录的音频和音素文件"""
    # 创建目标文件夹
    create_directory(target_speaker_dir)
    
    # 处理WAV文件
    for wav_file in speaker_dir.glob("*.WAV*"):
        # 获取干净的基本文件名
        base_name = get_clean_base_name(wav_file)
        
        # 检查是否已处理过此文件
        if base_name in processed_files:
            continue
        
        processed_files.add(base_name)
        
        # 新的wav文件路径
        new_wav_path = target_speaker_dir / f"{base_name}.wav"
        
        # 复制并重命名WAV文件
        copy_and_rename_file(wav_file, new_wav_path)
        
        # 将路径添加到相应的列表
        rel_path = os.path.relpath(new_wav_path, target_dir)
        file_list.write(f"{rel_path}\n")
        
        # 查找并复制对应的PHN文件
        phn_file = speaker_dir / f"{base_name}.PHN"
        if phn_file.exists():
            copy_and_rename_file(phn_file, target_speaker_dir / f"{base_name}.phn")
    
    return processed_files

def process_train_set(source_dir, target_dir, train_flist):
    """处理训练集数据"""
    print("处理训练集...")
    train_dir = source_dir / "TRAIN"
    processed_files = set()  # 跟踪已处理的文件，避免重复
    
    for dialect_dir in sorted(train_dir.glob("DR*")):
        dialect_name = dialect_dir.name.lower()
        
        for speaker_dir in sorted(dialect_dir.glob("*")):
            speaker_name = speaker_dir.name.lower()
            
            # 目标说话人目录
            target_speaker_dir = target_dir / "train" / dialect_name / speaker_name
            
            # 处理该说话人的所有文件
            processed_files = process_speaker_directory(
                speaker_dir, target_speaker_dir, train_flist, target_dir, processed_files
            )
    
    return processed_files

def process_test_set(source_dir, target_dir, valid_flist, test_flist):
    """处理测试集数据，分为验证和测试两部分"""
    print("处理测试集，分为验证和测试两部分...")
    test_dir = source_dir / "TEST"
    processed_files = set()  # 重置已处理文件集合
    
    for dialect_dir in sorted(test_dir.glob("DR*")):
        dialect_name = dialect_dir.name.lower()
        dialect_num = int(dialect_name[2:])
        
        # 根据方言区域决定是验证集还是测试集
        if 1 <= dialect_num <= 4:
            subset_name = "valid"
            flist = valid_flist
        else:  # 5-8
            subset_name = "test"
            flist = test_flist
        
        for speaker_dir in sorted(dialect_dir.glob("*")):
            speaker_name = speaker_dir.name.lower()
            
            # 目标说话人目录
            target_speaker_dir = target_dir / subset_name / dialect_name / speaker_name
            
            # 处理该说话人的所有文件
            processed_files = process_speaker_directory(
                speaker_dir, target_speaker_dir, flist, target_dir, processed_files
            )
    
    return processed_files

def process_timit_dataset(source_dir, target_dir):
    """
    处理TIMIT数据集的主函数:
    1. 将.WAV文件转换为.wav并复制到新目录
    2. 复制.PHN文件到对应位置并转换为小写
    3. 生成训练、验证和测试集的文件列表
    
    Args:
        source_dir: 原始TIMIT数据集目录
        target_dir: 处理后的数据集目录
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # 创建目标目录
    create_directory(target_dir)
    
    # 准备文件列表
    train_flist, valid_flist, test_flist = prepare_file_lists(target_dir)
    
    # 处理训练集
    process_train_set(source_dir, target_dir, train_flist)
    
    # 处理测试集
    process_test_set(source_dir, target_dir, valid_flist, test_flist)
    
    # 关闭文件列表
    close_file_lists(train_flist, valid_flist, test_flist)
    
    print(f"处理完成！文件已保存至 {target_dir}")
    print(f"生成的文件列表: train.flist, valid.flist, test.flist")

def main():
    """主函数，解析命令行参数并启动处理流程"""
    parser = argparse.ArgumentParser(description="处理TIMIT数据集")
    parser.add_argument("--source", type=str, required=True, help="原始TIMIT数据集目录")
    parser.add_argument("--target", type=str, required=True, help="处理后的数据集目录")
    
    args = parser.parse_args()
    process_timit_dataset(args.source, args.target)

if __name__ == "__main__":
    main()