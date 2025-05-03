#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频文件时长统计脚本
使用方法: python audio_duration.py [音频目录路径] [--recursive] [--formats wav,mp3,...]
"""

import os
import argparse
import subprocess
import json
from pathlib import Path
from datetime import timedelta
import re

def get_audio_duration_ffprobe(file_path):
    """使用ffprobe获取音频文件时长"""
    try:
        # 使用ffprobe获取音频文件信息
        result = subprocess.run(
            [
                'ffprobe', 
                '-v', 'quiet', 
                '-print_format', 'json', 
                '-show_format', 
                '-show_streams', 
                file_path
            ],
            capture_output=True,
            text=True
        )
        
        # 解析JSON输出
        data = json.loads(result.stdout)
        
        # 从格式信息中获取时长
        if 'format' in data and 'duration' in data['format']:
            return float(data['format']['duration'])
        
        # 如果格式信息中没有时长，尝试从流信息中获取
        if 'streams' in data:
            for stream in data['streams']:
                if 'duration' in stream:
                    return float(stream['duration'])
        
        return 0
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return 0

def get_audio_duration_soxi(file_path):
    """使用SoX (soxi) 获取音频文件时长"""
    try:
        # 使用soxi获取音频时长信息
        result = subprocess.run(
            ['soxi', '-D', file_path],
            capture_output=True,
            text=True
        )
        
        # 解析输出的秒数
        if result.stdout.strip():
            return float(result.stdout.strip())
        return 0
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return 0

def format_time(seconds):
    """将秒数格式化为时:分:秒.毫秒格式"""
    td = timedelta(seconds=seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    hours += days * 24
    milliseconds = round(td.microseconds / 1000)
    
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def count_audio_files_and_duration(directory, recursive=False, formats=None, use_soxi=True):
    """
    统计指定目录下的音频文件数量和总时长
    
    Args:
        directory: 要统计的目录
        recursive: 是否递归处理子目录
        formats: 要处理的音频格式列表，例如['wav', 'mp3']
        use_soxi: 是否使用soxi工具，否则使用ffprobe
    
    Returns:
        tuple: (文件数量, 总时长(秒), 文件列表)
    """
    if formats is None:
        formats = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma']
    
    # 转换格式为小写并移除点号
    formats = [fmt.lower().lstrip('.') for fmt in formats]
    
    # 决定使用哪个工具获取时长
    duration_func = get_audio_duration_soxi if use_soxi else get_audio_duration_ffprobe
    
    directory = Path(directory)
    total_files = 0
    total_duration = 0
    file_list = []
    
    # 获取文件列表的生成器
    if recursive:
        file_generator = directory.rglob('*')
    else:
        file_generator = directory.glob('*')
    
    # 遍历所有文件
    for file_path in file_generator:
        if file_path.is_file() and file_path.suffix.lstrip('.').lower() in formats:
            # 获取文件时长
            duration = duration_func(str(file_path))
            
            if duration > 0:
                total_files += 1
                total_duration += duration
                file_info = {
                    'path': str(file_path),
                    'duration': duration,
                    'duration_formatted': format_time(duration)
                }
                file_list.append(file_info)
    
    return total_files, total_duration, file_list

def check_tools():
    """检查必要的工具是否安装"""
    soxi_available = False
    ffprobe_available = False
    
    try:
        subprocess.run(['soxi', '-h'], capture_output=True)
        soxi_available = True
    except FileNotFoundError:
        pass
    
    try:
        subprocess.run(['ffprobe', '-h'], capture_output=True)
        ffprobe_available = True
    except FileNotFoundError:
        pass
    
    return soxi_available, ffprobe_available

def main():
    """主函数，解析命令行参数并执行统计"""
    parser = argparse.ArgumentParser(description='统计音频文件数量和总时长')
    parser.add_argument('directory', help='要扫描的目录路径')
    parser.add_argument('--recursive', '-r', action='store_true', help='是否递归处理子目录')
    parser.add_argument('--formats', '-f', default='wav,mp3', 
                        help='要处理的音频格式，用逗号分隔（默认: wav,mp3）')
    parser.add_argument('--use-ffprobe', action='store_true', 
                        help='强制使用ffprobe而不是soxi')
    parser.add_argument('--output', '-o', help='输出结果到文件')
    
    args = parser.parse_args()
    
    # 检查工具可用性
    soxi_available, ffprobe_available = check_tools()
    
    if not soxi_available and not ffprobe_available:
        print("错误: 需要安装 SoX 或 FFmpeg 才能运行此脚本")
        print("  安装 SoX: sudo apt-get install sox")
        print("  安装 FFmpeg: sudo apt-get install ffmpeg")
        return
    
    # 确定使用哪个工具
    use_soxi = soxi_available and not args.use_ffprobe
    tool_name = "soxi" if use_soxi else "ffprobe"
    print(f"使用 {tool_name} 工具获取音频时长")
    
    # 处理音频格式列表
    formats = [fmt.strip() for fmt in args.formats.split(',')]
    
    # 获取统计结果
    print(f"正在扫描目录: {args.directory}")
    print(f"处理格式: {', '.join(formats)}")
    print(f"递归处理子目录: {'是' if args.recursive else '否'}")
    
    file_count, total_duration, file_list = count_audio_files_and_duration(
        args.directory, args.recursive, formats, use_soxi
    )
    
    # 打印结果
    print("\n统计结果:")
    print(f"音频文件总数: {file_count}")
    print(f"总时长: {format_time(total_duration)} (约 {total_duration:.2f} 秒)")
    
    # 如果指定了输出文件，将详细结果写入文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"音频文件总数: {file_count}\n")
            f.write(f"总时长: {format_time(total_duration)} (约 {total_duration:.2f} 秒)\n\n")
            f.write("文件列表:\n")
            
            # 按时长降序排序文件
            sorted_files = sorted(file_list, key=lambda x: x['duration'], reverse=True)
            
            for i, file_info in enumerate(sorted_files, 1):
                f.write(f"{i}. {file_info['path']}\n")
                f.write(f"   时长: {file_info['duration_formatted']} ({file_info['duration']:.2f} 秒)\n")
        
        print(f"\n详细结果已保存到: {args.output}")

if __name__ == "__main__":
    main()