"""
Performance profiling script to identify training bottlenecks
"""
import time
import torch
import yaml
import numpy as np
from collections import defaultdict
import psutil
import GPUtil

def profile_data_loading(config_path, num_batches=10):
    """Profile data loading performance"""
    from dataset import create_data_loaders
    
    print("Creating data loaders...")
    data_info = create_data_loaders(config_path)
    train_loader = data_info['train_loader']
    
    # Timing statistics
    timing_stats = defaultdict(list)
    
    print(f"\nProfiling {num_batches} batches...")
    print("-" * 60)
    
    for i, (x, y) in enumerate(train_loader):
        if i >= num_batches:
            break
            
        batch_start = time.time()
        
        # Move to GPU and measure time
        gpu_start = time.time()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            torch.cuda.synchronize()
        gpu_time = time.time() - gpu_start
        
        batch_time = time.time() - batch_start
        
        timing_stats['batch_total'].append(batch_time)
        timing_stats['gpu_transfer'].append(gpu_time)
        timing_stats['data_loading'].append(batch_time - gpu_time)
        
        # Print batch info
        print(f"Batch {i+1}/{num_batches}:")
        print(f"  Shape: x={x.shape}, y={y.shape}")
        print(f"  Total time: {batch_time*1000:.1f}ms")
        print(f"  Data loading: {(batch_time-gpu_time)*1000:.1f}ms")
        print(f"  GPU transfer: {gpu_time*1000:.1f}ms")
        
        # Memory usage
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
            print(f"  GPU memory: {gpu_mem:.2f}GB")
    
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("-" * 60)
    
    for key, values in timing_stats.items():
        avg = np.mean(values) * 1000  # Convert to ms
        std = np.std(values) * 1000
        print(f"{key}:")
        print(f"  Average: {avg:.1f}ms ± {std:.1f}ms")
        print(f"  Min: {min(values)*1000:.1f}ms, Max: {max(values)*1000:.1f}ms")
    
    return timing_stats

def profile_model_forward(config_path, num_batches=10):
    """Profile model forward pass"""
    from mine_estimator import MINE
    from dataset import create_data_loaders
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loader
    data_info = create_data_loaders(config_path)
    train_loader = data_info['train_loader']
    
    # Initialize model
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    data_config = config.get('data', {})
    
    mine_model = MINE(
        x_type=model_config.get('x_type', 'float'),
        y_type=model_config.get('y_type', 'float'),
        x_dim=data_info['x_dim'],
        y_dim=data_info['y_dim'],
        x_vocab_size=model_config.get('x_vocab_size'),
        x_embedding_dim=model_config.get('x_embedding_dim'),
        y_vocab_size=model_config.get('y_vocab_size'),
        y_embedding_dim=model_config.get('y_embedding_dim'),
        x_proj_dim=model_config.get('x_proj_dim'),
        y_proj_dim=model_config.get('y_proj_dim'),
        context_frame_numbers=data_config.get('context_frame_numbers', 1),
        hidden_dims=model_config.get('hidden_dims', [128, 64]),
        lr=train_config.get('lr', 1e-4),
        device=train_config.get('device', 'cuda')
    )
    
    print(f"\nProfiling model forward pass ({num_batches} batches)...")
    print("-" * 60)
    
    timing_stats = defaultdict(list)
    
    for i, (x, y) in enumerate(train_loader):
        if i >= num_batches:
            break
        
        # Forward pass timing
        forward_start = time.time()
        with torch.no_grad():
            if mine_model.x_type == 'index':
                x = x.long().to(mine_model.device)
            else:
                x = x.float().to(mine_model.device)
                
            if mine_model.y_type == 'index':
                y = y.long().to(mine_model.device)
            else:
                y = y.float().to(mine_model.device)
            
            # Time MI computation
            mi_start = time.time()
            mi_estimate, _, _, _ = mine_model.compute_mutual_info(x, y)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            mi_time = time.time() - mi_start
            
        forward_time = time.time() - forward_start
        
        timing_stats['forward_total'].append(forward_time)
        timing_stats['mi_computation'].append(mi_time)
        
        print(f"Batch {i+1}/{num_batches}:")
        print(f"  Forward pass: {forward_time*1000:.1f}ms")
        print(f"  MI computation: {mi_time*1000:.1f}ms")
        print(f"  MI estimate: {mi_estimate.item():.4f}")
    
    print("\n" + "="*60)
    print("Model Forward Pass Summary:")
    print("-" * 60)
    
    for key, values in timing_stats.items():
        avg = np.mean(values) * 1000
        std = np.std(values) * 1000
        print(f"{key}:")
        print(f"  Average: {avg:.1f}ms ± {std:.1f}ms")

def check_system_resources():
    """Check system resources"""
    print("\nSystem Resources:")
    print("-" * 60)
    
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU: {cpu_percent}% ({cpu_count} cores)")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.percent}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    
    # GPU info
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
            print(f"  Utilization: {gpu.load*100:.1f}%")
    except:
        print("GPU: Unable to get GPU info (GPUtil not available)")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.percent}% used ({disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB)")

def analyze_data_distribution(config_path, num_samples=100):
    """Analyze data distribution and potential issues"""
    from dataset import MIDataset
    import pandas as pd
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    # Create dataset
    dataset = MIDataset(
        csv_path=data_config['train_csv'],
        features_dir=data_config['features_dir'],
        segment_length=data_config.get('segment_length', 100),
        normalization=data_config.get('normalization', 'standard'),
        x_type=model_config.get('x_type', 'float'),
        y_type=model_config.get('y_type', 'float'),
        x_repeat=data_config.get('x_repeat', 1),
        y_repeat=data_config.get('y_repeat', 1),
        context_frame_numbers=data_config.get('context_frame_numbers', 1)
    )
    
    print(f"\nAnalyzing {num_samples} samples...")
    print("-" * 60)
    
    load_times = []
    x_shapes = []
    y_shapes = []
    
    for i in range(min(num_samples, len(dataset))):
        start_time = time.time()
        x, y = dataset[i]
        load_time = time.time() - start_time
        
        load_times.append(load_time)
        x_shapes.append(x.shape)
        y_shapes.append(y.shape)
        
        if load_time > 0.1:  # Flag slow samples
            print(f"Sample {i}: Slow load time {load_time*1000:.1f}ms")
    
    print(f"\nLoad time statistics:")
    print(f"  Average: {np.mean(load_times)*1000:.1f}ms")
    print(f"  Std: {np.std(load_times)*1000:.1f}ms")
    print(f"  Max: {np.max(load_times)*1000:.1f}ms")
    
    # Check for shape consistency
    unique_x_shapes = set(x_shapes)
    unique_y_shapes = set(y_shapes)
    print(f"\nShape analysis:")
    print(f"  Unique X shapes: {unique_x_shapes}")
    print(f"  Unique Y shapes: {unique_y_shapes}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile MINE training performance')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num-batches', type=int, default=10, help='Number of batches to profile')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'data', 'model', 'system', 'distribution'],
                       help='What to profile')
    
    args = parser.parse_args()
    
    print("MINE Training Performance Profiler")
    print("=" * 60)
    
    if args.mode in ['all', 'system']:
        check_system_resources()
    
    if args.mode in ['all', 'data']:
        profile_data_loading(args.config, args.num_batches)
    
    if args.mode in ['all', 'model']:
        profile_model_forward(args.config, args.num_batches)
    
    if args.mode in ['all', 'distribution']:
        analyze_data_distribution(args.config)