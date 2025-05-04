#!/usr/bin/env python
"""
Preprocess dataset CSV to filter out samples with excessive length differences.
"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def check_sample_validity(x_path, y_path, x_repeat, y_repeat, features_dir=None):
    """
    Check if a sample is valid based on length difference after repeat.
    
    Args:
        x_path: Path to x feature file
        y_path: Path to y feature file
        x_repeat: Repeat factor for x
        y_repeat: Repeat factor for y
        features_dir: Optional base directory for feature files
        
    Returns:
        valid: Boolean indicating if sample is valid
        x_len: Length of x feature
        y_len: Length of y feature
        length_diff: Length difference after repeat
    """
    # Resolve paths if needed
    if features_dir:
        if not os.path.isabs(x_path):
            x_path = os.path.join(features_dir, x_path)
        if not os.path.isabs(y_path):
            y_path = os.path.join(features_dir, y_path)
    
    try:
        # Load features
        x_feature = np.load(x_path)
        y_feature = np.load(y_path)
        
        # Get original lengths
        x_len_original = x_feature.shape[0]
        y_len_original = y_feature.shape[0]
        
        # Calculate lengths after repeat
        x_len_after_repeat = x_len_original * x_repeat
        y_len_after_repeat = y_len_original * y_repeat
        
        # Calculate length difference
        length_diff = abs(x_len_after_repeat - y_len_after_repeat)
        
        # Check if difference exceeds maximum repeat factor
        max_repeat = max(x_repeat, y_repeat)
        is_valid = length_diff <= max_repeat
        
        return is_valid, x_len_original, y_len_original, length_diff
        
    except Exception as e:
        print(f"Error processing {x_path} or {y_path}: {e}")
        return False, 0, 0, float('inf')

def filter_dataset(csv_path, output_csv_path, x_repeat, y_repeat, features_dir=None):
    """
    Filter dataset CSV to remove invalid samples.
    
    Args:
        csv_path: Input CSV file path
        output_csv_path: Output CSV file path
        x_repeat: Repeat factor for x
        y_repeat: Repeat factor for y
        features_dir: Optional base directory for feature files
    """
    # Load CSV
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return
    
    # Check for required columns
    required_cols = ['id', 'x_feature', 'y_feature']
    if not all(col in data.columns for col in required_cols):
        # Check for old column names and rename if needed
        if 'source_feature' in data.columns and 'target_feature' in data.columns:
            data = data.rename(columns={'source_feature': 'x_feature', 'target_feature': 'y_feature'})
            print("Renamed columns 'source_feature' and 'target_feature' to 'x_feature' and 'y_feature'")
        else:
            print(f"CSV must contain columns {required_cols}")
            return
    
    # Initialize lists for valid samples and stats
    valid_samples = []
    skipped_samples = []
    stats = {
        'total': len(data),
        'valid': 0,
        'skipped': 0,
        'max_diff': 0,
        'avg_diff': 0,
    }
    
    # Process each sample
    print(f"Processing {len(data)} samples...")
    total_diff = 0
    
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        is_valid, x_len, y_len, length_diff = check_sample_validity(
            row['x_feature'], row['y_feature'], x_repeat, y_repeat, features_dir
        )
        
        if is_valid:
            valid_samples.append(row)
            stats['valid'] += 1
            total_diff += length_diff
            stats['max_diff'] = max(stats['max_diff'], length_diff)
        else:
            skipped_samples.append({
                'id': row['id'],
                'x_feature': row['x_feature'],
                'y_feature': row['y_feature'],
                'x_len': x_len,
                'y_len': y_len,
                'length_diff': length_diff
            })
            stats['skipped'] += 1
    
    # Calculate average difference for valid samples
    if stats['valid'] > 0:
        stats['avg_diff'] = total_diff / stats['valid']
    
    # Create new dataframe with valid samples
    valid_df = pd.DataFrame(valid_samples)
    
    # Save filtered dataset
    valid_df.to_csv(output_csv_path, index=False)
    
    # Save skipped samples for inspection
    skipped_df = pd.DataFrame(skipped_samples)
    skipped_path = output_csv_path.replace('.csv', '_skipped.csv')
    skipped_df.to_csv(skipped_path, index=False)
    
    # Print stats
    print(f"Total samples: {stats['total']}")
    print(f"Valid samples: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"Skipped samples: {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    print(f"Maximum length difference: {stats['max_diff']}")
    print(f"Average length difference: {stats['avg_diff']:.2f}")
    print(f"Filtered dataset saved to: {output_csv_path}")
    print(f"Skipped samples saved to: {skipped_path}")

def main():
    parser = argparse.ArgumentParser(description='Filter dataset based on length difference after repeat')
    parser.add_argument('csv_path', help='Input CSV file path')
    parser.add_argument('output_csv_path', help='Output CSV file path')
    parser.add_argument('--x-repeat', type=int, default=1, help='Repeat factor for x features')
    parser.add_argument('--y-repeat', type=int, default=1, help='Repeat factor for y features')
    parser.add_argument('--features-dir', help='Base directory for feature files')
    
    args = parser.parse_args()
    
    filter_dataset(
        args.csv_path, 
        args.output_csv_path, 
        args.x_repeat, 
        args.y_repeat, 
        args.features_dir
    )

if __name__ == '__main__':
    main()