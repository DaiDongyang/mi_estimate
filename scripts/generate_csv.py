#!/usr/bin/env python
"""
Simple script to create a CSV file listing matching .npy files from two directories.
Usage: python create_csv.py --x_dir /path/to/x_features --y_dir /path/to/y_features --output_csv output.csv
"""
import os
import csv
import argparse

def create_csv(x_dir, y_dir, output_csv):
    """
    Create a CSV file listing matching .npy files from two directories.
    
    Args:
        x_dir: Directory containing X feature .npy files
        y_dir: Directory containing Y feature .npy files
        output_csv: Path to output CSV file
    """
    # Ensure directories exist
    if not os.path.isdir(x_dir):
        raise ValueError(f"X directory does not exist: {x_dir}")
    if not os.path.isdir(y_dir):
        raise ValueError(f"Y directory does not exist: {y_dir}")
    
    # Get lists of .npy files
    x_files = {f for f in os.listdir(x_dir) if f.endswith('.npy')}
    y_files = {f for f in os.listdir(y_dir) if f.endswith('.npy')}
    
    # Find matching files
    matching_files = x_files.intersection(y_files)
    
    if not matching_files:
        print("Warning: No matching .npy files found in both directories.")
        return 0
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    # Write CSV file
    count = 0
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['id', 'x_feature', 'y_feature'])
        
        # Write data rows
        for filename in sorted(matching_files):
            x_path = os.path.join(x_dir, filename)
            y_path = os.path.join(y_dir, filename)
            
            # Get ID (filename without extension)
            file_id = os.path.splitext(filename)[0]
            
            # Write row
            writer.writerow([file_id, x_path, y_path])
            count += 1
    
    print(f"Created {output_csv} with {count} entries.")
    return count

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create CSV file for dataset with matching .npy files.')
    parser.add_argument('--x_dir', required=True, help='Directory containing X feature .npy files')
    parser.add_argument('--y_dir', required=True, help='Directory containing Y feature .npy files')
    parser.add_argument('--output_csv', required=True, help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Create the dataset CSV
    create_csv(args.x_dir, args.y_dir, args.output_csv)

if __name__ == "__main__":
    main()
