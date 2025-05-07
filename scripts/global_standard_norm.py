import os
import numpy as np
import argparse
import joblib
from sklearn.preprocessing import StandardScaler

def normalize_npy_files(input_dir, output_dir, stats_dir):
    """
    Use incremental computation to normalize all .npy files in a directory
    
    Parameters:
        input_dir: Input directory containing .npy files to be processed
        output_dir: Output directory for saving normalized .npy files
        stats_dir: Directory for saving mean and standard deviation statistics
    """
    # Create output directories (if they don't exist)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Get all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return
    
    # Check the shape of the first file to get the C dimension
    first_file = os.path.join(input_dir, npy_files[0])
    first_data = np.load(first_file)
    if len(first_data.shape) != 2:
        raise ValueError(f"Expected array shape [T, C], but got {first_data.shape}")
    
    C = first_data.shape[1]
    
    # Initialize statistics variables
    total_samples = 0
    sum_x = np.zeros(C)
    sum_x2 = np.zeros(C)
    
    # First pass: incremental calculation of statistics
    print("Calculating statistics (incremental method)...")
    for file_name in npy_files:
        file_path = os.path.join(input_dir, file_name)
        data = np.load(file_path)
        
        # Verify shape consistency
        if len(data.shape) != 2 or data.shape[1] != C:
            raise ValueError(f"File {file_name} expected shape [T, {C}], but got {data.shape}")
        
        # Accumulate statistics
        n_samples = data.shape[0]
        total_samples += n_samples
        sum_x += np.sum(data, axis=0)
        sum_x2 += np.sum(data**2, axis=0)
        
        # Clear data from memory
        del data
    
    # Calculate mean and standard deviation
    mean = sum_x / total_samples
    var = (sum_x2 / total_samples) - (mean**2)
    # Ensure variance is positive
    var = np.maximum(var, 1e-10)
    std = np.sqrt(var)
    
    # Save mean and standard deviation
    np.save(os.path.join(stats_dir, 'mean.npy'), mean)
    np.save(os.path.join(stats_dir, 'std.npy'), std)
    
    # Create and save StandardScaler (optional, for future use)
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = std
    scaler.var_ = var
    scaler.n_features_in_ = C
    joblib.dump(scaler, os.path.join(stats_dir, 'scaler.joblib'))
    
    print(f"Statistics calculation completed:")
    print(f"Total samples: {total_samples}")
    print(f"Mean shape: {mean.shape}")
    print(f"Standard deviation shape: {std.shape}")
    print(f"Statistics saved to {stats_dir}")
    
    # Second pass: normalize each file and save
    print("Normalizing files...")
    for file_name in npy_files:
        file_path = os.path.join(input_dir, file_name)
        data = np.load(file_path)
        
        # Use scaler to normalize data
        normalized_data = scaler.transform(data)
        
        # Save normalized data
        output_path = os.path.join(output_dir, file_name)
        np.save(output_path, normalized_data)
        
        # Clear data from memory
        del data
        del normalized_data
    
    print(f"Normalized {len(npy_files)} files. Output saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Normalize .npy files using incremental computation')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving normalized .npy files')
    parser.add_argument('--stats_dir', type=str, required=True, help='Directory for saving mean and standard deviation statistics')
    
    args = parser.parse_args()
    
    normalize_npy_files(args.input_dir, args.output_dir, args.stats_dir)

if __name__ == '__main__':
    main()