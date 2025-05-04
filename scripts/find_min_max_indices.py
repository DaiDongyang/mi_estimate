import os
import numpy as np
import argparse

def find_min_max_indices(directory_path):
    """
    Traverse all .npy files in a directory and find the global minimum and maximum index values.
    
    Args:
        directory_path (str): Path to the directory containing .npy files
        
    Returns:
        dict: Dictionary containing minimum index, maximum index, and number of files processed
    """
    global_min = float('inf')  # Initialize to infinity
    global_max = float('-inf')  # Initialize to negative infinity
    files_processed = 0  # Counter for processed files
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        return f"Error: Directory '{directory_path}' does not exist."
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory_path, filename)
            
            # Load the .npy file
            try:
                indices = np.load(file_path)
                
                # Update global min and max values
                if indices.size > 0:  # Check if array is not empty
                    file_min = np.min(indices)
                    file_max = np.max(indices)
                    
                    global_min = min(global_min, file_min)
                    global_max = max(global_max, file_max)
                
                files_processed += 1
                print(f"Processed file: {filename}, current global min: {global_min}, max: {global_max}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    # Check if any files were processed
    if files_processed == 0:
        return "No .npy files found in the directory."
    
    return {
        "minimum_index": int(global_min) if global_min != float('inf') else None,
        "maximum_index": int(global_max) if global_max != float('-inf') else None,
        "files_processed": files_processed
    }

def main():
    """Main function to parse arguments and execute the search"""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Find minimum and maximum index values across all .npy files in a directory'
    )
    
    # Add arguments
    parser.add_argument(
        'directory', 
        type=str, 
        help='Path to the directory containing .npy files'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help='Enable verbose output with details of each file'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set global verbose flag
    global verbose
    verbose = args.verbose
    
    # Run the main function
    result = find_min_max_indices(args.directory)
    
    # Print results
    if isinstance(result, str):
        print(result)
    else:
        print("\nResults Summary:")
        print(f"Minimum index found: {result['minimum_index']}")
        print(f"Maximum index found: {result['maximum_index']}")
        print(f"Total files processed: {result['files_processed']}")

if __name__ == "__main__":
    main()