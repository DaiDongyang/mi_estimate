"""
Convert TextGrid files to phoneme ID features with 10ms frame shift and save as npy format.

For text = "", ID = 0
For phonemes not in phn2id.json, ID = max(phn2id.json IDs) + 1

Uses the textgrid library to process TextGrid files.

Usage:
    python textgrid_to_features.py --flist path/to/flist.dev --output_dir path/to/output --phn2id path/to/phn2id.json
"""

import argparse
import json
import os
import numpy as np
import textgrid
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert TextGrid files to phoneme ID features with configurable frame shift and save as npy format."
    )
    parser.add_argument(
        "--flist",
        type=str,
        required=True,
        help="Path to the TextGrid file list"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output npy files"
    )
    parser.add_argument(
        "--phn2id",
        type=str,
        required=True,
        help="Path to phn2id.json file"
    )
    parser.add_argument(
        "--frame_shift",
        type=float,
        default=0.01,
        help="Frame shift in seconds (default: 0.01 = 10ms)"
    )
    return parser.parse_args()


def load_phn2id(phn2id_path):
    """Load phn2id.json file"""
    with open(phn2id_path, 'r', encoding='utf-8') as f:
        phn2id = json.load(f)
    
    # Calculate maximum ID value for unknown phonemes
    max_id = max(phn2id.values())
    unknown_id = max_id + 1
    
    logger.info(f"Loaded {len(phn2id)} phonemes, max ID: {max_id}, unknown phoneme ID: {unknown_id}")
    return phn2id, unknown_id


def create_feature_array(tg_file, phn2id, unknown_id, frame_shift=0.01):
    """Create feature array with 10ms frame shift using textgrid library"""
    # Get total duration
    total_duration = tg_file.maxTime
    
    # Calculate number of frames (ceiling)
    num_frames = int(np.ceil(total_duration / frame_shift))
    
    # Create feature array, default 0 (represents silence)
    features = np.zeros(num_frames, dtype=np.int32)
    
    # Get phones tier
    try:
        # Try to find by name
        phones_tier = tg_file.getFirst('phones')
    except Exception:
        # If by name fails, try to get the second tier (index 1)
        try:
            phones_tier = tg_file.tiers[1]
            logger.info(f"Could not find 'phones' tier by name, using second tier (name: {phones_tier.name})")
        except (IndexError, AttributeError):
            raise ValueError("Cannot find phones tier")
    
    # Fill the feature array
    for interval in phones_tier:
        if not interval.mark:  # Empty text remains 0
            continue
        
        # Calculate start and end frames
        # Use a half-open interval approach [start, end) to avoid overlap
        start_frame = int(np.floor(interval.minTime / frame_shift))
        end_frame = int(np.floor(interval.maxTime / frame_shift))
        
        # Add 1 to end_frame to include the last frame (unless it's exactly on the boundary)
        if interval.maxTime % frame_shift > 1e-10:  # Small epsilon to handle floating point errors
            end_frame += 1
        
        # Ensure we don't exceed boundaries
        end_frame = min(end_frame, num_frames)
        
        # Get phoneme ID, if not in phn2id use unknown_id
        phoneme = interval.mark
        phoneme_id = phn2id.get(phoneme, unknown_id)
        
        # Set corresponding frame values
        features[start_frame:end_frame] = phoneme_id
    
    return features


def process_textgrid_files(flist_path, output_dir, phn2id_path, frame_shift):
    """Process all TextGrid files in the flist"""
    # Load phn2id
    phn2id, unknown_id = load_phn2id(phn2id_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read file list
    with open(flist_path, 'r', encoding='utf-8') as f:
        textgrid_paths = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Read {len(textgrid_paths)} TextGrid file paths")
    logger.info(f"Using frame shift of {frame_shift} seconds ({frame_shift*1000} ms)")
    
    # Process each TextGrid file
    for idx, textgrid_path in enumerate(textgrid_paths):
        try:
            # Generate output file path
            output_filename = os.path.splitext(os.path.basename(textgrid_path))[0] + '.npy'
            output_path = os.path.join(output_dir, output_filename)
            
            # Load TextGrid file
            tg_file = textgrid.TextGrid.fromFile(textgrid_path)
            
            # Create feature array
            features = create_feature_array(tg_file, phn2id, unknown_id, frame_shift)
            
            # Save as npy file
            np.save(output_path, features)
            
            if (idx + 1) % 100 == 0 or idx == len(textgrid_paths) - 1:
                logger.info(f"Processed {idx + 1}/{len(textgrid_paths)} files")
                
        except Exception as e:
            logger.error(f"Error processing file {textgrid_path}: {str(e)}")
    
    logger.info("All files processed")


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info(f"Starting TextGrid processing")
    logger.info(f"File list: {args.flist}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"phn2id file: {args.phn2id}")
    logger.info(f"Frame shift: {args.frame_shift} seconds ({args.frame_shift*1000} ms)")
    
    # Process TextGrid files
    process_textgrid_files(args.flist, args.output_dir, args.phn2id, args.frame_shift)


if __name__ == "__main__":
    main()