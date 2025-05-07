"""
Script to process english_us_arpa.dict and generate phn2id.json

This script extracts all unique phonemes from an ARPA dictionary file
and creates a JSON mapping from phonemes to sequential IDs starting from 1.

Usage:
    python generate_phn2id.py --input english_us_arpa.dict --output phn2id.json --start-id 1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate phn2id.json from english_us_arpa.dict"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="english_us_arpa.dict",
        help="Path to the input ARPA dictionary file (default: english_us_arpa.dict)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="phn2id.json",
        help="Path to the output JSON file (default: phn2id.json)"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting ID for phoneme mapping (default: 1)"
    )
    return parser.parse_args()


def read_dict_file(file_path: str) -> List[str]:
    """Read the dictionary file and return non-empty lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)


def extract_phonemes(lines: List[str]) -> Set[str]:
    """Extract all unique phonemes from the dictionary lines."""
    phonemes = set()
    
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 5:
            phn = parts[-1].strip()
            # Split multi-phoneme entries (e.g., "AH0 Z" -> ["AH0", "Z"])
            for p in phn.split():
                if p.strip():
                    phonemes.add(p.strip())
    
    return phonemes


def create_phoneme_mapping(phonemes: Set[str], start_id: int) -> Dict[str, int]:
    """Create a mapping from phonemes to sequential IDs."""
    return {phn: id for id, phn in enumerate(sorted(phonemes), start_id)}


def save_json_file(data: Dict[str, int], file_path: str) -> None:
    """Save the phoneme mapping to a JSON file."""
    try:
        # Create directory if it doesn't exist
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error writing to file '{file_path}': {e}")
        sys.exit(1)


def main() -> None:
    """Main function to process the dictionary and generate the mapping."""
    args = parse_arguments()
    
    print(f"Reading dictionary file: {args.input}")
    lines = read_dict_file(args.input)
    
    print("Extracting phonemes...")
    phonemes = extract_phonemes(lines)
    
    print("Creating phoneme to ID mapping...")
    phn2id = create_phoneme_mapping(phonemes, args.start_id)
    
    print(f"Writing mapping to: {args.output}")
    save_json_file(phn2id, args.output)
    
    print(f"Successfully generated {args.output} with {len(phn2id)} unique phonemes.")
    print(f"Phoneme IDs start from: {args.start_id}")


if __name__ == "__main__":
    main()