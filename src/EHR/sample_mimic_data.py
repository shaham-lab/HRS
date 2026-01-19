"""
MIMIC-IV Data Sampling Script

This script creates small sample datasets from the MIMIC-IV clinical database.
It reads CSV files (both .csv and .csv.gz formats) from a source directory,
samples the first 2000 rows from each file, and saves them as uncompressed CSV files.

Features:
- Handles approximately 31 different CSV files with various schemas
- Automatically detects and handles both .csv and .csv.gz files
- Uses nrows parameter to avoid loading large files into memory
- Includes error handling for empty files and parsing issues
- Reports progress to console

Usage:
    python sample_mimic_data.py
"""

import pandas as pd
from pathlib import Path
import os

# Configuration
SOURCE_DIR = "./data/mimic4/raw"  # Directory containing MIMIC-IV CSV files
OUTPUT_DIR = "./data/mimic4/samples"  # Directory for sampled output files
SAMPLE_SIZE = 2000  # Number of rows to sample from each file


def ensure_output_directory():
    """Create the output directory if it doesn't exist."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_path.absolute()}")


def get_csv_files(source_dir):
    """
    Find all CSV files (both .csv and .csv.gz) in the source directory.
    
    Args:
        source_dir: Path to the source directory
        
    Returns:
        List of Path objects for CSV files
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"✗ Error: Source directory '{source_path.absolute()}' does not exist.")
        return []
    
    # Find both .csv and .csv.gz files
    csv_files = list(source_path.glob("*.csv")) + list(source_path.glob("*.csv.gz"))
    
    print(f"✓ Found {len(csv_files)} CSV file(s) in {source_path.absolute()}")
    return csv_files


def get_output_filename(input_path):
    """
    Generate output filename from input path.
    Removes .gz extension if present and keeps .csv extension.
    
    Args:
        input_path: Path object for input file
        
    Returns:
        Output filename as string
    """
    filename = input_path.name
    
    # Remove .gz extension if present
    if filename.endswith('.gz'):
        filename = filename[:-3]  # Remove '.gz'
    
    return filename


def sample_csv_file(input_path, output_dir, sample_size=2000):
    """
    Read a CSV file and save a sample of the first N rows.
    
    Args:
        input_path: Path to input CSV file (can be .csv or .csv.gz)
        output_dir: Path to output directory
        sample_size: Number of rows to sample (default: 2000)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n📄 Processing: {input_path.name}")
        
        # Read only the first N rows to avoid memory issues
        # pandas automatically handles .gz compression
        # low_memory=False ensures consistent dtype inference across chunks
        df = pd.read_csv(input_path, nrows=sample_size, low_memory=False)
        
        print(f"   Loaded {len(df)} rows × {len(df.columns)} columns")
        
        # Generate output filename (without .gz extension)
        output_filename = get_output_filename(input_path)
        output_path = Path(output_dir) / output_filename
        
        # Save as uncompressed CSV
        df.to_csv(output_path, index=False)
        
        print(f"   ✓ Saved to: {output_filename}")
        
        return True
        
    except pd.errors.EmptyDataError:
        print(f"   ✗ Error: File is empty - {input_path.name}")
        return False
        
    except pd.errors.ParserError as e:
        print(f"   ✗ Error parsing file: {e}")
        return False
        
    except Exception as e:
        print(f"   ✗ Unexpected error: {type(e).__name__}: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 60)
    print("MIMIC-IV Data Sampling Script")
    print("=" * 60)
    
    # Step 1: Create output directory
    ensure_output_directory()
    
    # Step 2: Get list of CSV files
    csv_files = get_csv_files(SOURCE_DIR)
    
    if not csv_files:
        print("\n⚠ No CSV files found. Please check the SOURCE_DIR path.")
        return
    
    # Step 3: Process each file
    print(f"\n📊 Starting to process {len(csv_files)} file(s)...\n")
    
    success_count = 0
    failure_count = 0
    
    for csv_file in csv_files:
        if sample_csv_file(csv_file, OUTPUT_DIR, SAMPLE_SIZE):
            success_count += 1
        else:
            failure_count += 1
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"✓ Successfully processed: {success_count} file(s)")
    if failure_count > 0:
        print(f"✗ Failed to process: {failure_count} file(s)")
    print(f"📂 Output saved to: {Path(OUTPUT_DIR).absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
