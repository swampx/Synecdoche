#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile


def check_mergecap():
    """Check if mergecap tool is available"""
    try:
        subprocess.run(['mergecap', '-h'],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def merge_pcaps_batch_with_mergecap(pcap_files, output_file, batch_size=50):
    """Use mergecap tool to merge pcap files in batches to avoid long command line arguments"""
    if not pcap_files:
        print(f"Warning: No pcap files found")
        return False

    if len(pcap_files) == 1:
        # Only one file, copy directly
        shutil.copy2(pcap_files[0], output_file)
        print(f"Copying single file: {pcap_files[0]} -> {output_file}")
        return True

    print(f"Starting to merge {len(pcap_files)} files in batches, batch size: {batch_size}")

    try:
        # If the number of files is not large, merge directly
        if len(pcap_files) <= batch_size:
            return merge_single_batch(pcap_files, output_file)

        # Merge in batches
        temp_files = []
        temp_dir = tempfile.mkdtemp(prefix='pcap_merge_')

        try:
            # Phase 1: Merge files in batches into temporary files
            for i in range(0, len(pcap_files), batch_size):
                batch = pcap_files[i:i + batch_size]
                batch_num = i // batch_size + 1
                temp_file = Path(temp_dir) / f"batch_{batch_num:04d}.pcap"

                print(f"  Merging batch {batch_num}/{(len(pcap_files) + batch_size - 1) // batch_size}: {len(batch)} files")

                if not merge_single_batch(batch, temp_file):
                    return False

                temp_files.append(temp_file)

            # Phase 2: Merge all temporary files
            print(f"  Final merge of {len(temp_files)} batch files")

            # If temporary files are still too many, recursively merge in batches
            if len(temp_files) > batch_size:
                return merge_pcaps_batch_with_mergecap(temp_files, output_file, batch_size)
            else:
                return merge_single_batch(temp_files, output_file)

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            if Path(temp_dir).exists():
                Path(temp_dir).rmdir()

    except Exception as e:
        print(f"Error during batch merge: {e}")
        return False


def merge_single_batch(pcap_files, output_file):
    """Merge a single batch of pcap files"""
    try:
        cmd = ['mergecap', '-w', str(output_file)] + [str(f) for f in pcap_files]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"mergecap error: {e}")
        print(f"Error output: {e.stderr}")
        return False


def merge_pcaps_simple_concat(pcap_files, output_file):
    """Simple binary concatenation method (not recommended, may cause format issues)"""
    if not pcap_files:
        print(f"Warning: No pcap files found")
        return False

    try:
        with open(output_file, 'wb') as outfile:
            for i, pcap_file in enumerate(pcap_files):
                with open(pcap_file, 'rb') as infile:
                    if i == 0:
                        # Copy the first file completely
                        outfile.write(infile.read())
                    else:
                        # Skip pcap file header for other files (usually 24 bytes)
                        infile.seek(24)  # Skip pcap global header
                        outfile.write(infile.read())

        print(f"Simple merge of {len(pcap_files)} pcap files to: {output_file}")
        print("Note: Using simple concatenation method, there may be format issues. Recommend installing wireshark to use mergecap")
        return True
    except Exception as e:
        print(f"Error during simple merge: {e}")
        return False


def merge_pcaps_simple_concat_batch(pcap_files, output_file, batch_size=100):
    """Merge in batches using simple concatenation"""
    if not pcap_files:
        print(f"Warning: No pcap files found")
        return False

    if len(pcap_files) <= batch_size:
        return merge_pcaps_simple_concat(pcap_files, output_file)

    print(f"Merging {len(pcap_files)} files in batches using simple concatenation, batch size: {batch_size}")

    try:
        temp_files = []
        temp_dir = tempfile.mkdtemp(prefix='pcap_simple_merge_')

        try:
            # Phase 1: Merge files in batches into temporary files
            for i in range(0, len(pcap_files), batch_size):
                batch = pcap_files[i:i + batch_size]
                batch_num = i // batch_size + 1
                temp_file = Path(temp_dir) / f"batch_{pcap_files[0].name}_{batch_num:04d}.pcap"

                print(f"  Merging batch {batch_num}/{(len(pcap_files) + batch_size - 1) // batch_size}: {len(batch)} files")

                if not merge_pcaps_simple_concat(batch, temp_file):
                    return False

                temp_files.append(temp_file)

            # Phase 2: Final merge
            print(f"  Final merge of {len(temp_files)} batch files")

            if len(temp_files) > batch_size:
                return merge_pcaps_simple_concat_batch(temp_files, output_file, batch_size)
            else:
                return merge_pcaps_simple_concat(temp_files, output_file)

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            if Path(temp_dir).exists():
                Path(temp_dir).rmdir()

    except Exception as e:
        print(f"Error during batch simple merge: {e}")
        return False


def get_optimal_batch_size():
    """Calculate optimal batch size based on system parameters"""
    try:
        # Get system ARG_MAX limit
        import os
        arg_max = os.sysconf('SC_ARG_MAX')
        # Conservative estimate, assuming average 200 characters per file path, reserve some space
        estimated_batch_size = min(50, max(10, arg_max // (200 * 8)))  # Divide by 8 for more conservative estimate
        return estimated_batch_size
    except:
        return 50  # Default value


def merge_pcaps_in_directory(input_dir, output_dir):
    """Merge all pcap files in the specified directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory does not exist - {input_dir}")
        return False

    dataset_name = input_path.name
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if mergecap tool is available
    use_mergecap = check_mergecap()
    if not use_mergecap:
        print("Warning: mergecap tool not found, will use simple concatenation method")
        print("It is recommended to install Wireshark to get mergecap for better merge results")

    # Get optimal batch size
    batch_size = get_optimal_batch_size()
    print(f"Using batch size: {batch_size}")

    success_count = 0
    total_categories = 0

    # Iterate through all subdirectories in the input directory
    for category_dir in input_path.iterdir():
        if not category_dir.is_dir():
            continue

        total_categories += 1
        category_name = category_dir.name
        print(f"\nProcessing category: {category_name}")

        # Find all pcap files in this category
        pcap_patterns = ['*.pcap', '*.pcapng', '*.cap']
        pcap_files = []

        for pattern in pcap_patterns:
            pcap_files.extend(list(category_dir.rglob(pattern)))

        if not pcap_files:
            print(f"  Warning: No pcap files found in {category_name}")
            continue

        # Sort by file name
        pcap_files.sort()
        print(f"  Found {len(pcap_files)} pcap files:")

        # Only display first few and last few files to avoid too much output
        display_files = pcap_files[:3] + (pcap_files[-3:] if len(pcap_files) > 6 else pcap_files[3:])
        if len(pcap_files) > 6:
            for pcap_file in pcap_files[:3]:
                file_size = pcap_file.stat().st_size / (1024 * 1024)  # MB
                print(f"    - {pcap_file.name} ({file_size:.2f} MB)")
            print(f"    ... ({len(pcap_files) - 6} files) ...")
            for pcap_file in pcap_files[-3:]:
                file_size = pcap_file.stat().st_size / (1024 * 1024)  # MB
                print(f"    - {pcap_file.name} ({file_size:.2f} MB)")
        else:
            for pcap_file in pcap_files:
                file_size = pcap_file.stat().st_size / (1024 * 1024)  # MB
                print(f"    - {pcap_file.name} ({file_size:.2f} MB)")

        # Create output subdirectory
        output_category_dir = output_path / category_name
        output_category_dir.mkdir(parents=True, exist_ok=True)

        # Output file name
        output_file = output_category_dir / f"{dataset_name}_{category_name}_merged.pcap"

        # Merge pcap files
        if use_mergecap:
            success = merge_pcaps_batch_with_mergecap(pcap_files, output_file, batch_size)
        else:
            success = merge_pcaps_simple_concat_batch(pcap_files, output_file, batch_size * 2)

        if success:
            success_count += 1
            # Display output file information
            if output_file.exists():
                output_size = output_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  Output file: {output_file} ({output_size:.2f} MB)")
        else:
            print(f"  Error: Failed to merge {category_name}")

    print(f"\nMerge completed!")
    print(f"Total {total_categories} categories processed, {success_count} successful")
    return success_count == total_categories


def main():
    """Main function"""

    input_directory = "dataset_pcap/cstnet1.3"
    output_directory = "dataset_pcap/cstnet1.3_merged"

    print(f"input: {input_directory}")
    print(f"output: {output_directory}")

    success = merge_pcaps_in_directory(input_directory, output_directory)

    if success:
        print("\nAll categories processed successfully!")
    else:
        print("\nSome categories failed to process, please check error messages")
        sys.exit(1)


if __name__ == "__main__":
    main()