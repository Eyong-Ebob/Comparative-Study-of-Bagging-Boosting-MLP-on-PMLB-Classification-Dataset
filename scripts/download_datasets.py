#!/usr/bin/env python
"""
Robust PMLB dataset downloader
Handles missing/renamed datasets gracefully
"""

from pmlb import classification_dataset_names, fetch_data
import os
import pandas as pd

os.makedirs('data', exist_ok=True)

print(f"Total available classification datasets: {len(classification_dataset_names)}")
print("Starting download with error handling...\n")

successful = 0
failed = 0
failed_datasets = []

for idx, name in enumerate(classification_dataset_names, 1):
    try:
        print(f"[{idx}/{len(classification_dataset_names)}] Downloading: {name}...", end=" ", flush=True)
        df = fetch_data(name)
        
        # Save as gzip TSV
        output_path = f'data/{name}.tsv.gz'
        df.to_csv(output_path, sep='\t', index=False, compression='gzip')
        
        print(f"OK ({len(df)} rows, {len(df.columns)} cols)")
        successful += 1
        
    except Exception as e:
        print(f"FAILED: {str(e)[:50]}")
        failed += 1
        failed_datasets.append(name)

print("\n" + "="*60)
print(f"Download Summary:")
print(f"  Successful: {successful}")
print(f"  Failed:     {failed}")
print(f"  Total:      {len(classification_dataset_names)}")
print("="*60)

print(f"\nDatasets saved to: data/")
