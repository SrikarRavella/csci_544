#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and preprocess the OPUS-100 dataset for multilingual translation.
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset

def main(args):
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    language_pairs = args.language_pairs

    # Directories
    raw_dir = os.path.join(data_dir, 'raw', dataset_name)
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    for lang_pair in language_pairs:
        src, tgt = lang_pair.split('-')
        config_name = f"{src}-{tgt}"
        reversed_dir = False
        # Attempt to load the specified config; if not found, try swapping
        try:
            dataset = load_dataset(dataset_name, config_name)
        except Exception:
            # Try reversed direction
            config_name = f"{tgt}-{src}"
            reversed_dir = True
            try:
                dataset = load_dataset(dataset_name, config_name)
            except Exception as e:
                print(f"Config not found for {lang_pair} or reverse: {e}. Skipping.")
                continue
        
        # Prepare directories (use original lang_pair for directory naming)
        out_dir = os.path.join(processed_dir, lang_pair)
        os.makedirs(out_dir, exist_ok=True)
        
        # Process each split
        for split in ['train', 'validation', 'test']:
            if split not in dataset:
                print(f"Split '{split}' not found for {config_name}, skipping")
                continue
            # The dataset uses nested 'translation' dict per example
            trans = dataset[split]['translation']
            df = pd.DataFrame(trans)
            # If we loaded reversed direction, swap columns
            if reversed_dir:
                # 'tgt-src' config, so translation maps {tgt: text_src, src: text_tgt}
                df = df.rename(columns={tgt: 'source', src: 'target'})
            else:
                df = df.rename(columns={src: 'source', tgt: 'target'})
            # Drop missing and duplicates
            df = df.dropna().drop_duplicates().reset_index(drop=True)
            # Add language token
            df['source_with_token'] = df['source'].apply(lambda x: f"<{src.upper()}> {x}")
            # Write to CSV
            filename = 'val.csv' if split == 'validation' else f"{split}.csv"
            df.to_csv(os.path.join(out_dir, filename), index=False)
            print(f"Saved {split} ({len(df)} examples) to {out_dir}/{filename}")

    print("\nOPUS-100 dataset download and preprocessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and preprocess OPUS-100 dataset for specified language pairs.")
    parser.add_argument('--dataset_name', type=str, default='opus100', help='Dataset name on Hugging Face')
    parser.add_argument('--language_pairs', nargs='+', default=['fr-en','de-en','it-en'], help='Language pairs to process')
    parser.add_argument('--data_dir', type=str, default='./data', help='Base data directory')
    args = parser.parse_args()
    main(args) 