#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and preprocess OPUS Europarl dataset for
German-English, French-English, and Italian-English language pairs.
"""

import os
import urllib.request
import zipfile
import argparse
import random
from tqdm import tqdm
import pandas as pd

# Dataset URLs - updated with current working URLs
OPUS_URLS = {
    "de-en": "https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/de-en.txt.zip",
    "fr-en": "https://object.pouta.csc.fi/OPUS-Europarl/v8/raw/fr-en.txt.zip",
    "it-en": "https://object.pouta.csc.fi/OPUS-Europarl/v8/raw/it-en.txt.zip"
}

# Output directories
RAW_DIR = "raw"
PROCESSED_DIR = "processed"
SPLITS = ["train", "val", "test"]

# Dataset sizes
TRAIN_SIZE = 100000  # 100k sentence pairs for training
VAL_SIZE = 5000      # 5k sentence pairs for validation
TEST_SIZE = 5000     # 5k sentence pairs for testing


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file from a URL with a progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path, extract_dir):
    """Extract a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def preprocess_data(lang_pair, data_dir):
    """Preprocess the data for a language pair."""
    raw_dir = os.path.join(data_dir, RAW_DIR, lang_pair)
    src_lang, tgt_lang = lang_pair.split('-')
    
    # Read source and target files
    src_path = os.path.join(raw_dir, f"Europarl.{lang_pair}.{src_lang}")
    tgt_path = os.path.join(raw_dir, f"Europarl.{lang_pair}.{tgt_lang}")
    
    with open(src_path, 'r', encoding='utf-8') as src_file:
        src_lines = src_file.readlines()
    
    with open(tgt_path, 'r', encoding='utf-8') as tgt_file:
        tgt_lines = tgt_file.readlines()
    
    # Create dataframe and clean data
    data = pd.DataFrame({
        'source': [line.strip() for line in src_lines],
        'target': [line.strip() for line in tgt_lines]
    })
    
    # Remove empty lines and duplicates
    data = data[(data['source'].str.strip() != '') & (data['target'].str.strip() != '')]
    data = data.drop_duplicates().reset_index(drop=True)
    
    # Add language token to source sentences
    lang_token = f"<{src_lang.upper()}>"
    data['source_with_token'] = data['source'].apply(lambda x: f"{lang_token} {x}")
    
    # Split data into train, val, test
    total_size = min(len(data), TRAIN_SIZE + VAL_SIZE + TEST_SIZE)
    indices = list(range(len(data)))
    random.shuffle(indices)
    selected_indices = indices[:total_size]
    
    train_indices = selected_indices[:TRAIN_SIZE]
    val_indices = selected_indices[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
    test_indices = selected_indices[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE]
    
    splits = {
        'train': data.iloc[train_indices].reset_index(drop=True),
        'val': data.iloc[val_indices].reset_index(drop=True),
        'test': data.iloc[test_indices].reset_index(drop=True)
    }
    
    # Save processed data
    processed_dir = os.path.join(data_dir, PROCESSED_DIR, lang_pair)
    os.makedirs(processed_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        split_data.to_csv(os.path.join(processed_dir, f"{split_name}.csv"), index=False)
    
    print(f"Processed {lang_pair} dataset:")
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} examples")


def main(args):
    """Main function to download and preprocess the OPUS Europarl dataset."""
    data_dir = args.data_dir
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, RAW_DIR), exist_ok=True)
    os.makedirs(os.path.join(data_dir, PROCESSED_DIR), exist_ok=True)
    
    # Download and extract datasets
    for lang_pair, url in OPUS_URLS.items():
        print(f"\nProcessing {lang_pair} dataset:")
        
        # Create language pair directory
        lang_pair_dir = os.path.join(data_dir, RAW_DIR, lang_pair)
        os.makedirs(lang_pair_dir, exist_ok=True)
        
        # Download dataset
        zip_path = os.path.join(lang_pair_dir, f"{lang_pair}.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading {lang_pair} dataset...")
            download_url(url, zip_path)
        
        # Extract dataset
        print(f"Extracting {lang_pair} dataset...")
        extract_zip(zip_path, lang_pair_dir)
        
        # Preprocess dataset
        print(f"Preprocessing {lang_pair} dataset...")
        preprocess_data(lang_pair, data_dir)
    
    print("\nAll datasets downloaded and preprocessed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess OPUS Europarl dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save the datasets")
    args = parser.parse_args()
    
    main(args) 