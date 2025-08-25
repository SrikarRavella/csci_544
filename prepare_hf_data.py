#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare data from Hugging Face datasets (opus100) and save to the expected format
for LoRA training. This matches the approach in the uploaded nlp_project.py.
"""

import os
import pandas as pd
import logging
from datasets import load_dataset
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Config matching the uploaded file
config = {
    "dataset_name": "opus100",
    "language_pairs": {
        "fr": "en-fr",
        "de": "de-en",
        "it": "en-it"
    },
    "samples": {
        "train": 6000,
        "val": 1000,
        "test": 1000
    }
}

def prepare_language_data(lang_code):
    """
    Prepare data for a specific language pair from Hugging Face datasets.
    
    Args:
        lang_code (str): Language code (fr, de, it)
    
    Returns:
        bool: Success status
    """
    language_pair = config["language_pairs"].get(lang_code)
    if not language_pair:
        logger.error(f"No language pair configuration for {lang_code}")
        return False
        
    # Create output directory
    output_dir = os.path.join("data", "sampled", f"{lang_code}-en")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"Loading {language_pair} from {config['dataset_name']}...")
        dataset = load_dataset(config["dataset_name"], language_pair)
        
        # Get the train split
        train_data = dataset["train"]
        
        # Define alignment filter as in the uploaded file
        def is_aligned(example):
            # Extract language codes
            src_lang, tgt_lang = language_pair.split("-")
            
            # For fr and it, we need to swap the source and target
            if language_pair in ["en-fr", "en-it"]:
                # Swap counts (we want foreign → English ratio)
                foreign_words = len(example["translation"][tgt_lang].split())  # fr/it
                english_words = len(example["translation"][src_lang].split())  # en
            else:
                # For de-en, keep as is
                foreign_words = len(example["translation"][src_lang].split())  # de
                english_words = len(example["translation"][tgt_lang].split())  # en
            
            # Check English/foreign ratio for alignment
            ratio = english_words / max(foreign_words, 1)
            return 0.5 <= ratio <= 2.0
        
        # Filter aligned examples only
        aligned_data = train_data.filter(is_aligned)
        logger.info(f"Filtered to {len(aligned_data)} aligned examples (from {len(train_data)})")
        
        # Create a validation set (we're taking a subset of train as in your CSV approach)
        # Shuffle then take first N samples for each split to ensure no overlap
        shuffled_indices = list(range(len(aligned_data)))
        random.seed(42)
        random.shuffle(shuffled_indices)
        
        # Get required sample counts
        train_size = min(config["samples"]["train"], len(aligned_data) - config["samples"]["val"] - config["samples"]["test"])
        val_size = min(config["samples"]["val"], len(aligned_data) - train_size)
        test_size = min(config["samples"]["test"], len(aligned_data) - train_size - val_size)
        
        # Select indices for each split
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size+val_size]
        test_indices = shuffled_indices[train_size+val_size:train_size+val_size+test_size]
        
        # Select the examples
        train_examples = aligned_data.select(train_indices)
        val_examples = aligned_data.select(val_indices)
        test_examples = aligned_data.select(test_indices)
        
        logger.info(f"Split into {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test examples")
        
        # Convert to DataFrame format with 'source_with_token' and 'target' columns
        def convert_to_df(dataset, tgt_lang):
            rows = []
            for item in dataset:
                # Extract language codes
                src_lang, tgt_lang = language_pair.split("-")
                
                # For fr and it, we need to swap the source and target
                if language_pair in ["en-fr", "en-it"]:
                    # Swap source and target (we want foreign → English)
                    source_text = item["translation"][tgt_lang]  # fr/it (foreign)
                    target_text = item["translation"][src_lang]  # en (English)
                else:
                    # For de-en, keep as is (already foreign → English)
                    source_text = item["translation"][src_lang]  # de (foreign)
                    target_text = item["translation"][tgt_lang]  # en (English)
                
                # Don't add language token - MarianMT models are direction-specific
                source_with_token = source_text
                
                rows.append({
                    "source": source_text,
                    "source_with_token": source_with_token,
                    "target": target_text
                })
            return pd.DataFrame(rows)
        
        # Get target language code (fr, de, it)
        tgt_lang = language_pair.split("-")[1]
        
        # Convert to DataFrames matching your existing format
        train_df = convert_to_df(train_examples, tgt_lang)
        val_df = convert_to_df(val_examples, tgt_lang)
        test_df = convert_to_df(test_examples, tgt_lang)
        
        # Save to CSV files in the expected location
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        
        logger.info(f"Data for {lang_code} saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error preparing {lang_code} data: {e}")
        return False

def main():
    """Prepare data for all language pairs."""
    # Create output directory
    os.makedirs(os.path.join("data", "sampled"), exist_ok=True)
    
    # Process each language
    for lang in ["fr", "de", "it"]:
        logger.info(f"Preparing data for {lang}...")
        prepare_language_data(lang)
    
    logger.info("All data preparation complete!")

if __name__ == "__main__":
    main() 