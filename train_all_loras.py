#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train all 3 LORA adapters with specified parameters:
- 25k training samples each
- LORA rank 32
- Batch size 32
- 5 epochs
- 2k validation and 2k test samples each
"""

import os
import subprocess
import pandas as pd
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def train_lora_adapter(language, base_model="Helsinki-NLP/opus-mt-LANG-en"):
    """
    Train a LORA adapter for the specified language.
    
    Args:
        language (str): Language code (e.g., 'fr', 'de', 'it')
        base_model (str): Base model to adapt, fixed to appropriate direction
    """
    # Use the correct model for the translation direction
    # We're translating foreign languages to English
    actual_base_model = f"Helsinki-NLP/opus-mt-{language}-en"
    
    # Print clear info about the direction
    logger.info(f"Using {actual_base_model} to translate {language.upper()} → EN")
    
    # Prepare command
    cmd = [
        "/opt/homebrew/bin/python3.9", "src/training/train_lora.py",
        "--data_dir", "./data/sampled",
        "--output_dir", "./outputs/lora_adapters",
        "--base_model", actual_base_model,
        "--lang", language,
        "--lora_rank", "32",
        "--lora_dropout", "0.05",
        "--batch_size", "16",
        "--learning_rate", "0.002",
        "--warmup_ratio", "0.1",
        "--num_epochs", "10"
    ]
    
    # Execute command
    logger.info(f"Starting training for {language}-en with command: {' '.join(cmd)}")
    process = subprocess.run(cmd, check=True)
    
    if process.returncode == 0:
        logger.info(f"Successfully trained LORA adapter for {language}-en")
        return True
    else:
        logger.error(f"Failed to train LORA adapter for {language}-en")
        return False

def main():
    # Create output directory
    os.makedirs(os.path.join('outputs', 'lora_adapters'), exist_ok=True)
    
    # Languages to train
    languages = ['fr', 'de', 'it']
    
    # Train LORA adapters for each language
    for lang in languages:
        logger.info(f"Training LORA adapter for {lang}-en...")
        train_lora_adapter(lang)

if __name__ == "__main__":
    main() 