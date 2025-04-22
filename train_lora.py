#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train LoRA adapters for MarianMT model.
"""

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import sys
import logging

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lora_adapter import LoraAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    """Dataset for translation task."""
    
    def __init__(self, dataframe, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            dataframe (pandas.DataFrame): DataFrame with 'source' and 'target' columns.
            tokenizer: Tokenizer for the model.
            max_length (int): Maximum sequence length.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get source and target texts
        source_text = self.dataframe.iloc[idx]['source_with_token']
        target_text = self.dataframe.iloc[idx]['target']
        
        # Tokenize source and target
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input IDs and attention mask
        input_ids = source_encoding['input_ids'].squeeze()
        attention_mask = source_encoding['attention_mask'].squeeze()
        
        # Get labels (target IDs)
        labels = target_encoding['input_ids'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def train_lora_adapter(args):
    """Train a LoRA adapter for the specified language."""
    # Check language
    if args.lang not in ['fr', 'it']:
        raise ValueError("Language must be 'fr' or 'it'")
    
    # Determine language pair
    lang_pair = f"{args.lang}-en"
    
    # Load data
    data_dir = os.path.join(args.data_dir, 'processed', lang_pair)
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    
    logger.info(f"Loaded {len(train_df)} training examples and {len(val_df)} validation examples.")
    
    # Initialize LoRA adapter
    lora_adapter = LoraAdapter(
        base_model_name=args.base_model,
        target_language=args.lang,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    model = lora_adapter.get_model()
    tokenizer = lora_adapter.get_tokenizer()
    device = lora_adapter.device
    
    # Create datasets and dataloaders
    train_dataset = TranslationDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = TranslationDataset(val_df, tokenizer, max_length=args.max_length)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_dataloader, device)
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir = os.path.join(args.output_dir, f"lora-adapter-{args.lang}-en")
            lora_adapter.save_adapter(output_dir)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    logger.info("Training completed.")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for MarianMT model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with the processed data")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the trained adapter")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="Helsinki-NLP/opus-mt-de-en", help="Base model to adapt")
    parser.add_argument("--lang", type=str, required=True, choices=['fr', 'it'], help="Language to adapt to")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of the LoRA adapter matrices")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Scaling factor for the LoRA adapter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for the LoRA adapter")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train LoRA adapter
    train_lora_adapter(args)


if __name__ == "__main__":
    main() 