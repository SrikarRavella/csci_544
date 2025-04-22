#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train a student model via knowledge distillation from multiple teacher models
for multilingual translation (German, French, Italian to English).
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import sys
import logging
from collections import defaultdict

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.distillation import MultilingualTranslationDistiller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MultilingualTranslationDataset(Dataset):
    """Dataset for multilingual translation task."""
    
    def __init__(self, dataframes, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            dataframes (dict): Dictionary mapping language codes to DataFrames.
            tokenizer: Tokenizer for the model.
            max_length (int): Maximum sequence length.
        """
        self.dataframes = dataframes
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Combine all dataframes
        self.combined_data = []
        for lang, df in dataframes.items():
            # Add language information
            lang_data = [(row.source_with_token, row.target, lang) for _, row in df.iterrows()]
            self.combined_data.extend(lang_data)
        
        # Shuffle data
        np.random.shuffle(self.combined_data)
    
    def __len__(self):
        return len(self.combined_data)
    
    def __getitem__(self, idx):
        # Get source, target, and language
        source_text, target_text, lang_code = self.combined_data[idx]
        
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
            'labels': labels,
            'lang_code': lang_code
        }


def train_epoch(distiller, dataloader, optimizer, scheduler, device):
    """Train the student model for one epoch."""
    student_model = distiller.get_student_model()
    student_model.train()
    
    total_loss = 0
    total_distill_loss = 0
    total_ce_loss = 0
    
    # Track losses by language
    lang_losses = defaultdict(float)
    lang_counts = defaultdict(int)
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Extract language codes
        lang_codes = batch.pop('lang_code')
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Store the original labels for loss computation
        labels = batch['labels'].clone()
        
        # Dictionary to store losses by language
        batch_losses = {}
        
        # Forward pass through student model
        student_outputs = distiller.forward_student(batch)
        student_logits = student_outputs.logits
        
        # Aggregate loss across all languages in the batch
        batch_loss = 0
        batch_count = 0
        
        # Process each language in the batch
        for lang in set(lang_codes):
            # Find indices for this language
            lang_indices = torch.tensor([i for i, code in enumerate(lang_codes) if code == lang]).to(device)
            
            if len(lang_indices) == 0:
                continue
            
            # Extract data for this language
            lang_student_logits = student_logits[lang_indices]
            lang_labels = labels[lang_indices]
            
            # Create a subset of the batch for the teacher
            lang_batch = {k: v[lang_indices] for k, v in batch.items()}
            
            # Forward pass through teacher model for this language
            teacher_outputs = distiller.forward_teacher(lang_batch, lang)
            teacher_logits = teacher_outputs.logits
            
            # Compute distillation loss
            loss, distill_loss, ce_loss = distiller.knowledge_distillation_loss(
                lang_student_logits, teacher_logits, lang_labels
            )
            
            # Weight loss by the number of samples
            weighted_loss = loss * len(lang_indices)
            batch_loss += weighted_loss
            batch_count += len(lang_indices)
            
            # Track language-specific losses
            lang_losses[lang] += loss.item() * len(lang_indices)
            lang_counts[lang] += len(lang_indices)
            
            # Store for logging
            batch_losses[lang] = loss.item()
        
        # Normalize batch loss
        if batch_count > 0:
            batch_loss = batch_loss / batch_count
            
            # Backward pass
            batch_loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update tracking
            total_loss += batch_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': batch_loss.item() if batch_count > 0 else 0,
            **{f"{lang}_loss": val for lang, val in batch_losses.items()}
        })
    
    # Compute average losses by language
    avg_lang_losses = {lang: losses / lang_counts[lang] for lang, losses in lang_losses.items()}
    
    return total_loss / len(dataloader), avg_lang_losses


def validate(distiller, dataloader, device):
    """Validate the student model."""
    student_model = distiller.get_student_model()
    student_model.eval()
    
    total_loss = 0
    
    # Track losses by language
    lang_losses = defaultdict(float)
    lang_counts = defaultdict(int)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            # Extract language codes
            lang_codes = batch.pop('lang_code')
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Store the original labels for loss computation
            labels = batch['labels'].clone()
            
            # Forward pass through student model
            student_outputs = distiller.forward_student(batch)
            student_logits = student_outputs.logits
            
            # Compute CE loss (no distillation during validation)
            loss = student_outputs.loss
            total_loss += loss.item()
            
            # Process each language in the batch
            for lang in set(lang_codes):
                # Find indices for this language
                lang_indices = [i for i, code in enumerate(lang_codes) if code == lang]
                
                if len(lang_indices) == 0:
                    continue
                
                # Track language-specific losses
                lang_loss = loss.item()  # Use the same loss for simplicity
                lang_losses[lang] += lang_loss * len(lang_indices)
                lang_counts[lang] += len(lang_indices)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    # Compute average losses by language
    avg_lang_losses = {lang: losses / lang_counts[lang] for lang, losses in lang_losses.items()}
    
    return total_loss / len(dataloader), avg_lang_losses


def train_student_model(args):
    """Train the student model via knowledge distillation."""
    # Load data for each language pair
    dataframes = {}
    
    for lang in ['de', 'fr', 'it']:
        lang_pair = f"{lang}-en"
        data_dir = os.path.join(args.data_dir, 'processed', lang_pair)
        
        if os.path.exists(data_dir):
            train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
            val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
            
            dataframes[f"{lang}_train"] = train_df
            dataframes[f"{lang}_val"] = val_df
            
            logger.info(f"Loaded {len(train_df)} training examples and {len(val_df)} validation examples for {lang_pair}")
        else:
            logger.warning(f"Data directory not found for {lang_pair}: {data_dir}")
    
    # Check if we have enough data
    if len(dataframes) == 0:
        raise ValueError("No valid data found. Please run the data download script first.")
    
    # Configure teacher adapters
    teacher_adapters = {}
    for lang in ['fr', 'it']:
        adapter_path = os.path.join(args.adapter_dir, f"lora-adapter-{lang}-en")
        if os.path.exists(adapter_path):
            teacher_adapters[lang] = adapter_path
            logger.info(f"Found adapter for {lang.upper()}-English: {adapter_path}")
        else:
            logger.warning(f"Adapter not found for {lang.upper()}-English: {adapter_path}")
    
    # Initialize distiller
    logger.info("Initializing distiller...")
    distiller = MultilingualTranslationDistiller(
        base_model_name=args.base_model,
        teacher_adapters=teacher_adapters,
        temperature=args.temperature,
        alpha=args.alpha
    )
    
    # Create datasets and dataloaders for training
    train_dfs = {lang.split('_')[0]: df for lang, df in dataframes.items() if lang.endswith('_train')}
    val_dfs = {lang.split('_')[0]: df for lang, df in dataframes.items() if lang.endswith('_val')}
    
    # Limit training data if requested
    if args.max_examples_per_language > 0:
        for lang, df in train_dfs.items():
            if len(df) > args.max_examples_per_language:
                train_dfs[lang] = df.sample(args.max_examples_per_language, random_state=42).reset_index(drop=True)
                logger.info(f"Limiting {lang} training examples to {args.max_examples_per_language}")
    
    # Create datasets
    tokenizer = distiller.get_tokenizer()
    
    train_dataset = MultilingualTranslationDataset(
        train_dfs, tokenizer, max_length=args.max_length
    )
    
    val_dataset = MultilingualTranslationDataset(
        val_dfs, tokenizer, max_length=args.max_length
    )
    
    # Create dataloaders
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
    
    # Get student model and device
    student_model = distiller.get_student_model()
    device = distiller.device
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        student_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
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
        train_loss, train_lang_losses = train_epoch(
            distiller, train_dataloader, optimizer, scheduler, device
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Log language-specific losses
        for lang, loss in train_lang_losses.items():
            logger.info(f"  {lang.upper()} train loss: {loss:.4f}")
        
        # Validate
        val_loss, val_lang_losses = validate(distiller, val_dataloader, device)
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Log language-specific losses
        for lang, loss in val_lang_losses.items():
            logger.info(f"  {lang.upper()} validation loss: {loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir = os.path.join(args.output_dir, "multilingual-translation-model")
            distiller.save_student_model(output_dir)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    logger.info("Training completed.")


def main():
    parser = argparse.ArgumentParser(description="Train a student model via knowledge distillation")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with the processed data")
    parser.add_argument("--adapter_dir", type=str, default="./outputs", help="Directory with the LoRA adapters")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the trained model")
    parser.add_argument("--max_examples_per_language", type=int, default=100000, help="Maximum number of examples per language")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="Helsinki-NLP/opus-mt-de-en", help="Base model")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train student model
    train_student_model(args)


if __name__ == "__main__":
    main() 