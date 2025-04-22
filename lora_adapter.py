#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of LoRA (Low-Rank Adaptation) for MarianMT models.
"""

import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    PeftModel, 
    PeftConfig
)

class LoraAdapter:
    """
    LoRA adapter for MarianMT models.
    
    This class handles the creation and management of LoRA adapters
    for adapting a pre-trained German-English MarianMT model to 
    other language pairs like French-English and Italian-English.
    """
    
    def __init__(
        self,
        base_model_name="Helsinki-NLP/opus-mt-de-en",
        target_language="fr",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        device=None
    ):
        """
        Initialize the LoRA adapter.
        
        Args:
            base_model_name (str): Hugging Face model identifier for the base model.
            target_language (str): Target language code ('fr' or 'it').
            lora_rank (int): Rank of the LoRA adapter matrices.
            lora_alpha (int): Scaling factor for the LoRA adapter.
            lora_dropout (float): Dropout probability for the LoRA adapter.
            device (str, optional): Device to load the model on ('cuda' or 'cpu').
        """
        self.base_model_name = base_model_name
        self.target_language = target_language
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        if device is None:
            # Prefer CUDA, then MPS, else CPU
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Load base model and tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(base_model_name)
        self.base_model = MarianMTModel.from_pretrained(base_model_name).to(self.device)
        
        # Configure LoRA
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.base_model, self.peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def get_model(self):
        """Get the LoRA-adapted model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
    
    def save_adapter(self, output_dir):
        """
        Save the LoRA adapter weights.
        
        Args:
            output_dir (str): Directory to save the adapter weights.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"LoRA adapter saved to {output_dir}")
    
    @classmethod
    def load_adapter(cls, adapter_path, base_model_name="Helsinki-NLP/opus-mt-de-en", device=None):
        """
        Load a LoRA adapter from disk.
        
        Args:
            adapter_path (str): Path to the saved adapter.
            base_model_name (str): Hugging Face model identifier for the base model.
            device (str, optional): Device to load the model on ('cuda' or 'cpu').
        
        Returns:
            tuple: (model, tokenizer) - The adapted model and tokenizer.
        """
        if device is None:
            # Prefer CUDA, then MPS, else CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Load tokenizer and base model
        tokenizer = MarianTokenizer.from_pretrained(base_model_name)
        base_model = MarianMTModel.from_pretrained(base_model_name).to(device)
        
        # Load adapter
        config = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        print(f"Loaded LoRA adapter from {adapter_path}")
        return model, tokenizer
    
    def translate(self, sentences, batch_size=8, max_length=128):
        """
        Translate a list of sentences.
        
        Args:
            sentences (list): List of sentences to translate.
            batch_size (int): Batch size for translation.
            max_length (int): Maximum length of generated translations.
        
        Returns:
            list: List of translated sentences.
        """
        self.model.eval()
        translations = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Translate
            with torch.no_grad():
                translated = self.model.generate(**inputs, max_length=max_length)
            
            # Decode
            translated_texts = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            translations.extend(translated_texts)
            
        return translations 