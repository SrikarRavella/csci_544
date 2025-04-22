#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of knowledge distillation for multilingual translation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer, MarianConfig

class MultilingualTranslationDistiller:
    """
    Knowledge distillation for multilingual translation.
    
    This class implements knowledge distillation from multiple teacher models
    (one for each language pair) to a single student model for all language pairs.
    """
    
    def __init__(
        self,
        base_model_name="Helsinki-NLP/opus-mt-de-en",
        teacher_adapters=None,
        student_config=None,
        temperature=2.0,
        alpha=0.5,
        device=None
    ):
        """
        Initialize the distiller.
        
        Args:
            base_model_name (str): Hugging Face model identifier for the base model.
            teacher_adapters (dict): Dictionary mapping language codes to adapter paths.
                                    Example: {'fr': 'path/to/fr_adapter', 'it': 'path/to/it_adapter'}
            student_config (dict, optional): Configuration for the student model.
                                           If None, a reduced version of the base model is used.
            temperature (float): Temperature for the soft targets.
            alpha (float): Weight for the distillation loss (vs. hard target loss).
            device (str, optional): Device to load the models on ('cuda' or 'cpu').
        """
        self.base_model_name = base_model_name
        self.teacher_adapters = teacher_adapters or {}
        self.temperature = temperature
        self.alpha = alpha
        
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
        
        # Load tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(base_model_name)
        
        # Add special tokens for language identification
        special_tokens = {'additional_special_tokens': ['<DE>', '<FR>', '<IT>']}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        
        # Load teacher models
        self.teacher_models = self._load_teacher_models()
        
        # Initialize student model
        self.student_model = self._initialize_student_model(student_config)
        
        # Resize token embeddings, falling back to CPU for MPS unsupported ops
        if self.device == "mps":
            # Temporarily move to CPU for resizing
            self.student_model.to("cpu")
            self.student_model.resize_token_embeddings(len(self.tokenizer))
            # Move back to original device
            self.student_model.to("mps")
        else:
            self.student_model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Initialized distillation with {len(self.teacher_models)} teacher models")
        print(f"Model will be trained on device: {self.device}")
    
    def _load_teacher_models(self):
        """Load teacher models for each language pair."""
        from models.lora_adapter import LoraAdapter
        
        teacher_models = {}
        
        # Load base German model
        print(f"Loading base model for German-English...")
        de_model = MarianMTModel.from_pretrained(self.base_model_name).to(self.device)
        de_model.eval()  # Set to evaluation mode
        teacher_models['de'] = de_model
        
        # Load adapted models for other languages
        for lang, adapter_path in self.teacher_adapters.items():
            print(f"Loading adapted model for {lang.upper()}-English...")
            if os.path.exists(adapter_path):
                model, _ = LoraAdapter.load_adapter(
                    adapter_path=adapter_path,
                    base_model_name=self.base_model_name,
                    device=self.device
                )
                model.eval()  # Set to evaluation mode
                teacher_models[lang] = model
            else:
                print(f"Warning: Adapter path for {lang} not found: {adapter_path}")
        
        return teacher_models
    
    def _initialize_student_model(self, student_config=None):
        """Initialize a smaller student model."""
        # Load base configuration
        base_config = MarianConfig.from_pretrained(self.base_model_name)
        
        # Create student configuration (reduced size)
        if student_config is None:
            # Default: reduce the number of layers by half
            student_config = base_config
            student_config.encoder_layers = base_config.encoder_layers // 2
            student_config.decoder_layers = base_config.decoder_layers // 2
        
        print(f"Initializing student model with {student_config.encoder_layers} encoder layers and {student_config.decoder_layers} decoder layers")
        
        # Initialize model with configuration
        student_model = MarianMTModel(student_config).to(self.device)
        
        # Initialize student weights from the base model
        # This helps speed up convergence
        base_model = MarianMTModel.from_pretrained(self.base_model_name)
        
        # Copy encoder layers (up to the student's number of layers)
        for i in range(student_config.encoder_layers):
            student_layer_idx = i
            teacher_layer_idx = i * base_config.encoder_layers // student_config.encoder_layers
            
            # Get source and target modules
            teacher_layer = base_model.model.encoder.layers[teacher_layer_idx]
            student_layer = student_model.model.encoder.layers[student_layer_idx]
            
            # Copy weights
            student_layer.load_state_dict(teacher_layer.state_dict())
        
        # Copy decoder layers (up to the student's number of layers)
        for i in range(student_config.decoder_layers):
            student_layer_idx = i
            teacher_layer_idx = i * base_config.decoder_layers // student_config.decoder_layers
            
            # Get source and target modules
            teacher_layer = base_model.model.decoder.layers[teacher_layer_idx]
            student_layer = student_model.model.decoder.layers[student_layer_idx]
            
            # Copy weights
            student_layer.load_state_dict(teacher_layer.state_dict())
        
        # Copy embedding and output layer weights
        student_model.model.shared.load_state_dict(base_model.model.shared.state_dict())
        student_model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        
        return student_model
    
    def get_student_model(self):
        """Get the student model."""
        return self.student_model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
    
    def knowledge_distillation_loss(self, student_logits, teacher_logits, labels, ignore_index=-100):
        """
        Compute the knowledge distillation loss.
        
        Args:
            student_logits (torch.Tensor): Logits from the student model.
            teacher_logits (torch.Tensor): Logits from the teacher model.
            labels (torch.Tensor): Ground truth labels.
            ignore_index (int): Index to ignore in the loss computation.
        
        Returns:
            tuple: (total_loss, distillation_loss, ce_loss)
        """
        # Align teacher logits to student vocab size
        t_v = teacher_logits.size(-1)
        s_v = student_logits.size(-1)
        if t_v < s_v:
            # Pad teacher logits with large negative values for new tokens
            pad_shape = (teacher_logits.size(0), teacher_logits.size(1), s_v - t_v)
            pad = teacher_logits.new_full(pad_shape, -1e9)
            teacher_logits = torch.cat([teacher_logits, pad], dim=-1)
        elif t_v > s_v:
            # Truncate extra logits
            teacher_logits = teacher_logits[:, :, :s_v]
        # Temperature-scaled softmax
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distillation_loss = -torch.sum(soft_targets * soft_prob, dim=-1)
        
        # Mask out padding tokens
        padding_mask = (labels != ignore_index).float()
        distillation_loss = torch.sum(distillation_loss * padding_mask) / torch.sum(padding_mask)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index
        )
        
        # Scale distillation loss
        distillation_loss = distillation_loss * (self.temperature ** 2)
        
        # Combine losses
        total_loss = (self.alpha * distillation_loss) + ((1 - self.alpha) * ce_loss)
        
        return total_loss, distillation_loss, ce_loss
    
    def forward_student(self, batch):
        """Forward pass through the student model."""
        return self.student_model(**batch)
    
    def forward_teacher(self, batch, lang_code):
        """Forward pass through the appropriate teacher model."""
        if lang_code not in self.teacher_models:
            raise ValueError(f"No teacher model found for language code: {lang_code}")
        
        return self.teacher_models[lang_code](**batch)
    
    def save_student_model(self, output_dir):
        """
        Save the student model and tokenizer.
        
        Args:
            output_dir (str): Directory to save the model.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.student_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Saved student model to {output_dir}")
    
    def translate(self, sentences, lang_code='de', batch_size=8, max_length=128):
        """
        Translate a list of sentences using the student model.
        
        Args:
            sentences (list): List of sentences to translate.
            lang_code (str): Language code for the source language.
            batch_size (int): Batch size for translation.
            max_length (int): Maximum length of generated translations.
        
        Returns:
            list: List of translated sentences.
        """
        self.student_model.eval()
        translations = []
        
        # Add language token
        lang_token = f"<{lang_code.upper()}>"
        sentences_with_token = [f"{lang_token} {sent}" for sent in sentences]
        
        # Process in batches
        for i in range(0, len(sentences_with_token), batch_size):
            batch = sentences_with_token[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Translate
            with torch.no_grad():
                translated = self.student_model.generate(**inputs, max_length=max_length)
            
            # Decode
            translated_texts = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            translations.extend(translated_texts)
            
        return translations 