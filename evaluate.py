#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate the multilingual translation model.
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import time
import sys
import json
from sacrebleu.metrics import BLEU
from transformers import MarianMTModel, MarianTokenizer

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

def calculate_model_size(model_path):
    """
    Calculate the model size in MB.
    
    Args:
        model_path (str): Path to the model directory.
    
    Returns:
        float: Model size in MB.
    """
    # If model_path is a directory (Hugging Face model)
    if os.path.isdir(model_path):
        # Get the size of all files in the directory
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    
    # If model_path is a file (ONNX model)
    else:
        total_size = os.path.getsize(model_path)
    
    # Convert to MB
    size_mb = total_size / (1024 * 1024)
    
    return size_mb

def load_model(model_path, language=None, device=None):
    """
    Load a model from disk.
    
    Args:
        model_path (str): Path to the model or base model directory.
        language (str, optional): Language code for loading specific adapter model.
        device (str, optional): Device to load the model on.
    
    Returns:
        tuple: (model, tokenizer, device, model_type)
    """
    # Prefer CUDA, then MPS, else CPU
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Handle language-specific loader logic
    if language is not None:
        # Try to load language-specific model first
        adapter_path = os.path.join(os.path.dirname(model_path), f"lora-adapter-{language}-en")
        
        if os.path.exists(adapter_path) and os.path.isdir(adapter_path):
            logger.info(f"Loading LoRA adapter for {language}-EN from {adapter_path}")
            
            try:
                from models.lora_adapter import LoraAdapter
                
                # Create adapter instance
                adapter = LoraAdapter(
                    base_model_name="Helsinki-NLP/opus-mt-de-en",
                    target_language=language
                )
                
                # Load the adapter model
                model, tokenizer = LoraAdapter.load_adapter(
                    adapter_path=adapter_path,
                    device=device
                )
                
                model.eval()
                return model, tokenizer, device, "hf"
            except Exception as e:
                logger.error(f"Error loading LoRA adapter: {e}")
                logger.warning(f"Falling back to multilingual model")
    
    # If we're here, we're either loading a non-language-specific model
    # or the language-specific loading failed
    logger.info(f"Loading translation model from {model_path}")
    model = MarianMTModel.from_pretrained(model_path).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model.eval()
    
    return model, tokenizer, device, "hf"

def translate_sentence(model, tokenizer, sentence, lang_code, device, model_type="hf", max_length=128):
    """
    Translate a single sentence.
    
    Args:
        model: The translation model.
        tokenizer: The tokenizer.
        sentence (str): The sentence to translate.
        lang_code (str): The language code of the source sentence.
        device (str): The device to run inference on.
        model_type (str): The model type ('hf' or 'onnx').
        max_length (int): Maximum length of the generated translation.
    
    Returns:
        str: The translated sentence.
    """
    # Add language token
    lang_token = f"<{lang_code.upper()}>"
    sentence_with_token = f"{lang_token} {sentence}"
    
    # Tokenize
    inputs = tokenizer(sentence_with_token, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    if model_type == "hf":
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            # Use the model's configured BOS token for decoding
            forced_bos_token_id = model.config.decoder_start_token_id
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=5,
                no_repeat_ngram_size=3
            )
        
        # Decode
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log first translation for debugging
        if hasattr(translate_sentence, 'first_call') == False:
            logger.info(f"DEBUG - First translation example:")
            logger.info(f"  Input: {sentence}")
            logger.info(f"  With lang token: {sentence_with_token}")
            logger.info(f"  Output: {translation}")
            translate_sentence.first_call = True
    
    elif model_type == "onnx":
        # Convert inputs to numpy for ONNX Runtime
        ort_inputs = {k: v.numpy() for k, v in inputs.items()}
        
        # Run inference
        ort_outputs = model.run(None, ort_inputs)
        
        # This is a placeholder - actual ONNX output processing would need more work
        # In a real implementation, you'd need to handle the ONNX outputs correctly
        # and convert them back to token IDs for decoding
        translation = "ONNX translation placeholder"
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return translation

def evaluate_model(model_path, lang_codes=['de', 'fr', 'it'], test_data=None, device=None):
    """
    Evaluate the model on test data.
    
    Args:
        model_path (str): Path to the model directory.
        lang_codes (list): List of language codes to evaluate.
        test_data (dict): Dictionary mapping language codes to test data.
        device (str, optional): Device to run evaluation on.
    
    Returns:
        dict: Evaluation metrics for each language.
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize metrics
    metrics = {}
    overall_bleu = 0
    total_time = 0
    total_sentences = 0
    
    # BLEU scorer
    bleu = BLEU()
    
    # Evaluate each language
    for lang in lang_codes:
        if lang not in test_data:
            logger.warning(f"No test data for language {lang}. Skipping.")
            continue
        
        # Load language-specific model with LoRA adapter
        model, tokenizer, device, model_type = load_model(model_path, language=lang)
        
        # Get test data
        test_df = test_data[lang]
        logger.info(f"Evaluating {len(test_df)} examples for {lang.upper()}")
        
        # Prepare sources and references
        sources = list(test_df['source'])
        references = [[ref] for ref in test_df['target']]  # Format references as list of lists
        
        # Generate translations
        translations = []
        
        # Measure time
        start_time = time.time()
        
        # Translate each sentence
        for source in tqdm(sources, desc=f"Translating {lang.upper()}"):
            translation = translate_sentence(model, tokenizer, source, lang, device, model_type)
            translations.append(translation)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        avg_time_per_sentence = inference_time / len(sources)
        
        # Calculate BLEU score
        try:
            # Print sample translations for debugging
            logger.info(f"DEBUG - Sample translations for {lang.upper()}:")
            for i in range(min(3, len(translations))):
                logger.info(f"  Source: {sources[i]}")
                logger.info(f"  Reference: {references[i][0]}")
                logger.info(f"  Translation: {translations[i]}")
            
            # Convert references to required format [[ref1], [ref2], ...]
            bleu_score = bleu.corpus_score(translations, [[ref[0]] for ref in references]).score
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            logger.error(f"References sample: {references[:3]}")
            logger.error(f"Translations sample: {translations[:3]}")
            bleu_score = 0.0
        
        # Store metrics
        metrics[lang] = {
            'bleu': bleu_score,
            'avg_time_per_sentence': avg_time_per_sentence,
            'total_time': inference_time,
            'num_examples': len(test_df)
        }
        
        logger.info(f"Evaluation results for {lang.upper()}:")
        logger.info(f"  BLEU: {bleu_score:.2f}")
        logger.info(f"  Avg. time per sentence: {avg_time_per_sentence * 1000:.2f} ms")
        logger.info(f"  Total time: {inference_time:.2f} s")
        
        # Update overall metrics
        overall_bleu += bleu_score * len(test_df)
        total_time += inference_time
        total_sentences += len(test_df)
        
        # Free up memory by removing the model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate overall metrics
    if total_sentences > 0:
        overall_bleu /= total_sentences
    else:
        overall_bleu = 0
    
    # Add overall metrics
    metrics['overall'] = {
        'bleu': overall_bleu,
        'total_time': total_time,
        'num_examples': total_sentences,
        'avg_time_per_sentence': total_time / total_sentences if total_sentences > 0 else 0
    }
    
    logger.info("Overall evaluation results:")
    logger.info(f"  BLEU: {overall_bleu:.2f}")
    logger.info(f"  Total time: {total_time:.2f} s")
    logger.info(f"  Total examples: {total_sentences}")
    logger.info(f"  Avg. time per sentence: {metrics['overall']['avg_time_per_sentence'] * 1000:.2f} ms")
    
    return metrics

def measure_memory_usage(model_path, lang_codes=['de', 'fr', 'it'], sentences=None, device=None):
    """
    Measure memory usage during inference.
    
    Args:
        model_path (str): Path to the model directory.
        lang_codes (list): List of language codes to measure.
        sentences (list): List of sentences to translate.
        device (str, optional): Device to run inference on.
    
    Returns:
        dict: Memory usage metrics.
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize metrics
    memory_metrics = {}
    
    # Skip if not on CUDA or if no model_path provided
    if device != "cuda" or model_path is None:
        logger.warning("Memory measurement only supported for models on CUDA. Skipping.")
        return {"peak_memory_mb": 0}
    
    # Ensure sentences is not None
    if sentences is None:
        sentences = ["Hello, how are you?", "This is a test sentence.", "I need to translate this."]
    
    # Measure baseline memory usage
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    baseline_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # Measure memory usage during inference
    torch.cuda.reset_peak_memory_stats()
    
    # Translate sentences for each language
    for lang in lang_codes:
        # Load model for this language
        model, tokenizer, device, model_type = load_model(model_path, language=lang)
        
        # Run inference
        for sentence in sentences:
            _ = translate_sentence(model, tokenizer, sentence, lang, device, model_type)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Get peak memory usage
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # Store metrics
    memory_metrics["baseline_memory_mb"] = baseline_memory
    memory_metrics["peak_memory_mb"] = peak_memory
    memory_metrics["used_memory_mb"] = peak_memory - baseline_memory
    
    logger.info(f"Memory usage:")
    logger.info(f"  Baseline memory: {baseline_memory:.2f} MB")
    logger.info(f"  Peak memory: {peak_memory:.2f} MB")
    logger.info(f"  Used memory: {memory_metrics['used_memory_mb']:.2f} MB")
    
    return memory_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate multilingual translation model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with the processed data")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation", help="Directory to save evaluation results")
    
    # Evaluation arguments
    parser.add_argument("--languages", nargs="+", default=["de", "fr", "it"], help="Languages to evaluate")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of test samples to use per language")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate model size
    model_size = calculate_model_size(args.model_path)
    logger.info(f"Model size: {model_size:.2f} MB")
    
    # Load test data
    test_data = {}
    
    for lang in args.languages:
        lang_pair = f"{lang}-en"
        test_path = os.path.join(args.data_dir, lang_pair, 'test.csv')
        
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            
            # Limit number of samples
            if len(test_df) > args.max_samples:
                test_df = test_df.sample(args.max_samples, random_state=42)
                
            test_data[lang] = test_df
            logger.info(f"Loaded {len(test_df)} test examples for {lang_pair}")
        else:
            logger.warning(f"Test data not found for {lang_pair}: {test_path}")
    
    # Evaluate model
    metrics = evaluate_model(args.model_path, args.languages, test_data)
    
    # Measure memory usage (sample sentences)
    sample_sentences = ["Hello, how are you?", "This is a test sentence.", "I need to translate this."]
    memory_metrics = measure_memory_usage(args.model_path, args.languages, sample_sentences)
    
    # Combine all metrics
    all_metrics = {
        "model_size_mb": model_size,
        "memory": memory_metrics,
        "translation": metrics
    }
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    # Print overall summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Model: {args.model_path}")
    print(f"Size: {model_size:.2f} MB")
    print(f"Overall BLEU: {metrics['overall']['bleu']:.2f}")
    print(f"Avg. inference time: {metrics['overall']['avg_time_per_sentence'] * 1000:.2f} ms per sentence")
    print(f"Peak memory usage: {memory_metrics['peak_memory_mb']:.2f} MB")
    print("=============================\n")


if __name__ == "__main__":
    main() 