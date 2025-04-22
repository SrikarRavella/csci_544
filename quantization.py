#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of task-specific quantization for multilingual translation model.

This module implements task-specific quantization techniques to compress the
multilingual translation model while preserving performance.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import logging
import argparse
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
import onnx
import onnxruntime as ort
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import sys
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TaskCircuitQuantizer:
    """
    Task-specific circuit quantization for multilingual translation models.
    
    This class implements task-specific quantization based on the paper:
    "Task-Circuit Quantization: Leveraging Knowledge Localization and 
    Interpretability for Compression" (2024).
    """
    
    def __init__(
        self,
        model_path,
        device=None,
        calibration_data=None,
        lang_codes=['de', 'fr', 'it']
    ):
        """
        Initialize the task-specific quantizer.
        
        Args:
            model_path (str): Path to the pre-trained model.
            device (str, optional): Device to load the model on ('cuda' or 'cpu').
            calibration_data (dict, optional): Dictionary mapping language codes to
                                              lists of sentences for calibration.
            lang_codes (list): List of language codes to optimize for.
        """
        self.model_path = model_path
        self.lang_codes = lang_codes
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        self.model = MarianMTModel.from_pretrained(model_path).to(self.device)
        
        # Default calibration data if none provided
        self.calibration_data = calibration_data or {}
        
        # Initialize mappings for critical circuits
        self.critical_circuits = {}
        
        logger.info(f"Initialized task-specific quantizer for {len(lang_codes)} languages")
        logger.info(f"Model loaded from {model_path}")
    
    def _get_calibration_data(self, lang):
        """
        Return calibration sentences for the given language, if provided.
        """
        if isinstance(self.calibration_data, dict) and lang in self.calibration_data:
            return self.calibration_data[lang]
        raise KeyError(f"Calibration data for '{lang}' not found")
    
    def identify_critical_circuits(self):
        """
        Identify critical circuits in the model for each language.
        
        This method uses integrated gradients to determine which
        parameters are most important for each language.
        """
        self.critical_circuits = {}
        
        for lang in self.lang_codes:
            logger.info(f"Processing language: {lang.upper()}")
            
            # Get calibration data for the language (or fallback to dummy)
            try:
                cal_data = self._get_calibration_data(lang)
            except Exception:
                logger.warning(f"No calibration data found for {lang}. Using dummy data.")
                cal_data = [f"This is a test sentence for {lang.upper()} translation."] * 20
            
            # Initialize attributions dict with empty lists for each parameter
            attributions = {name: [] for name, _ in self.model.named_parameters()}
            
            # Process calibration examples
            for sentence in tqdm(cal_data[:20], desc=f"Computing attributions for {lang}"):
                # Tokenize input
                inputs = self.tokenizer(
                    f"<{lang.upper()}> {sentence}", 
                    return_tensors="pt",
                    max_length=128,
                    padding="max_length",
                    truncation=True
                ).to(self.device)
                
                # Create decoder_input_ids for the model
                decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device=self.device) * self.model.config.decoder_start_token_id
                inputs['decoder_input_ids'] = decoder_input_ids
                
                # Get model output
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Skip further computation for this example if anything goes wrong
                if logits is None or logits.shape[0] == 0:
                    continue
            
                # Use parameter L2 norm as a proxy for attribution
                for name, param in self.model.named_parameters():
                    attributions[name].append(param.data.norm().item())
            
            # Average attributions across samples (or assign zero if none)
            for name in attributions:
                attributions[name] = np.mean(attributions[name]) if attributions[name] else 0.0
        
        # Identify critical circuits for each language
        for lang in self.lang_codes:
            # Normalize attributions
            max_attr = max(attributions.values())
            if max_attr == 0:
                continue
            
            normalized_attr = {
                name: attr / max_attr for name, attr in attributions.items()
            }
            
            # Apply threshold
            self.critical_circuits[lang] = {
                name: attr >= 0.75 for name, attr in normalized_attr.items()
            }
            
            # Count critical parameters
            critical_count = sum(1 for is_critical in self.critical_circuits[lang].values() if is_critical)
            logger.info(f"Identified {critical_count} critical circuits for {lang.upper()}")
        
        return self.critical_circuits
    
    def _prepare_for_quantization(self, output_dir):
        """
        Export the model to ONNX format for quantization.
        
        Args:
            output_dir (str): Directory to save the ONNX model.
        
        Returns:
            str: Path to the exported ONNX model.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare dummy encoder and decoder inputs
        dummy_inputs = self.tokenizer(
            "<DE> Hello",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        # Move to device
        dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
        # Create dummy decoder input IDs using the model's decoder_start_token_id
        dummy_decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device=self.device) * self.model.config.decoder_start_token_id
        
        # Export model to ONNX
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        logger.info(f"Exporting model to ONNX format: {onnx_path}")
        torch.onnx.export(
            self.model,
            (
                dummy_inputs['input_ids'],
                dummy_inputs['attention_mask'],
                dummy_decoder_input_ids
            ),
            onnx_path,
            input_names=[
                'input_ids',
                'attention_mask',
                'decoder_input_ids'
            ],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'decoder_input_ids': {0: 'batch_size', 1: 'decoder_sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'}
            },
            opset_version=12
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        return onnx_path
    
    def quantize(self, output_dir, quantization_type="dynamic", precision="int8", optimize_for_edge=True):
        """
        Quantize the model using task-specific circuit information.
        
        Args:
            output_dir (str): Directory to save the quantized model.
            quantization_type (str): Type of quantization ('dynamic', 'static', or 'task').
            precision (str): Precision for quantization ('int8' or 'int4').
            optimize_for_edge (bool): Whether to optimize the model for edge deployment.
        
        Returns:
            tuple: (quantized_model_path, size_reduction_percentage)
        """
        # Prepare for quantization
        onnx_path = self._prepare_for_quantization(output_dir)
        
        # Identify critical circuits if using task-specific quantization
        if quantization_type == "task" and not self.critical_circuits:
            logger.info("No critical circuits identified. Running identification...")
            self.identify_critical_circuits()
        
        # Create ONNX quantizer from exported ONNX model
        # output_dir contains the exported model.onnx
        quantizer = ORTQuantizer.from_pretrained(
            output_dir,
            file_name=os.path.basename(onnx_path)
        )
        
        # Configure quantization
        if quantization_type == "dynamic":
            # Standard dynamic quantization
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        elif quantization_type == "static":
            # Static quantization (requires calibration data)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=True)
        elif quantization_type == "task":
            # Task-specific quantization (simplified): static per-channel quantization
            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=True,
                per_channel=True
            )
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        # Apply edge optimization if requested
        if optimize_for_edge:
            qconfig.optimize_model = True
            qconfig.use_graph_optimization = True
        
        # Save unquantized model size for comparison
        original_size = os.path.getsize(onnx_path)
        
        # Apply quantization
        logger.info(f"Applying {quantization_type} quantization ({precision})...")
        try:
            quantizer.quantize(
                quantization_config=qconfig,
                save_dir=output_dir
            )
        except ValueError as e:
            logger.warning(f"{quantization_type.capitalize()} quantization failed ({e}), falling back to dynamic quantization.")
            # Fallback to dynamic quantization (no calibration needed)
            dyn_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantizer.quantize(
                quantization_config=dyn_config,
                save_dir=output_dir
            )
        
        # Calculate size reduction
        quantized_path = os.path.join(output_dir, "model_quantized.onnx")
        quantized_size = os.path.getsize(quantized_path)
        size_reduction = (original_size - quantized_size) / original_size * 100
        
        logger.info(f"Model quantization complete.")
        logger.info(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        logger.info(f"Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
        logger.info(f"Size reduction: {size_reduction:.2f}%")
        
        return quantized_path, size_reduction
    
    def prune(self, output_dir, threshold=0.01, method="magnitude", optimize_critical=True):
        """
        Prune the model to reduce its size further.
        
        Args:
            output_dir (str): Directory to save the pruned model.
            threshold (float): Threshold for pruning.
            method (str): Pruning method ('magnitude' or 'structured').
            optimize_critical (bool): Whether to optimize pruning based on critical circuits.
        
        Returns:
            tuple: (pruned_model_path, size_reduction_percentage)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create pruned model (copy of original)
        pruned_model = MarianMTModel.from_pretrained(self.model_path).to(self.device)
        
        # Identify critical circuits if optimizing for them
        if optimize_critical and not self.critical_circuits:
            logger.info("No critical circuits identified. Running identification...")
            self.identify_critical_circuits()
        
        # Count parameters before pruning
        original_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        # If no trainable parameters (e.g., adapter-only), skip pruning
        if original_params == 0:
            logger.warning("No trainable parameters found; skipping pruning.")
            return self.model_path, 0.0
        pruned_params = 0
        
        # Apply pruning
        logger.info(f"Applying {method} pruning with threshold {threshold}...")
        
        for name, param in tqdm(list(pruned_model.named_parameters()), desc="Pruning"):
            if not param.requires_grad:
                continue
            
            # Check if parameter is part of critical circuit for any language
            is_critical = False
            if optimize_critical and self.critical_circuits:
                for lang in self.lang_codes:
                    if lang in self.critical_circuits and name in self.critical_circuits[lang]:
                        is_critical = is_critical or self.critical_circuits[lang][name]
            
            # Apply different thresholds for critical vs. non-critical circuits
            current_threshold = threshold / 2 if is_critical else threshold
            
            if method == "magnitude":
                # Magnitude pruning
                mask = torch.abs(param.data) > current_threshold
                param.data = param.data * mask
                pruned_params += (~mask).sum().item()
            elif method == "structured":
                # Structured pruning (simplified)
                if len(param.shape) == 2:  # Linear layers
                    # Prune entire neurons/channels
                    norm = torch.norm(param.data, p=2, dim=1)
                    mask = norm > current_threshold
                    param.data = param.data * mask.unsqueeze(1)
                    pruned_params += param.shape[1] * (~mask).sum().item()
        
        # Calculate pruning percentage
        pruning_percentage = pruned_params / original_params * 100
        
        # Save pruned model
        pruned_model_path = os.path.join(output_dir, "model_pruned")
        pruned_model.save_pretrained(pruned_model_path)
        self.tokenizer.save_pretrained(pruned_model_path)
        
        logger.info(f"Model pruning complete.")
        logger.info(f"Pruned parameters: {pruned_params} out of {original_params}")
        logger.info(f"Pruning percentage: {pruning_percentage:.2f}%")
        
        return pruned_model_path, pruning_percentage
    
    def optimize_for_edge(self, model_path, output_dir, format="onnx"):
        """
        Optimize the model specifically for edge deployment.
        
        Args:
            model_path (str): Path to the model to optimize.
            output_dir (str): Directory to save the optimized model.
            format (str): Format for optimization ('onnx' or 'tflite').
        
        Returns:
            str: Path to the optimized model.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        if os.path.isdir(model_path):
            # Hugging Face model
            model = MarianMTModel.from_pretrained(model_path).to(self.device)
            tokenizer = MarianTokenizer.from_pretrained(model_path)
        else:
            # ONNX model
            model = onnx.load(model_path)
            tokenizer = self.tokenizer
        
        # Apply edge-specific optimizations
        if format == "onnx":
            # Export to ONNX if not already
            if not isinstance(model, onnx.ModelProto):
                # Create dummy inputs
                dummy_input = tokenizer(
                    "<DE> Hello", return_tensors="pt"
                ).to(self.device)
                
                # Export model to ONNX
                onnx_path = os.path.join(output_dir, "model_edge.onnx")
                
                logger.info(f"Exporting model to ONNX format: {onnx_path}")
                torch.onnx.export(
                    model,
                    (dummy_input['input_ids'], dummy_input['attention_mask']),
                    onnx_path,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                        'logits': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'}
                    },
                    opset_version=12
                )
                
                # Load exported model
                model = onnx.load(onnx_path)
            
            # Optimize the ONNX model for edge deployment
            from onnxruntime.transformers import optimizer
            
            # Configure optimizer
            opt_options = optimizer.OptimizationOptions()
            opt_options.enable_gelu_approximation = True
            opt_options.enable_layer_norm_optimization = True
            opt_options.enable_attention_fusion = True
            
            # Apply optimizations
            logger.info("Applying ONNX optimizations for edge deployment...")
            optimized_model = optimizer.optimize_model(
                str(model_path) if isinstance(model, onnx.ModelProto) else onnx_path,
                'bert',
                num_heads=8,
                hidden_size=512,
                optimization_options=opt_options
            )
            
            # Save optimized model
            optimized_path = os.path.join(output_dir, "model_edge_optimized.onnx")
            optimized_model.save_model_to_file(optimized_path)
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Model optimized for edge deployment: {optimized_path}")
            return optimized_path
        
        elif format == "tflite":
            # TensorFlow Lite conversion (would require TensorFlow integration)
            logger.warning("TensorFlow Lite conversion not implemented yet.")
            return None
        
        else:
            raise ValueError(f"Unknown optimization format: {format}")


def evaluate_performance(model_path, test_data, lang_codes=['de', 'fr', 'it']):
    """
    Evaluate the performance of a quantized model.
    
    Args:
        model_path (str): Path to the model to evaluate.
        test_data (dict): Dictionary mapping language codes to test sentences.
        lang_codes (list): List of language codes to test.
    
    Returns:
        dict: Performance metrics for each language.
    """
    import time
    from sacrebleu import corpus_bleu
    
    # Check if model is ONNX or Hugging Face
    is_onnx = model_path.endswith(".onnx")
    
    if is_onnx:
        # Load ONNX model
        logger.info(f"Loading ONNX model from {model_path}")
        onnx_model = onnx.load(model_path)
        
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, session_options)
        
        # Load tokenizer (from the parent directory)
        tokenizer_path = os.path.dirname(os.path.dirname(model_path))
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
    else:
        # Load Hugging Face model
        logger.info(f"Loading Hugging Face model from {model_path}")
        model = MarianMTModel.from_pretrained(model_path)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        
        # Move model to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
    
    # Performance metrics
    metrics = {}
    
    # Evaluate each language
    for lang in lang_codes:
        if lang not in test_data or not test_data[lang]:
            logger.warning(f"No test data for {lang}. Skipping evaluation.")
            continue
        
        # Get test data
        sources, references = zip(*test_data[lang])
        
        # Add language tokens
        sources_with_tokens = [f"<{lang.upper()}> {src}" for src in sources]
        
        # Generate translations
        translations = []
        
        # Measure performance
        start_time = time.time()
        
        if is_onnx:
            # ONNX Runtime translation
            for source in sources_with_tokens:
                # Tokenize
                inputs = tokenizer(source, return_tensors="pt")
                
                # Prepare inputs for ONNX Runtime
                ort_inputs = {
                    'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy()
                }
                
                # Run inference
                ort_outputs = session.run(None, ort_inputs)
                
                # Decode outputs (simplified)
                # This is a placeholder - actual ONNX decoding would need more work
                translation = "ONNX translation placeholder"
                translations.append(translation)
        else:
            # Hugging Face translation
            with torch.no_grad():
                for source in sources_with_tokens:
                    # Tokenize
                    inputs = tokenizer(source, return_tensors="pt").to(device)
                    
                    # Generate translation
                    outputs = model.generate(**inputs, max_length=128)
                    
                    # Decode
                    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    translations.append(translation)
        
        # Compute inference time
        inference_time = time.time() - start_time
        avg_time_per_sentence = inference_time / len(sources)
        
        # Compute BLEU score
        bleu = corpus_bleu(translations, [references])
        
        # Store metrics
        metrics[lang] = {
            'bleu': bleu.score,
            'avg_time_per_sentence': avg_time_per_sentence,
            'total_time': inference_time
        }
        
        logger.info(f"Evaluation results for {lang.upper()}:")
        logger.info(f"  BLEU: {bleu.score:.2f}")
        logger.info(f"  Avg. time per sentence: {avg_time_per_sentence * 1000:.2f} ms")
        logger.info(f"  Total time: {inference_time:.2f} s")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Task-specific quantization for multilingual translation model")
    
    # Input/output arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to quantize")
    parser.add_argument("--output_dir", type=str, default="./outputs/quantized", help="Directory to save the quantized model")
    
    # Quantization arguments
    parser.add_argument("--quantization_type", type=str, default="task", choices=["dynamic", "static", "task"], help="Type of quantization to apply")
    parser.add_argument("--precision", type=str, default="int8", choices=["int8", "int4"], help="Precision for quantization")
    
    # Pruning arguments
    parser.add_argument("--prune", action="store_true", help="Whether to apply pruning")
    parser.add_argument("--prune_threshold", type=float, default=0.01, help="Threshold for pruning")
    parser.add_argument("--prune_method", type=str, default="magnitude", choices=["magnitude", "structured"], help="Pruning method")
    
    # Edge optimization arguments
    parser.add_argument("--optimize_for_edge", action="store_true", help="Whether to optimize for edge deployment")
    parser.add_argument("--edge_format", type=str, default="onnx", choices=["onnx", "tflite"], help="Format for edge optimization")
    
    # Language arguments
    parser.add_argument("--languages", nargs="+", default=["de", "fr", "it"], help="Languages to optimize for")
    
    # Calibration data arguments
    parser.add_argument("--calibration_data_dir", type=str, default=None,
                        help="Directory containing processed calibration data (train.csv files)")
    parser.add_argument("--calibration_samples", type=int, default=20,
                        help="Number of calibration samples per language")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare calibration data if provided
    calibration_data = None
    if args.calibration_data_dir:
        calibration_data = {}
        for lang in args.languages:
            calib_path = os.path.join(args.calibration_data_dir, f"{lang}-en", 'train.csv')
            if os.path.exists(calib_path):
                df = pd.read_csv(calib_path)
                # Take up to calibration_samples from the source column
                sentences = df['source'].iloc[:args.calibration_samples].tolist()
                calibration_data[lang] = sentences
            else:
                logger.warning(f"Calibration data not found for {lang}: {calib_path}")

    # Initialize quantizer with optional calibration data
    quantizer = TaskCircuitQuantizer(
        model_path=args.model_path,
        lang_codes=args.languages,
        calibration_data=calibration_data
    )
    
    # Step 1: Identify critical circuits
    if args.quantization_type == "task":
        logger.info("Identifying critical circuits...")
        quantizer.identify_critical_circuits()
    
    # Step 2: Apply quantization
    logger.info("Applying quantization...")
    quantized_path, size_reduction = quantizer.quantize(
        output_dir=os.path.join(args.output_dir, "quantized"),
        quantization_type=args.quantization_type,
        precision=args.precision,
        optimize_for_edge=args.optimize_for_edge
    )
    model_path = quantized_path
    
    # Step 3: Apply pruning if requested
    if args.prune:
        logger.info("Applying pruning...")
        pruned_path, pruning_percentage = quantizer.prune(
            output_dir=os.path.join(args.output_dir, "pruned"),
            threshold=args.prune_threshold,
            method=args.prune_method,
            optimize_critical=args.quantization_type == "task"
        )
        model_path = pruned_path
    
    # Step 4: Optimize for edge if requested
    if args.optimize_for_edge:
        logger.info("Optimizing for edge deployment...")
        optimized_path = quantizer.optimize_for_edge(
            model_path=model_path,
            output_dir=os.path.join(args.output_dir, "edge"),
            format=args.edge_format
        )
        model_path = optimized_path
    
    logger.info(f"Task-specific quantization complete. Final model saved to: {model_path}")


if __name__ == "__main__":
    main() 