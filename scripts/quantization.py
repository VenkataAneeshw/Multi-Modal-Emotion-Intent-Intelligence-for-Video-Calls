#!/usr/bin/env python3
"""
Advanced Model Quantization and Optimization for EMOTIA
Supports INT8, FP16 quantization, pruning, and edge deployment
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedQuantizer:
    """Advanced quantization utilities for EMOTIA models"""

    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.quantized_model = None
        self.calibration_data = []

    def prepare_for_quantization(self) -> nn.Module:
        """Prepare model for quantization-aware training"""
        # Fuse Conv2d + BatchNorm2d layers
        self.model = self._fuse_modules()

        # Insert quantization stubs
        self.model = self._insert_quant_stubs()

        # Set quantization config
        self.model.qconfig = quant.get_default_qat_qconfig('fbgemm')

        # Prepare for QAT
        quant.prepare_qat(self.model, inplace=True)

        logger.info("Model prepared for quantization-aware training")
        return self.model

    def _fuse_modules(self) -> nn.Module:
        """Fuse compatible layers for better quantization"""
        fusion_patterns = [
            ['conv1', 'bn1'],
            ['conv2', 'bn2'],
            ['conv3', 'bn3'],
        ]

        for pattern in fusion_patterns:
            try:
                quant.fuse_modules(self.model, pattern, inplace=True)
                logger.info(f"Fused modules: {pattern}")
            except Exception as e:
                logger.warning(f"Could not fuse {pattern}: {e}")

        return self.model

    def _insert_quant_stubs(self) -> nn.Module:
        """Insert quantization and dequantization stubs"""
        # Add quant stubs at model input
        self.model.quant = QuantStub()
        self.model.dequant = DeQuantStub()

        return self.model

    def calibrate(self, calibration_loader: DataLoader, num_batches: int = 100):
        """Calibrate quantization parameters"""
        logger.info("Starting quantization calibration...")

        self.model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_loader):
                if i >= num_batches:
                    break

                # Forward pass for calibration
                _ = self.model(inputs)

                if i % 20 == 0:
                    logger.info(f"Calibration progress: {i}/{num_batches}")

        logger.info("Calibration completed")

    def convert_to_quantized(self) -> nn.Module:
        """Convert to quantized model"""
        logger.info("Converting to quantized model...")

        # Convert to quantized model
        self.quantized_model = quant.convert(self.model.eval(), inplace=False)

        logger.info("Model quantized successfully")
        return self.quantized_model

    def quantize_static(self, calibration_loader: DataLoader) -> nn.Module:
        """Perform static quantization"""
        # Prepare for static quantization
        self.model.qconfig = quant.get_default_qconfig('fbgemm')
        quant.prepare(self.model, inplace=True)

        # Calibrate
        self.calibrate(calibration_loader)

        # Convert
        return self.convert_to_quantized()

    def quantize_dynamic(self) -> nn.Module:
        """Perform dynamic quantization"""
        logger.info("Performing dynamic quantization...")

        # Dynamic quantization for LSTM/GRU layers
        self.quantized_model = quant.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8,
            inplace=False
        )

        logger.info("Dynamic quantization completed")
        return self.quantized_model

class AdvancedPruner:
    """Advanced model pruning utilities"""

    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.pruned_model = None

    def apply_structured_pruning(self, amount: float = 0.3):
        """Apply structured pruning to convolutional layers"""
        logger.info(f"Applying structured pruning with amount: {amount}")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                logger.info(f"Pruned Conv2d layer: {name}")

        return self.model

    def apply_unstructured_pruning(self, amount: float = 0.2):
        """Apply unstructured pruning"""
        logger.info(f"Applying unstructured pruning with amount: {amount}")

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                logger.info(f"Pruned layer: {name}")

        return self.model

    def remove_pruning_masks(self):
        """Remove pruning masks and make pruning permanent"""
        logger.info("Removing pruning masks...")

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.remove(module, 'weight')

        logger.info("Pruning masks removed")
        return self.model

class ModelOptimizer:
    """Comprehensive model optimization pipeline"""

    def __init__(self, model_path: str, config_path: str):
        self.model_path = Path(model_path)
        self.config = self._load_config(config_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_config(self, config_path: str) -> Dict:
        """Load optimization configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from {self.model_path}")

        # Import model classes (adjust based on your model structure)
        from models.advanced.advanced_fusion import AdvancedFusionModel

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = AdvancedFusionModel(self.config['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")
        return self.model

    def optimize_pipeline(self, output_dir: str = 'optimized_models'):
        """Run complete optimization pipeline"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Pruning
        if self.config.get('pruning', {}).get('enabled', False):
            pruner = AdvancedPruner(self.model, self.config['pruning'])
            if self.config['pruning']['type'] == 'structured':
                self.model = pruner.apply_structured_pruning(
                    self.config['pruning']['amount']
                )
            else:
                self.model = pruner.apply_unstructured_pruning(
                    self.config['pruning']['amount']
                )
            pruner.remove_pruning_masks()

            # Save pruned model
            self._save_model(self.model, output_dir / 'pruned_model.pth')

        # 2. Quantization
        if self.config.get('quantization', {}).get('enabled', False):
            quantizer = AdvancedQuantizer(self.model, self.config['quantization'])

            if self.config['quantization']['type'] == 'static':
                # Would need calibration data here
                pass
            elif self.config['quantization']['type'] == 'dynamic':
                self.model = quantizer.quantize_dynamic()
            elif self.config['quantization']['type'] == 'qat':
                self.model = quantizer.prepare_for_quantization()
                # Would need QAT training here
                self.model = quantizer.convert_to_quantized()

            # Save quantized model
            self._save_model(self.model, output_dir / 'quantized_model.pth')

        # 3. ONNX Export
        if self.config.get('onnx', {}).get('enabled', False):
            self._export_onnx(output_dir / 'model.onnx')

        # 4. TensorRT Optimization (if available)
        if self.config.get('tensorrt', {}).get('enabled', False):
            self._optimize_tensorrt(output_dir)

        logger.info("Optimization pipeline completed")

    def _save_model(self, model: nn.Module, path: Path):
        """Save optimized model"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'optimization_info': {
                'timestamp': time.time(),
                'device': str(self.device),
                'torch_version': torch.__version__
            }
        }, path)
        logger.info(f"Model saved to {path}")

    def _export_onnx(self, output_path: Path):
        """Export model to ONNX format"""
        logger.info("Exporting to ONNX...")

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        logger.info(f"ONNX model exported to {output_path}")

    def _optimize_tensorrt(self, output_dir: Path):
        """Optimize for TensorRT deployment"""
        logger.info("Optimizing for TensorRT...")

        try:
            import torch_tensorrt

            # Convert to TensorRT
            trt_model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
                enabled_precisions={torch_tensorrt.dtype.f16}
            )

            # Save TensorRT model
            torch.jit.save(trt_model, output_dir / 'tensorrt_model.pth')

            logger.info("TensorRT optimization completed")

        except ImportError:
            logger.warning("TensorRT not available, skipping optimization")

class EdgeDeploymentOptimizer:
    """Optimize models for edge deployment"""

    def __init__(self, model: nn.Module, target_platform: str):
        self.model = model
        self.target_platform = target_platform

    def optimize_for_mobile(self):
        """Optimize for mobile deployment"""
        logger.info("Optimizing for mobile deployment...")

        # Use mobile-optimized quantization
        self.model.qconfig = quant.get_default_qconfig('qnnpack')
        quant.prepare(self.model, inplace=True)

        # Convert to quantized model
        self.model = quant.convert(self.model, inplace=True)

        return self.model

    def optimize_for_web(self):
        """Optimize for web deployment (ONNX.js, WebGL)"""
        logger.info("Optimizing for web deployment...")

        # Ensure model is compatible with ONNX.js
        # This would involve specific layer conversions if needed

        return self.model

    def optimize_for_embedded(self):
        """Optimize for embedded systems"""
        logger.info("Optimizing for embedded deployment...")

        # Extreme quantization and pruning for embedded
        quantizer = AdvancedQuantizer(self.model, {'type': 'dynamic'})
        self.model = quantizer.quantize_dynamic()

        pruner = AdvancedPruner(self.model, {'type': 'unstructured', 'amount': 0.5})
        self.model = pruner.apply_unstructured_pruning(0.5)
        pruner.remove_pruning_masks()

        return self.model

def benchmark_model(model: nn.Module, input_shape: Tuple, num_runs: int = 100):
    """Benchmark model performance"""
    logger.info("Benchmarking model performance...")

    model.eval()
    device = next(model.parameters()).device

    # Warmup
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(time.time() - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    logger.info(".4f")
    logger.info(".4f")
    logger.info(".2f")

    return {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'fps': 1.0 / avg_time,
        'model_size_mb': calculate_model_size(model)
    }

def calculate_model_size(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def main():
    """Main optimization script"""
    import argparse

    parser = argparse.ArgumentParser(description='EMOTIA Model Optimization')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--config_path', required=True, help='Path to optimization config')
    parser.add_argument('--output_dir', default='optimized_models', help='Output directory')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarking')

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = ModelOptimizer(args.model_path, args.config_path)
    optimizer.load_model()

    # Run optimization pipeline
    optimizer.optimize_pipeline(args.output_dir)

    # Benchmark if requested
    if args.benchmark:
        results = benchmark_model(optimizer.model, (1, 3, 224, 224))
        with open(Path(args.output_dir) / 'benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Benchmarking completed")

if __name__ == '__main__':
    main()